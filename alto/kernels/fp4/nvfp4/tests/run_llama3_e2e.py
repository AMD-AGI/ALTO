#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Llama3 E2E low-precision training comparison: BF16 vs NVFP4 QDQ.

This script trains a Llama3 debug model (6M params, dim=256, 6 layers) on a
synthetic next-token-prediction task, comparing BF16 baseline with NVFP4 QDQ
(stochastic rounding) Linear replacement.

Usage:
    python run_llama3_e2e.py [--steps 3000] [--output loss_curve.png]
"""

import argparse
import json
import os
import copy

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchtitan.models.llama3 import llama3_configs
from alto.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction


class NVFP4Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 use_2dblock_x=False, use_2dblock_w=False,
                 use_sr_grad=True, use_per_tensor_scale=False):
        super().__init__(in_features, out_features, bias=bias)
        self.use_2dblock_x = use_2dblock_x
        self.use_2dblock_w = use_2dblock_w
        self.use_sr_grad = use_sr_grad
        self.use_per_tensor_scale = use_per_tensor_scale

    def forward(self, x):
        y = NVFP4LinearFunction.apply(
            x, self.weight,
            self.use_2dblock_x, self.use_2dblock_w,
            self.use_sr_grad, self.use_per_tensor_scale,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


def replace_linear_with_nvfp4(model, **kwargs):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            nvfp4_lin = NVFP4Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None, **kwargs,
            )
            nvfp4_lin.weight = module.weight
            if module.bias is not None:
                nvfp4_lin.bias = module.bias
            setattr(model, name, nvfp4_lin)
        else:
            replace_linear_with_nvfp4(module, **kwargs)


def train_loop(model, steps, batch_size, seq_len, vocab_size, lr, device, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        torch.manual_seed(step)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size), target.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 100 == 0 or step == steps - 1:
            print(f"  [{label}] step {step:5d}/{steps}  loss={loss.item():.4f}")

    return losses


def plot_losses(losses_bf16, losses_nvfp4, steps, output_path):
    window = max(1, steps // 100)
    bf16_s = np.convolve(losses_bf16, np.ones(window)/window, mode="valid")
    nvfp4_s = np.convolve(losses_nvfp4, np.ones(window)/window, mode="valid")
    x_axis = np.arange(len(bf16_s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(x_axis, bf16_s, color="#1976D2", linewidth=2, label="BF16 Baseline")
    ax1.plot(x_axis, nvfp4_s, color="#E64A19", linewidth=2, label="NVFP4 QDQ (SR)")
    ax1.set_xlabel("Training Steps", fontsize=13)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax1.set_title(f"Llama3 Debug Model — BF16 vs NVFP4 (SR)\n{steps} steps", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    ratio_s = nvfp4_s / np.maximum(bf16_s, 1e-12)
    ax2.plot(x_axis, ratio_s, color="#4CAF50", linewidth=2)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ratio = 1.0")
    ax2.set_xlabel("Training Steps", fontsize=13)
    ax2.set_ylabel("Loss Ratio (NVFP4 / BF16)", fontsize=13)
    ax2.set_title("Loss Ratio Over Training", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, max(2.0, ratio_s.max() * 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="nvfp4_llama3_e2e.png")
    parser.add_argument("--json_output", type=str, default="nvfp4_llama3_e2e.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(42)

    print("=" * 60)
    print("Llama3 Debug Model E2E Training: BF16 vs NVFP4 (SR)")
    print("=" * 60)

    cfg = llama3_configs["debugmodel"]
    model_bf16 = cfg.build().to(device).bfloat16()
    model_bf16.init_weights(buffer_device=device)
    nparams = sum(p.numel() for p in model_bf16.parameters())
    vocab_size = cfg.vocab_size
    print(f"Model: Llama3 debugmodel (dim=256, 6 layers, {nparams/1e6:.2f}M params)")
    print(f"Vocab: {vocab_size}, Seq len: {args.seq_len}, Batch: {args.batch_size}")
    print(f"Steps: {args.steps}, LR: {args.lr}")

    model_nvfp4 = copy.deepcopy(model_bf16)
    replace_linear_with_nvfp4(model_nvfp4, use_sr_grad=True)

    n_nvfp4 = sum(1 for m in model_nvfp4.modules() if isinstance(m, NVFP4Linear))
    print(f"NVFP4 Linear layers: {n_nvfp4}")
    print()

    print("--- Training BF16 Baseline ---")
    losses_bf16 = train_loop(
        model_bf16, args.steps, args.batch_size, args.seq_len,
        vocab_size, args.lr, device, "BF16",
    )

    print()
    print("--- Training NVFP4 QDQ (SR) ---")
    losses_nvfp4 = train_loop(
        model_nvfp4, args.steps, args.batch_size, args.seq_len,
        vocab_size, args.lr, device, "NVFP4",
    )

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  BF16  — initial: {losses_bf16[0]:.4f}, final: {losses_bf16[-1]:.4f}")
    print(f"  NVFP4 — initial: {losses_nvfp4[0]:.4f}, final: {losses_nvfp4[-1]:.4f}")
    print(f"  Final ratio: {losses_nvfp4[-1] / max(losses_bf16[-1], 1e-12):.4f}")

    with open(args.json_output, "w") as f:
        json.dump({
            "bf16": losses_bf16,
            "nvfp4_sr": losses_nvfp4,
            "config": {
                "model": "llama3_debugmodel",
                "dim": 256, "n_layers": 6, "vocab_size": vocab_size,
                "steps": args.steps, "batch_size": args.batch_size,
                "seq_len": args.seq_len, "lr": args.lr,
            },
        }, f)
    print(f"  Loss data saved to {args.json_output}")

    plot_losses(losses_bf16, losses_nvfp4, args.steps, args.output)


if __name__ == "__main__":
    main()
