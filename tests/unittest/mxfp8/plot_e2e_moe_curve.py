# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Plot toy-MoE training loss curves (mxfp8 vs bf16) for PLAN.md §7.

Standalone helper, not a pytest test. The training loop is identical to
test_e2e_moe.py (same routing, init, lr, steps) so the curves match what the
sanity test asserts on. Self-contained (no relative imports) so it runs directly:

    python tests/unittest/mxfp8/plot_e2e_moe_curve.py

Outputs e2e_moe_loss_curve.png next to this file and prints the full per-step
series for both paths.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from alto.kernels.mxfp8.mxfp8_grouped_gemm import mxfp8_grouped_gemm
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M


def prepare_data(tensor_shape, data_type):
    """Same outlier-injecting init as tests/unittest/mxfp8/utils.py:prepare_data."""
    torch.manual_seed(1234)
    x = torch.randn(tensor_shape, dtype=data_type, device="cuda")
    p_mask = torch.bernoulli(torch.ones_like(x) * 0.005)
    x += 100 * torch.randn_like(x) * p_mask
    return x


def _make_indices(num_groups, num_experts, device):
    indices = torch.empty(num_groups * ALIGN_SIZE_M, dtype=torch.int32, device=device)
    for g in range(num_groups):
        indices[g * ALIGN_SIZE_M:(g + 1) * ALIGN_SIZE_M] = g % num_experts
    return indices


def _bf16_grouped_matmul(inputs, expert_weights, indices):
    m_total, n_dim = inputs.shape[0], expert_weights.shape[1]
    out = torch.zeros((m_total, n_dim), dtype=inputs.dtype, device=inputs.device)
    for start in range(0, m_total, ALIGN_SIZE_M):
        end = start + ALIGN_SIZE_M
        out[start:end] = inputs[start:end] @ expert_weights[indices[start].item()].t()
    return out


def _train(grouped_fn, inputs, w_init, indices, target, steps, lr):
    weights = w_init.clone().detach().requires_grad_(True)
    losses = []
    for _ in range(steps):
        out = grouped_fn(inputs, weights, indices)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        losses.append(loss.item())
        weights.grad = None
        loss.backward()
        with torch.no_grad():
            weights -= lr * weights.grad
    return losses


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_groups, n_dim, k_dim = 4, 128, 128
    m_total = num_groups * ALIGN_SIZE_M
    steps, lr = 100, 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, num_experts in zip(axes, (2, 4)):
        inputs = prepare_data((m_total, k_dim), dtype)
        target = prepare_data((m_total, n_dim), dtype)
        w_init = prepare_data((num_experts, n_dim, k_dim), dtype) * 0.1
        indices = _make_indices(num_groups, num_experts, device)

        mxfp8_losses = _train(
            lambda x, w, idx: mxfp8_grouped_gemm(x, w, idx, trans_weights=True),
            inputs, w_init, indices, target, steps, lr,
        )
        bf16_losses = _train(_bf16_grouped_matmul, inputs, w_init, indices, target, steps, lr)

        print(f"\n===== num_experts={num_experts} =====")
        print(f"{'step':>4} {'mxfp8':>14} {'bf16':>14}")
        for i, (ml, bl) in enumerate(zip(mxfp8_losses, bf16_losses)):
            print(f"{i:>4} {ml:>14.6f} {bl:>14.6f}")

        ax.plot(mxfp8_losses, label="mxfp8 (e4m3)", color="tab:blue")
        ax.plot(bf16_losses, label="bf16 baseline", color="tab:orange", linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE loss (log)")
        ax.set_title(f"toy MoE, num_experts={num_experts}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_moe_loss_curve.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved curve to {out_path}")


if __name__ == "__main__":
    main()
