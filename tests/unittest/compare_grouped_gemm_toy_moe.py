# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Toy-MoE training comparison across low-precision grouped GEMM kernels.

Extends the mxfp8 toy-MoE sanity (mxfp8/test_e2e_moe.py, PLAN.md §7) to mxfp4
and nvfp4 using the *identical* config (routing, init, data, lr, steps) so all
three low-precision paths can be compared against the same bf16 baseline on one
plot.

The three kernels share the same entry-point shape:
    fn(inputs[M_total,K], expert_weights[E,N,K], expert_indices[M_total],
       trans_weights=True) -> [M_total, N]
with ALIGN_SIZE_M=128 each, so the toy task transfers unchanged.

Determinism note: nvfp4 defaults to use_sr_grad=True (stochastic rounding),
which would make its loss curve noisy/non-reproducible. We force use_sr_grad
=False on every quantized path so all four curves are a fair, repeatable
comparison.

Standalone (no relative imports) so it runs directly:

    python tests/unittest/compare_grouped_gemm_toy_moe.py

Outputs compare_grouped_gemm_toy_moe.png next to this file and prints the full
per-step loss series for every path.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from alto.kernels.mxfp8.mxfp8_grouped_gemm import mxfp8_grouped_gemm
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm import mxfp4_grouped_gemm
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import nvfp4_grouped_gemm

ALIGN_SIZE_M = 128  # shared across mxfp8 / mxfp4 / nvfp4 grouped GEMM


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


# Quantized paths, all forced deterministic (use_sr_grad=False) for fair comparison.
PATHS = {
    "bf16 baseline": _bf16_grouped_matmul,
    "mxfp8 (e4m3)": lambda x, w, idx: mxfp8_grouped_gemm(x, w, idx, trans_weights=True),
    "mxfp4": lambda x, w, idx: mxfp4_grouped_gemm(x, w, idx, trans_weights=True, use_sr_grad=False),
    "nvfp4": lambda x, w, idx: nvfp4_grouped_gemm(x, w, idx, trans_weights=True, use_sr_grad=False),
}
COLORS = {
    "bf16 baseline": "tab:orange",
    "mxfp8 (e4m3)": "tab:blue",
    "mxfp4": "tab:green",
    "nvfp4": "tab:red",
}


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

        series = {}
        for name, fn in PATHS.items():
            try:
                series[name] = _train(fn, inputs, w_init, indices, target, steps, lr)
            except Exception as e:
                # nvfp4's quant kernel has a pre-existing Triton compile bug on
                # CDNA3 (F4_E2M1_MAX not constexpr); skip rather than abort the
                # whole comparison.
                print(f"[skip] {name}: {type(e).__name__}: {str(e).splitlines()[0]}")
                series[name] = None

        print(f"\n===== num_experts={num_experts} =====")
        live = [n for n in PATHS if series[n] is not None]
        skipped = [n for n in PATHS if series[n] is None]
        header = f"{'step':>4} " + " ".join(f"{name:>16}" for name in live)
        print(header)
        for i in range(steps):
            row = f"{i:>4} " + " ".join(f"{series[name][i]:>16.6f}" for name in live)
            print(row)
        if skipped:
            print(f"(unavailable: {', '.join(skipped)})")

        linestyle = {"bf16 baseline": "--"}
        for name in PATHS:
            if series[name] is None:
                continue
            ax.plot(series[name], label=name, color=COLORS[name],
                    linestyle=linestyle.get(name, "-"))
        if skipped:
            ax.plot([], [], " ", label=f"{', '.join(skipped)}: N/A (CDNA3)")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE loss (log)")
        ax.set_title(f"toy MoE, num_experts={num_experts}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "compare_grouped_gemm_toy_moe.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved curve to {out_path}")


if __name__ == "__main__":
    main()
