# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""toy MoE training sanity for mxfp8 grouped GEMM (PLAN.md §7).

Single-step autograd correctness is already covered in
test_mxfp8_grouped_gemm.py. This file validates the §0 assumption that
V1 all-e4m3 is good enough to *train* a toy MoE for ~100 steps without diverging:
loss must stay finite, trend down, and track a bf16 baseline run with the same
init / data / routing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required.")
from alto.kernels.mxfp8.mxfp8_grouped_gemm import mxfp8_grouped_gemm
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M

from .utils import prepare_data, make_indices


def _bf16_grouped_matmul(inputs, expert_weights, indices):
    """bf16 reference for Y = X @ W[expert]^T, per contiguous group."""
    m_total, n_dim = inputs.shape[0], expert_weights.shape[1]
    out = torch.zeros((m_total, n_dim), dtype=inputs.dtype, device=inputs.device)
    for start in range(0, m_total, ALIGN_SIZE_M):
        end = start + ALIGN_SIZE_M
        out[start:end] = inputs[start:end] @ expert_weights[indices[start].item()].t()
    return out


def _train(grouped_fn, inputs, w_init, indices, target, steps, lr):
    """Run `steps` of SGD fitting W so grouped_fn(X, W) -> target. Returns loss list."""
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


def _window_mean(xs, frac=0.2):
    w = max(1, int(len(xs) * frac))
    return sum(xs[:w]) / w, sum(xs[-w:]) / w


@pytest.mark.parametrize("num_experts", [2, 4])
def test_toy_moe_trains_and_tracks_bf16(num_experts):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_groups, n_dim, k_dim = 4, 128, 128
    m_total = num_groups * ALIGN_SIZE_M
    steps, lr = 100, 0.5

    inputs = prepare_data((m_total, k_dim), dtype)
    target = prepare_data((m_total, n_dim), dtype)
    w_init = prepare_data((num_experts, n_dim, k_dim), dtype) * 0.1
    indices = make_indices(num_groups, num_experts, device)

    mxfp8_losses = _train(
        lambda x, w, idx: mxfp8_grouped_gemm(x, w, idx, trans_weights=True),
        inputs, w_init, indices, target, steps, lr,
    )
    bf16_losses = _train(_bf16_grouped_matmul, inputs, w_init, indices, target, steps, lr)

    # 1. Finite throughout — the core "does e4m3 blow up" check.
    assert all(torch.isfinite(torch.tensor(l)) for l in mxfp8_losses), \
        f"mxfp8 loss went non-finite: {mxfp8_losses}"

    # 2. Loss trends down (robust to per-step quant jitter: compare window means).
    head, tail = _window_mean(mxfp8_losses)
    assert tail < head * 0.9, f"mxfp8 loss did not trend down: head={head:.4f} tail={tail:.4f}"

    # 3. Tracks the bf16 baseline — same init/data/routing, so curves should be same shape.
    bf16_head, bf16_tail = _window_mean(bf16_losses)
    assert tail < bf16_tail * 2.0, \
        f"mxfp8 final loss diverged from bf16: mxfp8_tail={tail:.4f} bf16_tail={bf16_tail:.4f}"
