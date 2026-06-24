# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Op-level precision tests for AMD-FP4 Grouped GEMM (UE5M3 inner scale).

Mirrors the two ``scale_format``-parametrised tests from
:mod:`tests.unittest.nvfp4.test_nvfp_grouped_gemm`:

* ``test_amdfp4_grouped_gemm_autograd``        — full O / dX / dW SNR
  parity vs BF16 reference, K-aware floors;
* ``test_amdfp4_grouped_gemm_recipe_variants_smoke`` — recipe knobs
  (2D-w, outer_scale, Hadamard, DGE, combinations) run end-to-end and
  produce finite outputs + gradients.

Recipe-shared regressions (boundary shapes, non-aligned-M guards,
single-expert == linear, native-dispatch parity, padded buffers) are
not duplicated here: they exercise the shared kernel layer that NVFP4's
``test_nvfp_grouped_gemm.py`` already covers, and the AMD-FP4 path
lands on the *exact same* autograd Function and Triton kernels.
"""

from __future__ import annotations

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.amdfp4 import (
    ALIGN_SIZE_M,
    amdfp4_grouped_gemm,
)
from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr

from amdfp4.utils import calc_cossim, calc_snr, prepare_data


# ---------------------------------------------------------------------------
# Helpers shared across tests (mirrors nvfp4 test helpers)
# ---------------------------------------------------------------------------

def _make_contiguous_expert_indices(
    M_total: int, num_groups: int, num_experts: int, device,
) -> torch.Tensor:
    indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        eid = torch.randint(0, num_experts, (1,), device=device).item()
        s = g * ALIGN_SIZE_M
        indices[s : s + ALIGN_SIZE_M] = eid
    return indices


def _bf16_grouped_ref_forward(
    inputs, expert_weights, expert_indices, M_total, N, num_groups, trans_weights,
):
    dtype = inputs.dtype
    device = inputs.device
    y = torch.zeros(M_total, N, dtype=dtype, device=device)
    for g in range(num_groups):
        s, e = g * ALIGN_SIZE_M, (g + 1) * ALIGN_SIZE_M
        eid = expert_indices[s].item()
        w = expert_weights[eid]
        y[s:e] = inputs[s:e] @ (w.T if trans_weights else w)
    return y


# ---------------------------------------------------------------------------
# Autograd parity (O / dX / dW SNR vs BF16 reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (512, 256, 256, 4),
    (1024, 512, 512, 8),
    (512, 512, 2048, 4),
])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_amdfp4_grouped_gemm_autograd(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type,
):
    """O, dX, dW SNR vs BF16 autograd reference must remain healthy under
    the AMD-FP4 (UE5M3) inner-scale path."""
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")

    inputs_ref = prepare_data((M_total, K), data_type).requires_grad_(True)
    weights_ref = prepare_data((num_experts, N, K), data_type).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device,
    )
    target = prepare_data((M_total, N), data_type)

    y_ref = _bf16_grouped_ref_forward(
        inputs_ref, weights_ref, expert_indices, M_total, N, num_groups,
        trans_weights=True,
    )
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)
    loss_ref.backward()
    dx_ref = inputs_ref.grad.clone()
    dw_ref = weights_ref.grad.clone()

    inputs = inputs_ref.detach().clone().requires_grad_(True)
    weights = weights_ref.detach().clone().requires_grad_(True)

    y = amdfp4_grouped_gemm(
        inputs, weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
    )
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    o_snr = calc_snr(y_ref.detach(), y.detach())
    dx_snr = calc_snr(dx_ref, inputs.grad)
    dw_snr = calc_snr(dw_ref, weights.grad)
    o_sim = calc_cossim(y_ref.detach(), y.detach())
    dx_sim = calc_cossim(dx_ref, inputs.grad)
    dw_sim = calc_cossim(dw_ref, weights.grad)

    print()
    print(tabulate(
        [
            ["O",  f"{o_snr:.2f}",  f"{o_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR(dB)", "CosSim"],
        tablefmt="github",
    ))

    check_nvfp4_autograd_snr(
        {"O": o_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=use_sr_grad,
        kind="nvfp4_grouped_gemm",  # shared K-aware floors
        context=(
            f"AMDFP4GroupedGEMM shape={shape} dtype={data_type} "
            f"x_2d={use_2dblock_x} w_2d={use_2dblock_w}"
        ),
    )

    if not use_sr_grad and data_type == torch.bfloat16:
        fwd_min_snr = 10 if not (use_2dblock_x or use_2dblock_w) else 5
        assert o_snr > fwd_min_snr, (
            f"Forward SNR too low on non-SR BF16 path: {o_snr:.2f} "
            f"(min {fwd_min_snr})"
        )
        assert o_sim > 0.95, f"Forward CosSim too low: {o_sim:.6f}"


# ---------------------------------------------------------------------------
# Recipe-variants smoke (2D-w, outer_scale, Hadamard, DGE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_2dblock_x,use_2dblock_w,use_outer_scale,use_hadamard,use_dge", [
    (False, False, False, False, False),
    (False, True,  False, False, False),
    (False, True,  True,  False, False),
    (False, True,  False, True,  False),
    (False, True,  False, False, True),
    (False, True,  False, True,  True),
])
def test_amdfp4_grouped_gemm_recipe_variants_smoke(
    use_2dblock_x, use_2dblock_w, use_outer_scale, use_hadamard, use_dge,
):
    """Grouped recipe variants must run end-to-end on the AMD-FP4 path."""
    M_total, N, K, num_experts = 512, 256, 256, 4
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16
    inputs = prepare_data((M_total, K), dtype).requires_grad_(True)
    expert_weights = prepare_data((num_experts, N, K), dtype).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(M_total, num_groups, num_experts, device)
    target = prepare_data((M_total, N), dtype)

    y = amdfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=True,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=True,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    assert y.shape == (M_total, N)
    assert y.dtype == dtype
    assert torch.isfinite(y).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert expert_weights.grad is not None and torch.isfinite(expert_weights.grad).all()
