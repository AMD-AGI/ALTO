# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Outer-block scaling tests for the NVFP4 MoE Grouped GEMM autograd path.

These mirror two existing suites:

* ``test_nvfp_grouped_gemm.py`` — grouped-GEMM scaffolding (BF16 per-expert
  reference, ALIGN_SIZE_M=128 contiguous expert blocks, ``calc_snr`` +
  ``check_nvfp4_autograd_snr`` usage, M_total % 128 == 0 contract).
* ``test_nvfp_outer_block_linear.py`` — the dense outer-block patterns
  (autograd SNR contract, outer-block-vs-tensorwise dominance on the
  heavy-tail ``lognormal_channel_outlier`` data, rejection guards).

Grouped outer-block recipe (see ``NVFP4GroupedGEMM.forward``):
  * activations always use a 1D outer block (1 x ``outer_block_size`` over K);
  * weights use ``use_outer_2dblock_w`` (per-expert 2D tiles) or 1D otherwise;
  * tensorwise (``use_outer_scale``) and outer-block scaling are mutually
    exclusive, and DGE is unsupported together with outer-block scaling.

Test layers
-----------
1. ``test_nvfp4_grouped_gemm_outer_block_autograd``
   Forward O + dX + dW SNR vs a BF16 per-expert autograd reference, swept over
   inner 1D/2D x outer 1D/2D weight layouts.  SR off for determinism.
2. ``test_nvfp4_grouped_gemm_outer_block_finite_shapes``
   Outputs/grads finite with the documented shapes ([M_total,K] dX, [E,N,K] dW).
3. ``test_nvfp4_grouped_gemm_outer_block_dominates_tensorwise``
   On a per-expert heavy-tail input the grouped outer-block forward SNR must not
   regress against the tensorwise path (and typically wins by several dB).
4. ``test_nvfp4_grouped_gemm_outer_block_rejects_dge`` /
   ``test_nvfp4_grouped_gemm_outer_block_rejects_tensorwise``
   Contract guards for the mutually-exclusive / unsupported combinations.
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import (
    ALIGN_SIZE_M,
    nvfp4_grouped_gemm,
)
from .utils import prepare_data, calc_snr, calc_cossim


# Default outer block size used throughout; the 2D-outer-W grid needs
# N, K > OUTER_BLOCK_SIZE so the per-expert tile holds more than one block.
OUTER_BLOCK_SIZE = 128


# (inner_2d, outer_2d_w) — activations always use a 1D outer block, so only the
# weight outer layout has a 2D variant in the grouped API.
OUTER_CASES = [
    pytest.param(False, False, id="inner_1d_outer_w_1d"),
    pytest.param(False, True, id="inner_1d_outer_w_2d"),
    pytest.param(True, False, id="inner_2d_outer_w_1d"),
    pytest.param(True, True, id="inner_2d_outer_w_2d"),
]


# ---------------------------------------------------------------------------
# Helpers (self-contained copies of the grouped-GEMM scaffolding so this file
# does not couple to another test module).
# ---------------------------------------------------------------------------

def _make_contiguous_expert_indices(
    M_total: int, num_groups: int, num_experts: int, device, *, seed: int = 0,
) -> torch.Tensor:
    """Expert-index tensor where every ALIGN_SIZE_M-row group shares one expert.

    Assigns experts round-robin so every expert is exercised when
    ``num_groups >= num_experts`` (deterministic given ``num_groups``).
    """
    indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        eid = (g + seed) % num_experts
        s = g * ALIGN_SIZE_M
        indices[s : s + ALIGN_SIZE_M] = eid
    return indices


def _bf16_grouped_ref_forward(
    inputs, expert_weights, expert_indices, M_total, N, num_groups,
):
    """BF16 per-expert reference forward: y[group] = x[group] @ w[eid].T."""
    dtype = inputs.dtype
    device = inputs.device
    y = torch.zeros(M_total, N, dtype=dtype, device=device)
    for g in range(num_groups):
        s, e = g * ALIGN_SIZE_M, (g + 1) * ALIGN_SIZE_M
        eid = expert_indices[s].item()
        y[s:e] = inputs[s:e] @ expert_weights[eid].T
    return y


# ---------------------------------------------------------------------------
# Test 1 – autograd (O, dX, dW) SNR contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inner_2d,outer_2d_w", OUTER_CASES)
def test_nvfp4_grouped_gemm_outer_block_autograd(inner_2d, outer_2d_w):
    """Grouped outer-block forward + backward SNR vs BF16 per-expert reference.

    Swept over inner 1D/2D scaling and 1D/2D outer-W tiles.  SR is off so the
    comparison is deterministic and the K-aware floor is meaningful.
    """
    M_total, N, K, num_experts = 512, 256, 256, 4
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs_ref = prepare_data((M_total, K), dtype).requires_grad_(True)
    weights_ref = prepare_data((num_experts, N, K), dtype).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device,
    )
    target = prepare_data((M_total, N), dtype)

    y_ref = _bf16_grouped_ref_forward(
        inputs_ref, weights_ref, expert_indices, M_total, N, num_groups,
    )
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)
    loss_ref.backward()
    dx_ref = inputs_ref.grad.clone()
    dw_ref = weights_ref.grad.clone()

    inputs = inputs_ref.detach().clone().requires_grad_(True)
    weights = weights_ref.detach().clone().requires_grad_(True)

    y = nvfp4_grouped_gemm(
        inputs, weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=inner_2d,
        use_2dblock_w=inner_2d,
        use_sr_grad=False,
        use_outer_scale=False,
        use_outer_block_scale=True,
        use_outer_2dblock_w=outer_2d_w,
        outer_block_size=OUTER_BLOCK_SIZE,
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
            ["O", f"{o_snr:.2f}", f"{o_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR(dB)", "CosSim"],
        tablefmt="github",
    ))

    check_nvfp4_autograd_snr(
        {"O": o_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=False,
        # No grouped-outer-block-specific tier exists; outer-block scaling is
        # mutually exclusive with use_outer_scale, so the inner-only grouped
        # floor is the applicable K-aware contract here.
        kind="nvfp4_grouped_gemm",
        use_outer_scale=False,
        context=(
            f"NVFP4 grouped outer-block inner_2d={inner_2d} "
            f"outer_2d_w={outer_2d_w} K={K}"
        ),
    )

    assert o_sim > 0.95, f"Forward CosSim too low: {o_sim:.6f}"


# ---------------------------------------------------------------------------
# Test 2 – finiteness + documented shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inner_2d,outer_2d_w", OUTER_CASES)
def test_nvfp4_grouped_gemm_outer_block_finite_shapes(inner_2d, outer_2d_w):
    """Forward output and gradients are finite with the documented shapes."""
    M_total, N, K, num_experts = 384, 256, 256, 3
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((M_total, K), dtype).requires_grad_(True)
    expert_weights = prepare_data((num_experts, N, K), dtype).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device,
    )
    target = prepare_data((M_total, N), dtype)

    y = nvfp4_grouped_gemm(
        inputs, expert_weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=inner_2d,
        use_2dblock_w=inner_2d,
        use_sr_grad=False,
        use_outer_scale=False,
        use_outer_block_scale=True,
        use_outer_2dblock_w=outer_2d_w,
        outer_block_size=OUTER_BLOCK_SIZE,
    )
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    assert y.shape == (M_total, N), f"Bad output shape: {y.shape}"
    assert y.dtype == dtype
    assert torch.isfinite(y).all(), "Output not finite"
    assert inputs.grad is not None
    assert inputs.grad.shape == (M_total, K), f"Bad dX shape: {inputs.grad.shape}"
    assert torch.isfinite(inputs.grad).all(), "dX not finite"
    assert expert_weights.grad is not None
    assert expert_weights.grad.shape == (num_experts, N, K), (
        f"Bad dW shape: {expert_weights.grad.shape}"
    )
    assert torch.isfinite(expert_weights.grad).all(), "dW not finite"


# ---------------------------------------------------------------------------
# Test 3 – outer-block vs tensorwise on a heavy-tail / per-expert-outlier input
# ---------------------------------------------------------------------------

def test_nvfp4_grouped_gemm_outer_block_dominates_tensorwise():
    """Grouped outer-block forward SNR must not regress against tensorwise on a
    heavy-tail, per-expert-outlier activation/weight distribution.

    The tensorwise (``use_outer_scale``) path shares a single FP32 scale across
    the whole tensor (and, for weights, across *all* experts), so a hot expert
    or massive channel forces the ~99% well-behaved blocks into E4M3 inner-scale
    underflow.  The per-outer-block FP32 scale localises each outlier, so the
    rest of the tensor keeps precision.  We assert no regression with a loose
    -0.5 dB margin (outer-block typically wins by several dB here, but we keep
    the assertion robust rather than over-tight).
    """
    M_total, N, K, num_experts = 512, 256, 512, 4
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1e5 outliers push the shared tensorwise scale into E4M3 inner-scale
    # underflow for the non-outlier blocks — the regime outer-block targets.
    inputs = prepare_data(
        (M_total, K), dtype, pattern="lognormal_channel_outlier", seed=7,
        outlier_scale=1e5,
    )
    weights = prepare_data(
        (num_experts, N, K), dtype, pattern="lognormal_channel_outlier", seed=13,
        outlier_scale=1e5,
    )
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device,
    )

    y_ref = _bf16_grouped_ref_forward(
        inputs, weights, expert_indices, M_total, N, num_groups,
    )

    y_tw = nvfp4_grouped_gemm(
        inputs, weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False,
        use_outer_scale=True,
        use_outer_block_scale=False,
    )
    y_ob = nvfp4_grouped_gemm(
        inputs, weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False,
        use_outer_scale=False,
        use_outer_block_scale=True,
        use_outer_2dblock_w=True,
        outer_block_size=OUTER_BLOCK_SIZE,
    )

    tw_snr = calc_snr(y_ref, y_tw)
    ob_snr = calc_snr(y_ref, y_ob)

    print(
        f"\nheavy-tail grouped forward SNR: tensorwise={tw_snr:.2f} dB  "
        f"outer-block={ob_snr:.2f} dB  (margin={ob_snr - tw_snr:+.2f} dB)"
    )

    assert torch.isfinite(y_tw).all()
    assert torch.isfinite(y_ob).all()
    assert ob_snr >= tw_snr - 0.5, (
        f"grouped outer-block forward SNR ({ob_snr:.2f}) regressed vs "
        f"tensorwise ({tw_snr:.2f}) on heavy-tail per-expert-outlier data."
    )


# ---------------------------------------------------------------------------
# Test 4 – rejection guards
# ---------------------------------------------------------------------------

def _small_grouped_args(device, dtype, *, num_experts=2, N=256, K=256):
    M_total = ALIGN_SIZE_M
    inputs = prepare_data((M_total, K), dtype)
    expert_weights = prepare_data((num_experts, N, K), dtype)
    expert_indices = _make_contiguous_expert_indices(
        M_total, M_total // ALIGN_SIZE_M, num_experts, device,
    )
    return inputs, expert_weights, expert_indices


def test_nvfp4_grouped_gemm_outer_block_rejects_dge():
    """DGE together with outer-block scaling is unsupported and must raise."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    inputs, expert_weights, expert_indices = _small_grouped_args(device, dtype)
    with pytest.raises(RuntimeError, match="DGE"):
        nvfp4_grouped_gemm(
            inputs, expert_weights, expert_indices,
            trans_weights=True,
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False,
            use_outer_block_scale=True,
            use_dge=True,
        )


def test_nvfp4_grouped_gemm_outer_block_rejects_tensorwise():
    """Tensorwise (``use_outer_scale``) and outer-block scaling are mutually
    exclusive and must raise when both are requested."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    inputs, expert_weights, expert_indices = _small_grouped_args(device, dtype)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        nvfp4_grouped_gemm(
            inputs, expert_weights, expert_indices,
            trans_weights=True,
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False,
            use_outer_scale=True,
            use_outer_block_scale=True,
        )
