# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Outer-block integration tests for the NVFP4 Linear autograd path."""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr
from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _to_nvfp4_then_scaled_mm,
)
from .utils import prepare_data, calc_snr, calc_cossim


OUTER_CASES = [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True, id="inner_1x16_outer_128x128"),
    pytest.param(True, False, id="inner_16x16_outer_1x128"),
    pytest.param(True, True, id="inner_16x16_outer_128x128"),
]


def _run_linear_autograd_case(
    *,
    shape: tuple[int, int, int, int],
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_outer_2dblock_x: bool,
    use_outer_2dblock_w: bool,
    use_sr_grad: bool,
) -> None:
    B, M, N, K = shape
    dtype = torch.bfloat16
    inputs = prepare_data((B, M, K), dtype).requires_grad_(True)
    weights = prepare_data((N, K), dtype).requires_grad_(True)
    target = prepare_data((B, M, N), dtype)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_()
    weights.grad.zero_()

    outputs = _to_nvfp4_then_scaled_mm(
        inputs,
        weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=False,
        use_outer_block_scale=True,
        use_outer_2dblock_x=use_outer_2dblock_x,
        use_outer_2dblock_w=use_outer_2dblock_w,
        outer_block_size=128,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs_ref, outputs)
    dx_snr = calc_snr(grad_inputs_ref, inputs.grad)
    dw_snr = calc_snr(grad_weights_ref, weights.grad)
    output_sim = calc_cossim(outputs_ref, outputs)
    dx_sim = calc_cossim(grad_inputs_ref, inputs.grad)
    dw_sim = calc_cossim(grad_weights_ref, weights.grad)

    print()
    print(tabulate(
        [
            ["O", f"{output_snr:.2f}", f"{output_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR", "Cosine Sim"],
        tablefmt="github",
    ))

    check_nvfp4_autograd_snr(
        {"O": output_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=use_sr_grad,
        kind="nvfp4_linear",
        context=(
            f"NVFP4Linear outer-block shape={shape} "
            f"inner_x_2d={use_2dblock_x} inner_w_2d={use_2dblock_w} "
            f"outer_x_2d={use_outer_2dblock_x} outer_w_2d={use_outer_2dblock_w}"
        ),
    )


@pytest.mark.parametrize("use_2dblock,use_outer_2dblock", OUTER_CASES)
@pytest.mark.parametrize("use_sr_grad", [False, True])
def test_nvfp4_linear_outer_block_autograd(use_2dblock, use_outer_2dblock, use_sr_grad):
    """Outer-block Linear should satisfy the existing K-aware SNR contract."""
    _run_linear_autograd_case(
        shape=(1, 512, 384, 128),
        use_2dblock_x=use_2dblock,
        use_2dblock_w=use_2dblock,
        use_outer_2dblock_x=use_outer_2dblock,
        use_outer_2dblock_w=use_outer_2dblock,
        use_sr_grad=use_sr_grad,
    )


@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_outer_2dblock_x", [False, True])
@pytest.mark.parametrize("use_outer_2dblock_w", [False, True])
def test_nvfp4_linear_outer_block_independent_xw_layouts(
    use_2dblock_x,
    use_2dblock_w,
    use_outer_2dblock_x,
    use_outer_2dblock_w,
):
    """X and W inner/outer layout knobs must be independently usable."""
    _run_linear_autograd_case(
        shape=(1, 128, 128, 128),
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_outer_2dblock_x=use_outer_2dblock_x,
        use_outer_2dblock_w=use_outer_2dblock_w,
        use_sr_grad=False,
    )


def test_nvfp4_linear_outer_block_large_k_sr_stress():
    """Keep one K=2048 SR stress case without exploding the test matrix."""
    _run_linear_autograd_case(
        shape=(4, 1024, 1024, 2048),
        use_2dblock_x=True,
        use_2dblock_w=True,
        use_outer_2dblock_x=True,
        use_outer_2dblock_w=True,
        use_sr_grad=True,
    )


def test_nvfp4_linear_outer_block_rejects_dge():
    """Outer-block + DGE needs raw-scale plumbing that is not implemented yet."""
    x = prepare_data((1, 128, 128), torch.bfloat16)
    w = prepare_data((128, 128), torch.bfloat16)
    with pytest.raises(RuntimeError, match="does not yet support DGE"):
        NVFP4LinearFunction.apply(
            x,
            w,
            False,
            False,
            False,
            False,
            None,
            True,
            True,
            False,
            False,
            128,
        )


# ---------------------------------------------------------------------------
# T1 / T3 (NVFP4_Outer_Block_Review.md §3.3): direct outer-block vs PTS
# dominance tests on a paper-realistic LLM-activation distribution
# (lognormal heavy-tailed + ~1% massive channels).
# ---------------------------------------------------------------------------


def _run_linear_one_pass(
    *,
    inputs: torch.Tensor,
    weights: torch.Tensor,
    use_outer_block_scale: bool,
    use_outer_scale: bool,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = False,
    use_outer_2dblock_w: bool = False,
    align_x_forward_wgrad: bool = False,
    use_dx_rht: bool = False,
    use_sr_grad: bool = False,
) -> torch.Tensor:
    return _to_nvfp4_then_scaled_mm(
        inputs,
        weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_outer_block_scale=use_outer_block_scale,
        use_outer_2dblock_x=False,
        use_outer_2dblock_w=use_outer_2dblock_w,
        outer_block_size=128,
        align_x_forward_wgrad=align_x_forward_wgrad,
        use_dx_rht=use_dx_rht,
    )


def test_outer_block_snr_dominates_pts_on_lognormal():
    """T3: outer-block must strictly dominate PTS on heavy-tailed activations.

    Uses the lognormal-channel-outlier pattern (LLM-activation proxy).
    Asserts both:
      (i) outer-block forward output SNR ≥ PTS forward output SNR,
     (ii) outer-block dX SNR ≥ PTS dX SNR,
    on identical inputs/weights with a meaningful margin (≥ 1 dB).
    """
    dtype = torch.bfloat16
    M, N, K = 256, 128, 128
    inputs = prepare_data(
        (1, M, K), dtype, pattern="lognormal_channel_outlier", seed=7,
    ).requires_grad_(True)
    weights = prepare_data(
        (N, K), dtype, pattern="lognormal_channel_outlier", seed=13,
    ).requires_grad_(True)
    target = prepare_data((1, M, N), dtype, pattern="random", seed=23)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    # PTS path
    outputs_pts = _run_linear_one_pass(
        inputs=inputs, weights=weights,
        use_outer_block_scale=False, use_outer_scale=True,
    )
    loss_pts = torch.nn.functional.mse_loss(outputs_pts, target)
    loss_pts.backward()
    pts_o_snr = calc_snr(outputs_ref, outputs_pts)
    pts_dx_snr = calc_snr(grad_inputs_ref, inputs.grad)
    inputs.grad.zero_(); weights.grad.zero_()

    # Outer-block path
    outputs_ob = _run_linear_one_pass(
        inputs=inputs, weights=weights,
        use_outer_block_scale=True, use_outer_scale=False,
        use_outer_2dblock_w=True,  # 2D outer for W (axis-invariant)
    )
    loss_ob = torch.nn.functional.mse_loss(outputs_ob, target)
    loss_ob.backward()
    ob_o_snr = calc_snr(outputs_ref, outputs_ob)
    ob_dx_snr = calc_snr(grad_inputs_ref, inputs.grad)

    # Margins below are conservative; on a typical heavy-tailed activation
    # we observe outer-block ≥ PTS by 3-8 dB.
    assert ob_o_snr >= pts_o_snr + 1.0, (
        f"outer-block forward output SNR ({ob_o_snr:.2f}) did not "
        f"dominate PTS ({pts_o_snr:.2f}) by >=1 dB on lognormal pattern."
    )
    assert ob_dx_snr >= pts_dx_snr + 1.0, (
        f"outer-block dX SNR ({ob_dx_snr:.2f}) did not dominate PTS "
        f"({pts_dx_snr:.2f}) by >=1 dB on lognormal pattern."
    )


def test_outer_block_paper_recipe_meets_tighter_threshold():
    """T2: outer-block paper recipe must satisfy the tighter `nvfp4_linear_outer_block`
    threshold (+2 dB over `nvfp4_linear`).

    Covers the paper-recipe configuration (1×16 inner W, 1×128 outer X,
    2D outer W, X̂ align, no Hadamard, SR off).
    """
    B, M, N, K = 1, 512, 384, 128
    dtype = torch.bfloat16
    inputs = prepare_data((B, M, K), dtype).requires_grad_(True)
    weights = prepare_data((N, K), dtype).requires_grad_(True)
    target = prepare_data((B, M, N), dtype)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    outputs = _run_linear_one_pass(
        inputs=inputs, weights=weights,
        use_outer_block_scale=True, use_outer_scale=False,
        use_outer_2dblock_w=True,
        align_x_forward_wgrad=True,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    check_nvfp4_autograd_snr(
        {
            "O": calc_snr(outputs_ref, outputs),
            "dX": calc_snr(grad_inputs_ref, inputs.grad),
            "dW": calc_snr(grad_weights_ref, weights.grad),
        },
        K=K, use_sr_grad=False,
        kind="nvfp4_linear_outer_block",
        context="outer-block paper-recipe paths must clear the tighter SNR contract",
    )


# ---------------------------------------------------------------------------
# T6 (NVFP4_Outer_Block_Review.md §3.3): Hadamard × outer-block SNR coverage.
# Existing tests only cover Hadamard × outer-block via outer-scale-share
# reduction counting; add an SNR-level check for the M-axis wgrad RHT
# (use_hadamard=True) combination.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_2dblock,use_outer_2dblock", OUTER_CASES)
def test_outer_block_with_hadamard_autograd(use_2dblock, use_outer_2dblock):
    """Outer-block × M-axis wgrad RHT (legacy use_hadamard) must clear the
    base nvfp4_linear SNR contract."""
    B, M, N, K = 1, 512, 384, 128
    dtype = torch.bfloat16
    inputs = prepare_data((B, M, K), dtype).requires_grad_(True)
    weights = prepare_data((N, K), dtype).requires_grad_(True)
    target = prepare_data((B, M, N), dtype)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    outputs = _to_nvfp4_then_scaled_mm(
        inputs, weights,
        use_2dblock_x=use_2dblock, use_2dblock_w=use_2dblock,
        use_sr_grad=False, use_outer_scale=False,
        use_hadamard=True,
        use_outer_block_scale=True,
        use_outer_2dblock_x=use_outer_2dblock,
        use_outer_2dblock_w=use_outer_2dblock,
        outer_block_size=128,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()
    check_nvfp4_autograd_snr(
        {
            "O": calc_snr(outputs_ref, outputs),
            "dX": calc_snr(grad_inputs_ref, inputs.grad),
            "dW": calc_snr(grad_weights_ref, weights.grad),
        },
        K=K, use_sr_grad=False,
        kind="nvfp4_linear",
        context=(
            f"outer-block + use_hadamard (M-axis wgrad RHT) "
            f"inner_2d={use_2dblock} outer_2d={use_outer_2dblock}"
        ),
    )


# ---------------------------------------------------------------------------
# P0-2 / P0-4 feature-flag tests.
# ---------------------------------------------------------------------------


def test_align_x_forward_wgrad_skips_axis0_x_qdq_when_m_not_divisible():
    """P0-2: with align_x_forward_wgrad=True the M-axis divisibility check
    must no longer fire (axis=0 X QDQ is skipped entirely)."""
    # M = batch * seq = 1 * 24 = 24, which is NOT divisible by
    # BLOCK_SIZE_DEFAULT (16).  Without align the wgrad-axis QDQ raises;
    # with align we should sail through.
    dtype = torch.bfloat16
    inputs = prepare_data((1, 24, 128), dtype).requires_grad_(True)
    weights = prepare_data((128, 128), dtype).requires_grad_(True)
    target = prepare_data((1, 24, 128), dtype)

    outputs = _run_linear_one_pass(
        inputs=inputs, weights=weights,
        use_outer_block_scale=True, use_outer_scale=False,
        use_outer_2dblock_w=True,
        align_x_forward_wgrad=True,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(weights.grad).all()


def test_align_x_forward_wgrad_requires_outer_block():
    """P0-2: align flag is paper-recipe-specific and must reject non-outer-block."""
    inputs = prepare_data((1, 32, 128), torch.bfloat16)
    weights = prepare_data((128, 128), torch.bfloat16)
    with pytest.raises(RuntimeError, match="align_x_forward_wgrad"):
        # Use_2dblock_x=False, outer_block off, align on → should fail.
        NVFP4LinearFunction.apply(
            inputs, weights,
            False,  # use_2dblock_x
            False,  # use_2dblock_w
            False,  # use_sr_grad
            False,  # use_outer_scale
            None,   # hadamard_transform
            False,  # use_dge
            False,  # use_outer_block_scale  ← off
            False, False, 128,
            True,   # align_x_forward_wgrad  ← on
            None,   # dx_rht_transform
        )


def test_align_x_forward_wgrad_incompatible_with_hadamard():
    """P0-2: M-axis Hadamard and X̂ align are mutually exclusive."""
    from alto.kernels.hadamard_transform import HadamardFactory
    inputs = prepare_data((1, 32, 128), torch.bfloat16)
    weights = prepare_data((128, 128), torch.bfloat16)
    h = HadamardFactory.create_transform(device=inputs.device)
    with pytest.raises(RuntimeError, match="incompatible with M-axis Hadamard"):
        NVFP4LinearFunction.apply(
            inputs, weights,
            False, False, False, False,
            h,      # hadamard_transform  ← on
            False,
            True,   # use_outer_block_scale
            False, False, 128,
            True,   # align_x_forward_wgrad  ← on
            None,
        )


def test_dx_rht_runs_and_matches_unrotated_in_expectation():
    """P0-4: dX RHT path must produce finite gradients with reasonable SNR.

    The N-axis rotation preserves ``(g @ H) @ (H^T @ W) = g @ W`` modulo
    NVFP4 noise; over many rotations the unbiased SR noise averages out.
    For a single deterministic forward we just require the SNR to clear
    the regular nvfp4_linear contract — the rotation can only redistribute
    outliers, not bias the gradient.
    """
    B, M, N, K = 1, 512, 384, 128
    dtype = torch.bfloat16
    inputs = prepare_data((B, M, K), dtype).requires_grad_(True)
    weights = prepare_data((N, K), dtype).requires_grad_(True)
    target = prepare_data((B, M, N), dtype)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    outputs = _run_linear_one_pass(
        inputs=inputs, weights=weights,
        use_outer_block_scale=True, use_outer_scale=False,
        use_outer_2dblock_w=True,
        align_x_forward_wgrad=True,
        use_dx_rht=True,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()
    check_nvfp4_autograd_snr(
        {
            "O": calc_snr(outputs_ref, outputs),
            "dX": calc_snr(grad_inputs_ref, inputs.grad),
            "dW": calc_snr(grad_weights_ref, weights.grad),
        },
        K=K, use_sr_grad=False,
        kind="nvfp4_linear",  # rotation noise is independent of inner block;
                              # keep base contract until paper-recipe arms can
                              # be benchmarked end-to-end.
        context="outer-block + dX RHT (N-axis grad_output + H^T @ W)",
    )


def test_dx_rht_requires_outer_block_via_modifier():
    """P0-4: yaml-level validator must reject dx_rht without outer-block."""
    from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
    with pytest.raises(ValueError, match='use_dx_rht=True requires'):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            targets=["Linear"],
            two_level_scaling="tensorwise",
            use_dx_rht=True,
        )
