# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the NVFP4 linear op.

Two tests, scoped like ``test_mxfp_linear.py`` but adapted to NVFP4's
software-simulated path (there is no native NVFP4 GEMM kernel yet on AMD, so
there is nothing analogous to ``test_mxfp4_linear_kernel`` to add):

* ``test_nvfp4_qdq_roundtrip`` — per-operand QDQ accuracy, the building
  block that ``NVFP4LinearFunction`` stacks together.  Guards the
  single-pass quantise → dequantise behaviour independently of any GEMM.

* ``test_nvfp4_linear_autograd_function`` — MXFP4-parity check, asserting
  the forward output and both gradients (``dX``, ``dW``) track the BF16
  reference within an SNR threshold.

Recipe knobs such as ``use_hadamard`` / ``use_dge`` are not parametrised here
to keep the matrix small; their correctness under the production dispatch
layer is covered by the parametrised ``test_linear_supported_knobs_still_work``
in ``test_nvfp_dispatch_guards.py``.
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr
from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _qdq,
    _to_nvfp4_then_scaled_mm,
)
from .utils import prepare_data, calc_snr, calc_cossim


# ---------------------------------------------------------------------------
# QDQ round-trip: BF16 -> NVFP4 -> BF16 (no GEMM).  Each forward/backward
# reduction axis in ``NVFP4LinearFunction`` feeds off of one such round-trip,
# so this is the narrowest test we can write before layering in the matmul.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(128, 64), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_outer_scale", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_nvfp4_qdq_roundtrip(
    shape, axis, is_2d_block, use_outer_scale, use_sr,
):
    """Verify precision of a single NVFP4 quant -> dequant round-trip.

    The SNR / cosine-similarity floors are picked to be the loosest value
    still consistent with correct behaviour in every parametrised cell,
    so regressions (e.g. an incorrect scale encoding) show up as threshold
    failures rather than as a subtle degradation that slips through.
    """
    x = prepare_data(shape, torch.bfloat16)

    x_qdq = _qdq(
        x, axis=axis,
        is_2d_block=is_2d_block,
        use_outer_scale=use_outer_scale,
        use_sr=use_sr,
    )

    assert x_qdq.shape == x.shape
    assert x_qdq.dtype == x.dtype

    snr = calc_snr(x, x_qdq)
    cossim = calc_cossim(x, x_qdq)

    print()
    print(tabulate(
        [["SNR (dB)", f"{snr:.2f}"], ["Cosine Sim", f"{cossim:.6f}"]],
        headers=["Metric", "Value"], tablefmt="github",
    ))

    # 1D block: dense per-row/col scales -> tight SNR.  2D block: scales
    # are shared across a block_size × block_size tile so precision drops.
    # 2D + SR is the weakest combination (E4M3 block-scale rounding plus
    # stochastic-rounding variance), so its cossim floor is the loosest.
    min_snr = 8 if not is_2d_block else 5
    # 2D block + SR case: E4M3 block scale rounding reduces precision slightly
    # compared to float32 scales.  The SNR threshold (5 dB) is the binding
    # criterion here; cosine similarity is set conservatively at 0.94.
    min_cossim = 0.99 if not is_2d_block else (0.94 if use_sr else 0.99)
    assert snr > min_snr, f"QDQ SNR too low: {snr:.2f}"
    assert cossim > min_cossim, f"QDQ cosine similarity too low: {cossim:.6f}"


# ---------------------------------------------------------------------------
# Full autograd-function parity test.  Equivalent in structure to MXFP4's
# ``test_mxfp4_linear_autograd_function``: given identical inputs and a
# deterministic MSE target, the NVFP4 forward output and both gradients
# (dX, dW) must track a reference BF16 linear within an SNR threshold.
#
# ``use_hadamard`` / ``use_dge`` are intentionally NOT part of this matrix:
# they are wgrad-only refinements that only meaningfully change the dW
# distribution under long training horizons, and their dispatch-level
# integration is covered separately in ``test_nvfp_dispatch_guards.py``.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 64, 64, 64), (1, 512, 384, 128), (4, 1024, 1024, 2048)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_nvfp4_linear_autograd_function(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type,
):
    """Forward + dX + dW must match a BF16 nn.Linear reference in SNR.

    The check is K-aware: large reduction dimensions amplify stochastic
    rounding noise roughly as sqrt(K), so K=2048 stress cases get a lower
    per-tensor hard floor while small/medium cases keep tighter aggregate
    quality checks.
    """
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)
    target = prepare_data((B, M, N), data_type)

    # Reference BF16 / FP32 nn.Linear forward + backward.
    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    # NVFP4 QDQ path.  This is NOT a shared MXFP4/NVFP4 test; it is specific
    # to NVFP4's autograd function.  We still keep
    # ``use_outer_scale=False`` here on purpose so the parity check stays
    # focused on the shared 6-QDQ linear math (forward + dX + dW).  The
    # outer-level scale path is exercised separately in the QDQ round-trip
    # and quantization tests, where it can be isolated without broadening
    # this matrix or coupling the SNR thresholds to an NVFP4-only knob.
    outputs = NVFP4LinearFunction.apply(
        inputs, weights,
        use_2dblock_x, use_2dblock_w, use_sr_grad,
        False,  # use_outer_scale
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(weights.grad, grad_weights_ref)
    dw_sim = calc_cossim(weights.grad, grad_weights_ref)

    print()
    print(tabulate(
        [
            ["O",  f"{output_snr:.2f}", f"{output_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}",     f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}",     f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github",
    ))

    check_nvfp4_autograd_snr(
        {"O": output_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=use_sr_grad,
        kind="nvfp4_linear",
        context=(
            f"NVFP4Linear shape={shape} dtype={data_type} "
            f"x_2d={use_2dblock_x} w_2d={use_2dblock_w}"
        ),
    )


@pytest.mark.parametrize("shape,use_2dblock_x,use_2dblock_w", [
    # Shared with both NVFP4 and MXFP4 linear-autograd test matrices.
    ((1, 512, 384, 128), False, False),
    ((1, 512, 384, 128), False, True),
    ((4, 1024, 1024, 2048), False, False),
])
def test_nvfp4_linear_forward_compares_with_mxfp4_on_shared_cases(
    shape, use_2dblock_x, use_2dblock_w,
):
    """NVFP4 and MXFP4 forward paths should both stay close to BF16 reference.

    This is not an equality test between formats: NVFP4 uses E4M3 block scales
    and MXFP4 uses E8M0 scales, so their quantization errors differ.  The goal
    is to make sure both format families run on representative dense-linear
    cases that already exist in both original test suites and remain in a
    reasonable SNR band versus the same BF16 reference.
    """
    B, M, N, K = shape
    dtype = torch.bfloat16
    x = prepare_data((B, M, K), dtype)
    w = prepare_data((N, K), dtype)
    # Use one BF16 reference for both formats.  The comparison is therefore
    # "NVFP4 vs BF16" and "MXFP4 vs BF16", not "NVFP4 vs MXFP4".
    y_ref = torch.nn.functional.linear(x, w)

    # Keep both recipes deliberately minimal and analogous: 1D block scales,
    # deterministic rounding on the forward path, no clipping, no Hadamard,
    # no DGE, no macro/outer scaling.  This keeps the test focused on whether
    # each format family can run the same dense-linear shape and produce a
    # finite output with reasonable forward error.
    y_nv = _to_nvfp4_then_scaled_mm(
        x,
        w,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
        use_outer_scale=False,
    )
    y_mx = _to_mxfp4_then_scaled_mm(
        x,
        w,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
        use_dge=False,
        clip_mode="none",
        use_hadamard=False,
        use_macro_block_scaling=False,
    )

    nv_snr = calc_snr(y_ref, y_nv)
    mx_snr = calc_snr(y_ref, y_mx)
    print()
    print(tabulate(
        [
            ["NVFP4", f"{nv_snr:.2f}", f"{calc_cossim(y_ref, y_nv):.6f}"],
            ["MXFP4", f"{mx_snr:.2f}", f"{calc_cossim(y_ref, y_mx):.6f}"],
        ],
        headers=["Format", "SNR", "Cosine Sim"], tablefmt="github",
    ))

    assert torch.isfinite(y_nv).all()
    assert torch.isfinite(y_mx).all()
    # The two formats use different scale encodings (NVFP4: E4M3 block scale;
    # MXFP4: E8M0 block scale), so bitwise equality is neither expected nor
    # useful.  A 10 dB forward-SNR floor is intentionally loose enough to avoid
    # making this test a recipe-quality benchmark, but tight enough to catch
    # routing or catastrophic quantization regressions.
    assert nv_snr > 10, f"NVFP4 forward SNR too low: {nv_snr:.2f}"
    assert mx_snr > 10, f"MXFP4 forward SNR too low: {mx_snr:.2f}"


def test_nvfp4_linear_rejects_unaligned_axis0_activation_view():
    """1D wgrad activation QDQ must fail fast when the leading dim is not
    divisible by the NVFP4 block size, rather than silently reusing the fprop
    axis=-1 view and changing the recipe."""
    # Use a non-multiple of 16 that is still > 64 so the Triton tile size stays
    # at BLOCK_M=64 (power-of-2) and the test reaches our runtime guard rather
    # than failing earlier in Triton's shape checks.
    x = prepare_data((150, 32), torch.bfloat16)
    w = prepare_data((32, 32), torch.bfloat16)
    with pytest.raises(RuntimeError, match="activation QDQ requires"):
        _ = NVFP4LinearFunction.apply(
            x, w,
            False,  # use_2dblock_x
            True,   # use_2dblock_w -> keep the weight side aligned so x trips first
            False,
            False,
        )


def test_nvfp4_linear_rejects_unaligned_axis0_weight_view():
    """1D wgrad weight QDQ must fail fast when the leading dim is not
    divisible by the NVFP4 block size, rather than silently reusing the fprop
    axis=-1 view and changing the recipe."""
    x = prepare_data((16, 32), torch.bfloat16)
    w = prepare_data((150, 32), torch.bfloat16)
    with pytest.raises(RuntimeError, match="weight QDQ requires"):
        _ = NVFP4LinearFunction.apply(
            x, w,
            True,   # use_2dblock_x -> keep activation side aligned so w trips first
            False,  # use_2dblock_w
            False,
            False,
        )


# ---------------------------------------------------------------------------
# AMP / autocast dtype cross: forward sees BF16 activations + FP32 weights
# (or vice versa).  Without explicit dtype alignment the saved QDQ tensors
# end up in the weight's dtype while ``grad_output`` arrives in the
# activation's dtype, so the backward matmul fails with
# ``BFloat16 != float``.  This test pins down the contract.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x_dtype,w_dtype", [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
])
def test_nvfp4_linear_backward_handles_dtype_cross(x_dtype, w_dtype):
    x = prepare_data((128, 64), x_dtype).requires_grad_(True)
    w = prepare_data((64, 64), w_dtype).requires_grad_(True)
    y = NVFP4LinearFunction.apply(
        x, w,
        False,  # use_2dblock_x
        False,  # use_2dblock_w
        False,  # use_sr_grad
        False,  # use_outer_scale
    )
    assert y.dtype == x_dtype
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert x.grad.dtype == x_dtype
    assert w.grad.dtype == w_dtype
