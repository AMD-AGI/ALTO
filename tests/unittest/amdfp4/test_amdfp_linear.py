# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the AMD-FP4 linear op (UE5M3 inner scale).

Mirrors :mod:`tests.unittest.nvfp4.test_nvfp_linear` but pins the inner
grid to UE5M3 throughout and uses the AMD-FP4 op surface
(``AMDFP4LinearFunction`` / ``_to_amdfp4_then_scaled_mm``).

Two test families:

* ``test_amdfp4_qdq_roundtrip`` -- per-operand QDQ accuracy (the
  building block that ``AMDFP4LinearFunction`` stacks together).
* ``test_amdfp4_linear_autograd_function`` -- forward output and both
  gradients (``dX``, ``dW``) track a BF16 reference within an SNR
  threshold, identical structure to NVFP4's autograd test.

The unaligned-axis guard tests and the BF16/FP32 dtype-cross test are
not duplicated here: the underlying autograd function is *the same
class* as NVFP4's, so those guards are already covered in the NVFP4
suite and any breakage there shows up in both.
"""

from __future__ import annotations

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.amdfp4 import AMDFP4LinearFunction, _to_amdfp4_then_scaled_mm
from alto.kernels.fp4.nvfp4.nvfp_linear import _qdq, _to_nvfp4_then_scaled_mm
from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr

from amdfp4.utils import calc_cossim, calc_snr, prepare_data


# ---------------------------------------------------------------------------
# QDQ round-trip: BF16 -> AMD-FP4 -> BF16 (no GEMM)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(128, 64), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_outer_scale", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_amdfp4_qdq_roundtrip(
    shape, axis, is_2d_block, use_outer_scale, use_sr,
):
    """Verify precision of a single AMD-FP4 quant -> dequant round-trip."""
    x = prepare_data(shape, torch.bfloat16)

    x_qdq = _qdq(
        x, axis=axis,
        is_2d_block=is_2d_block,
        use_outer_scale=use_outer_scale,
        use_sr=use_sr,
        scale_format="ue5m3",
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

    # Same SNR / cossim band as NVFP4 -- UE5M3 has wider dynamic range
    # but the same mantissa width as E4M3, so single-block round-trip
    # error on Gaussian inputs is comparable.  Floors picked so any
    # encoding-level regression trips them.
    min_snr = 8 if not is_2d_block else 5
    min_cossim = 0.99 if not is_2d_block else (0.94 if use_sr else 0.99)
    assert snr > min_snr, f"QDQ SNR too low: {snr:.2f}"
    assert cossim > min_cossim, f"QDQ cosine similarity too low: {cossim:.6f}"


# ---------------------------------------------------------------------------
# Full autograd-function parity test (forward + dX + dW vs BF16 reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 64, 64, 64), (1, 512, 384, 128), (4, 1024, 1024, 2048)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_amdfp4_linear_autograd_function(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type,
):
    """Forward + dX + dW must match a BF16 nn.Linear reference in SNR.

    Same K-aware threshold scheme as NVFP4; the AMD-FP4 path lands on
    the same autograd function and is expected to clear the same
    floors.
    """
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)
    target = prepare_data((B, M, N), data_type)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    outputs = AMDFP4LinearFunction.apply(
        inputs, weights,
        use_2dblock_x, use_2dblock_w, use_sr_grad,
        False,    # use_outer_scale
        None,     # hadamard_transform
        False,    # use_dge
        # ``scale_format`` is the LAST positional arg of the forward (appended
        # after the outer-block params); apply() takes no kwargs, so the
        # intervening outer-block params are filled with their defaults here.
        False,    # use_outer_block_scale
        False,    # use_outer_2dblock_x
        False,    # use_outer_2dblock_w
        128,      # outer_block_size (OUTER_BLOCK_SIZE_DEFAULT)
        "ue5m3",  # scale_format pinned to AMD-FP4 inner grid
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
        kind="nvfp4_linear",  # same K-aware floor scheme as NVFP4
        context=(
            f"AMDFP4Linear shape={shape} dtype={data_type} "
            f"x_2d={use_2dblock_x} w_2d={use_2dblock_w}"
        ),
    )


# ---------------------------------------------------------------------------
# Thin-wrapper surface: ``_to_amdfp4_then_scaled_mm`` must actually be the
# UE5M3-pinned variant of ``_to_nvfp4_then_scaled_mm`` (TEST-1).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
def test_amdfp4_to_scaled_mm_wrapper_pins_ue5m3(use_2dblock_x, use_2dblock_w):
    """Exercise the *real* AMD-FP4 thin wrapper end-to-end.

    The autograd-function test above calls ``AMDFP4LinearFunction`` directly
    with an explicit ``scale_format="ue5m3"`` and never touches
    ``_to_amdfp4_then_scaled_mm`` — the helper the dispatch layer actually
    routes through.  This test calls that helper and asserts it is bit-for-bit
    identical to the NVFP4 helper invoked with ``scale_format="ue5m3"`` (i.e.
    the wrapper's only job — pinning the inner grid to UE5M3 — is real and
    correct), while differing from the default E4M3 NVFP4 path.

    NOTE: E4M3 and UE5M3 share the same 3-bit mantissa, so on normal-range
    inputs both grids round identically and the outputs are bit-equal.  To
    *distinguish* the two (and thus prove the pin actually selects UE5M3) we
    use wide-dynamic-range data where E4M3's inner scale saturates (amax/6 >
    448) but UE5M3's does not.
    """
    torch.manual_seed(0)
    M, N, K = 64, 64, 128
    # randn * 8000 -> per-block amax ~ 1.5e4..3e4 -> inner_scale_raw far above
    # the E4M3 max (448), so E4M3 and UE5M3 genuinely diverge.
    a = (torch.randn((M, K), device="cuda") * 8000.0).to(torch.bfloat16)
    w = (torch.randn((N, K), device="cuda") * 8000.0).to(torch.bfloat16)

    common = dict(
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )

    # Linear semantics: the helper computes ``x_dq @ w_dq.T`` with weight of
    # shape (N, K), so pass ``w`` directly (not transposed).
    y_amd = _to_amdfp4_then_scaled_mm(a, w, **common)
    y_nv_ue5m3 = _to_nvfp4_then_scaled_mm(a, w, scale_format="ue5m3", **common)
    y_nv_e4m3 = _to_nvfp4_then_scaled_mm(a, w, scale_format="e4m3", **common)

    assert y_amd.shape == (M, N)
    assert torch.isfinite(y_amd).all()
    # The AMD-FP4 wrapper is exactly NVFP4 pinned to UE5M3.
    assert torch.equal(y_amd, y_nv_ue5m3), (
        "_to_amdfp4_then_scaled_mm must be bit-identical to "
        "_to_nvfp4_then_scaled_mm(scale_format='ue5m3')"
    )
    # ...and genuinely on a different inner grid than the E4M3 default
    # (guards against the wrapper silently falling back to E4M3).
    assert not torch.equal(y_amd, y_nv_e4m3), (
        "AMD-FP4 wrapper output must differ from the E4M3 NVFP4 path; "
        "identical output suggests scale_format pinning was lost"
    )


@pytest.mark.parametrize("use_outer_2dblock_w", [False, True])
def test_amdfp4_to_scaled_mm_wrapper_outer_block(use_outer_2dblock_w):
    """AMD-FP4 outer-block scaling shares the NVFP4 outer-block code path.

    The AMD-FP4 wrapper must thread BOTH the outer-block knobs and the UE5M3
    inner grid into ``_to_nvfp4_then_scaled_mm``, so its output must be
    bit-identical to the NVFP4 helper called with ``scale_format="ue5m3"`` and
    the same outer-block config.

    NOTE: under outer-block scaling the per-outer-block FP32 scale normalizes
    by the inner grid's max (448 for E4M3, 114688 for UE5M3 -- a factor of
    256 = 2**8).  Because that factor is an exact power of two and both grids
    share the same 3-bit mantissa, the E4M3 and UE5M3 outer-block paths produce
    bit-identical results on well-ranged data (the grid choice is absorbed by
    the outer scale).  We therefore assert UE5M3 parity but do NOT assert
    divergence from E4M3 here -- that divergence only appears at intra-outer-
    block dynamic ranges wide enough to underflow one grid but not the other.
    """
    torch.manual_seed(0)
    # Dims chosen so the outer grids are non-degenerate at outer_block_size=128:
    # K > 128 (1D outer on activations) and N,K > 128 (2D outer on weights).
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device="cuda").to(torch.bfloat16)
    w = torch.randn((N, K), device="cuda").to(torch.bfloat16)

    ob = dict(
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_sr_grad=False,
        use_outer_block_scale=True,
        use_outer_2dblock_w=use_outer_2dblock_w,
        outer_block_size=128,
    )
    y_amd = _to_amdfp4_then_scaled_mm(a, w, **ob)
    y_nv_ue5m3 = _to_nvfp4_then_scaled_mm(a, w, scale_format="ue5m3", **ob)

    assert y_amd.shape == (M, N)
    assert torch.isfinite(y_amd).all()
    assert torch.equal(y_amd, y_nv_ue5m3), (
        "_to_amdfp4_then_scaled_mm outer-block output must be bit-identical to "
        "_to_nvfp4_then_scaled_mm(scale_format='ue5m3') with the same outer-block config"
    )
