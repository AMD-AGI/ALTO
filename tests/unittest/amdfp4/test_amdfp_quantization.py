# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Op-level regression tests for the AMD-FP4 quant / dequant kernels.

Mirrors :mod:`tests.unittest.nvfp4.test_nvfp_quantization` but exercises
the AMD-FP4-side ATen ops (``alto::convert_to_amdfp4`` /
``alto::convert_from_amdfp4``) and pins the inner-scale dtype to
UE5M3 throughout.  The shared blockwise body is regression-tested by
the NVFP4 suite; this file's job is to lock down the AMD-FP4 surface:

* the UE5M3 inner-scale path produces the same outputs as the
  PyTorch UE5M3 oracle (kernel-vs-oracle bit-equality / mismatch rate);
* outer-scale dynamic refresh stays bit-equal between caller-driven and
  kernel-driven paths;
* edge-case patterns (zeros, large saturation) and edge tile shapes
  stay finite;
* a NaN spike in the input does not contaminate inner scales or
  dequant output (D2' defense layer regression).
"""

from __future__ import annotations

import pytest
import torch

from alto.kernels.fp4.amdfp4 import (
    convert_from_amdfp4,
    convert_to_amdfp4,
)
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    convert_from_nvfp4,
    convert_to_nvfp4,
)
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    _OUTER_SCALE_DIVZERO_FLOOR,
    _SCALE_FORMAT_TABLE,
    compute_dynamic_outer_scale,
    is_cdna4,
)

from amdfp4.utils import (  # noqa: E402  (see amdfp4/conftest sys.path)
    convert_from_amdfp4_pytorch,
    convert_to_amdfp4_pytorch,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Bit-exact kernel-vs-oracle agreement on the full axis / shape / SR /
# outer-scale matrix (UE5M3 only — the E4M3 half lives in NVFP4).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("use_outer_scale", [False, True])
@pytest.mark.parametrize("compile", [False])
def test_amdfp4_quantization(tensor_shape, axis, is_2d_block, data_type,
                             use_sr, use_outer_scale, compile):
    block_size = 16
    device = torch.device("cuda")

    if compile:
        quant_func = torch.compile(
            torch.ops.alto.convert_to_amdfp4, fullgraph=True)
        dequant_func = torch.compile(
            torch.ops.alto.convert_from_amdfp4, fullgraph=True)
    else:
        quant_func = torch.ops.alto.convert_to_amdfp4
        dequant_func = torch.ops.alto.convert_from_amdfp4

    if use_outer_scale:
        outer_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        outer_scale = None

    x = prepare_data(tensor_shape, data_type)

    data_lp_ref, scales_ref = convert_to_amdfp4_pytorch(
        x, block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale,
    )
    x_dq_ref = convert_from_amdfp4_pytorch(
        data_lp_ref, scales_ref,
        output_dtype=data_type, block_size=block_size, axis=axis,
        is_2d_block=is_2d_block, outer_scale=outer_scale,
    )

    data_lp, scales = quant_func(
        x, block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale, update_outer_scale=False,
        use_sr=use_sr,
    )
    # Both reference and kernel apply the same float32->UE5M3->float32
    # rounding in the same order, so their scales are bit-for-bit identical.
    assert torch.equal(scales_ref, scales), (
        f"Scale mismatch: max abs diff = {(scales_ref - scales).abs().max().item():.6e}"
    )

    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if not use_sr:
        if is_hip:
            mismatch_rate = (data_lp_ref != data_lp).float().mean().item()
            assert mismatch_rate < 1e-3, (
                f"Quantized data mismatch rate {mismatch_rate:.4%} exceeds 0.1% threshold"
            )
        else:
            assert torch.all(data_lp_ref == data_lp).item()
    else:
        data_lp_ref_lo = (data_lp_ref.to(torch.int8) & 0xF)
        data_lp_ref_hi = ((data_lp_ref.to(torch.int8) >> 4) & 0xF)
        data_lp_lo = (data_lp.to(torch.int8) & 0xF)
        data_lp_hi = ((data_lp.to(torch.int8) >> 4) & 0xF)
        assert torch.all(
            torch.max(torch.abs(data_lp_ref_lo - data_lp_lo)) <= 1).item()
        assert torch.all(
            torch.max(torch.abs(data_lp_ref_hi - data_lp_hi)) <= 1).item()

    x_dq = dequant_func(
        data_lp, scales,
        output_dtype=data_type, block_size=block_size, axis=axis,
        is_2d_block=is_2d_block, outer_scale=outer_scale,
    )

    if not use_sr:
        if is_hip:
            x_dq_cross = convert_from_amdfp4_pytorch(
                data_lp, scales,
                output_dtype=data_type, block_size=block_size, axis=axis,
                is_2d_block=is_2d_block, outer_scale=outer_scale,
            )
            assert torch.equal(x_dq_cross, x_dq), (
                f"Dequant not bit-exact: "
                f"mismatches={int((x_dq_cross != x_dq).sum())}/{x_dq.numel()}, "
                f"max diff={float((x_dq_cross - x_dq).abs().max())}"
            )
        else:
            assert torch.allclose(x_dq_ref, x_dq)
    else:
        dq_mae = (x_dq_ref - x_dq).abs().mean().item()
        assert dq_mae < 1.0 if is_2d_block else 0.5


# ---------------------------------------------------------------------------
# Dynamic outer-scale refresh path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_amdfp4_dynamic_outer_scale(tensor_shape, axis, data_type):
    """``update_outer_scale=True`` must refresh the caller's scale buffer
    in place and produce the same quantized output as a manual
    pre-compute via :func:`compute_dynamic_outer_scale`."""
    block_size = 16

    x = prepare_data(tensor_shape, data_type)

    # Path 1: manual pre-compute, pass in, no update.
    outer_scale_manual = compute_dynamic_outer_scale(x, scale_format="ue5m3")
    data_lp_manual, scales_manual = convert_to_amdfp4(
        x, block_size=block_size, axis=axis,
        outer_scale=outer_scale_manual, update_outer_scale=False,
    )

    # Path 2: caller-owned buffer, dynamically refreshed in-place.
    outer_scale_dyn = torch.empty(1, dtype=torch.float32, device=x.device)
    data_lp_dyn, scales_dyn = convert_to_amdfp4(
        x, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn, update_outer_scale=True,
    )

    assert torch.equal(outer_scale_manual, outer_scale_dyn), (
        f"Dynamic outer_scale mismatch: "
        f"{outer_scale_manual.item()} vs {outer_scale_dyn.item()}"
    )
    assert torch.equal(data_lp_manual, data_lp_dyn)
    assert torch.equal(scales_manual, scales_dyn)

    x_dq = convert_from_amdfp4(
        data_lp_dyn, scales_dyn,
        output_dtype=data_type, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn,
    )
    assert x_dq.shape == x.shape
    assert x_dq.dtype == data_type

    x_dq_ref = convert_from_amdfp4_pytorch(
        data_lp_dyn, scales_dyn,
        output_dtype=data_type, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn,
    )
    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_hip:
        assert torch.equal(x_dq_ref, x_dq)
    else:
        assert torch.allclose(x_dq_ref, x_dq)


# ---------------------------------------------------------------------------
# Edge-case patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["zeros", "large"])
def test_amdfp4_special_values(tensor_shape, axis, data_type, pattern):
    """Edge-case input patterns:

    * ``zeros``: all-zero tensor — exercises the UE5M3 lower clamp on
      the stored block scale.
    * ``large``: 5000.0 everywhere — well below ``UE5M3_MAX * F4_E2M1_MAX``
      (~688k) so for AMD-FP4 the inner-scale upper clamp does *not*
      fire on this pattern (it does for NVFP4); regression target is
      the kernel emitting the same answer as the oracle.
    """
    block_size = 16

    x = prepare_data(tensor_shape, data_type, pattern=pattern)

    data_lp_ref, scales_ref = convert_to_amdfp4_pytorch(
        x, block_size=block_size, axis=axis,
    )

    data_lp, scales = convert_to_amdfp4(
        x, block_size=block_size, axis=axis, update_outer_scale=False,
    )
    assert torch.equal(scales_ref, scales), (
        f"Scale mismatch for pattern={pattern}: "
        f"max diff={float((scales_ref - scales).abs().max())}"
    )

    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_hip:
        mismatch_rate = (data_lp_ref != data_lp).float().mean().item()
        assert mismatch_rate < 1e-3, (
            f"Quantized data mismatch rate {mismatch_rate:.4%} for pattern={pattern}"
        )
    else:
        assert torch.all(data_lp_ref == data_lp).item(), (
            f"Quantized data mismatch for pattern={pattern}"
        )

    x_dq = convert_from_amdfp4(
        data_lp, scales,
        output_dtype=data_type, block_size=block_size, axis=axis,
    )
    x_dq_ref = convert_from_amdfp4_pytorch(
        data_lp, scales,
        output_dtype=data_type, block_size=block_size, axis=axis,
    )
    assert torch.equal(x_dq_ref, x_dq), (
        f"Dequant mismatch for pattern={pattern}"
    )


# ---------------------------------------------------------------------------
# Outer-scale + zero tensor: divzero floor and FP32-normal range invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_2d_block", [False, True])
def test_amdfp4_zero_tensor_with_outer_scale(is_2d_block):
    """All-zero input on the outer-scale branch must round-trip to 0 with
    finite scales, and the effective per-block divisor
    ``inner_scale * outer_scale`` must stay in FP32 normal range
    (``UE5M3_EPS * _OUTER_SCALE_DIVZERO_FLOOR`` is the worst case)."""
    block_size = 16
    axis = -1
    data_type = torch.bfloat16
    x = prepare_data((128, 64), data_type, pattern="zeros")

    outer_scale_buf = torch.empty(1, dtype=torch.float32, device=x.device)
    fmt_eps, _ = _SCALE_FORMAT_TABLE["ue5m3"]
    data_lp, scales = convert_to_amdfp4(
        x, block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale_buf, update_outer_scale=True,
    )

    expected_floor_fp32 = torch.tensor(
        _OUTER_SCALE_DIVZERO_FLOOR, dtype=torch.float32
    ).item()
    outer_scale_value = outer_scale_buf.item()
    assert outer_scale_value == expected_floor_fp32, (
        f"Zero-tensor outer_scale must be floored to "
        f"_OUTER_SCALE_DIVZERO_FLOOR (FP32-rounded {expected_floor_fp32:.6e}), "
        f"got {outer_scale_value:.6e}"
    )

    fp32_min_normal = torch.finfo(torch.float32).tiny
    eff_min = scales.min().item() * outer_scale_value
    assert eff_min >= fp32_min_normal, (
        f"Effective quant_scale ({eff_min:.6e}) fell into FP32 subnormal range; "
        f"_OUTER_SCALE_DIVZERO_FLOOR * ue5m3_EPS ({fmt_eps:.6e}) would be "
        f"flushed to zero under FTZ."
    )

    x_dq = convert_from_amdfp4(
        data_lp, scales, output_dtype=data_type,
        block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale_buf,
    )
    assert torch.isfinite(scales).all(), "stored block scale has NaN/Inf"
    assert torch.isfinite(x_dq).all(), "outer_scale+zero dequant produced NaN/Inf"
    assert (x_dq == 0).all(), "outer_scale+zero dequant must be exactly 0"
    assert (data_lp == 0).all(), "outer_scale+zero packed FP4 must be all zero bins"


# ---------------------------------------------------------------------------
# Edge-tile shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_amdfp4_non_aligned_m_no_nan_inf(data_type):
    """Regression for the edge-tile bug on non-aligned M.

    Same shared kernel as NVFP4; this test makes sure the AMD-FP4 op
    surface inherits the masked load/store edge-tile fix.
    """
    x = prepare_data((150, 128), data_type)
    data_lp, scales = convert_to_amdfp4(
        x,
        block_size=16,
        axis=-1,
        is_2d_block=False,
        update_outer_scale=False,
    )
    x_dq = convert_from_amdfp4(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=16,
        axis=-1,
        is_2d_block=False,
    )
    assert torch.isfinite(scales).all(), "non-aligned M produced Inf/NaN scales"
    assert torch.isfinite(x_dq).all(), "non-aligned M produced Inf/NaN dequant output"


# ---------------------------------------------------------------------------
# D2' NaN-input sanitization regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_amdfp4_quantization_nan_input_sanitized(data_type):
    """A NaN spike in the input MUST NOT propagate to inner scales / dequant.

    Under D2' the UE5M3 cast is spec-aligned (NaN input -> 0xFF NaN code).
    The defense layer in the shared ``_calculate_inner_scales`` (Triton)
    and ``_quantize_inner_scale`` (PyTorch oracle) sanitises NaN
    ``max_abs`` before the cast, so the resulting inner-scale tensor is
    finite and downstream GEMM never sees a NaN.

    Mirrors the industry pattern (TransformerEngine "caller-side
    sanitize", vLLM "zero-out padding", TRT-LLM "amax clamp").
    """
    block_size = 16
    x = prepare_data((128, 64), data_type)
    x_nan = x.clone()
    x_nan[0, 7] = float("nan")

    data_lp, scales = convert_to_amdfp4(
        x_nan,
        block_size=block_size,
        axis=-1,
        is_2d_block=False,
        update_outer_scale=False,
    )
    assert torch.isfinite(scales.float()).all(), (
        "NaN input produced non-finite inner scales under AMD-FP4; "
        "D2' defense layer should have sanitised the block amax."
    )

    x_dq = convert_from_amdfp4(
        data_lp, scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=-1,
        is_2d_block=False,
    )
    assert torch.isfinite(x_dq).all(), (
        "NaN input contaminated AMD-FP4 dequant output; "
        "defense layer must keep downstream GEMM input finite."
    )


# ---------------------------------------------------------------------------
# Wide-dynamic-range advantage: the regime AMD-FP4 exists for (TEST-2).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_amdfp4_beats_e4m3_on_wide_dynamic_range(data_type):
    """AMD-FP4's reason to exist: when a block's ideal inner scale exceeds the
    E4M3 max (448), NVFP4 saturates the inner scale and clamps the block's
    large elements, whereas UE5M3 (max 114688) represents it faithfully.

    We build data whose per-block amax forces ``inner_scale_raw = amax/6`` well
    above 448 but below UE5M3_MAX, then round-trip through both inner grids
    (no outer scale) and assert AMD-FP4's reconstruction error is dramatically
    lower than NVFP4(e4m3) on the *same* data.
    """
    torch.manual_seed(0)
    block_size = 16
    # randn * 8000 -> block amax ~ 1.5e4..3e4 -> inner_scale_raw ~ 2.5e3..5e3,
    # far above E4M3 max 448 (saturates) but far below UE5M3 max 114688.
    x = (torch.randn((256, 256), device="cuda") * 8000.0).to(data_type)

    # NVFP4 (E4M3 inner): inner scale saturates -> large elements clamped.
    nv_lp, nv_scales = convert_to_nvfp4(
        x, block_size=block_size, axis=-1,
        update_outer_scale=False, scale_format="e4m3",
    )
    x_nv = convert_from_nvfp4(
        nv_lp, nv_scales, output_dtype=torch.float32,
        block_size=block_size, axis=-1, scale_format="e4m3",
    )

    # AMD-FP4 (UE5M3 inner): inner scale fits -> faithful representation.
    amd_lp, amd_scales = convert_to_amdfp4(
        x, block_size=block_size, axis=-1, update_outer_scale=False,
    )
    x_amd = convert_from_amdfp4(
        amd_lp, amd_scales, output_dtype=torch.float32,
        block_size=block_size, axis=-1,
    )

    xf = x.float()
    err_nv = (x_nv - xf).abs().mean().item()
    err_amd = (x_amd - xf).abs().mean().item()

    # Sanity: this data must actually saturate E4M3 (otherwise the test is
    # vacuous).  E4M3 max-normal inner scale * F4_E2M1_MAX caps any block's
    # representable magnitude at 448*6=2688; our amax is >> that.
    assert xf.abs().max().item() > 2688.0, (
        "test data does not enter the E4M3-saturation regime"
    )
    # AMD-FP4 must be at least ~4x more accurate here; in practice it is
    # orders of magnitude better.
    assert err_amd < err_nv * 0.25, (
        f"AMD-FP4 should crush NVFP4 on wide-dynamic-range data: "
        f"err_amd={err_amd:.2f} vs err_nv={err_nv:.2f}"
    )


# ---------------------------------------------------------------------------
# BUG-1 repro: NaN + dynamic outer scale must not poison the whole tensor.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_amdfp4_nan_with_dynamic_outer_scale_is_contained(is_2d_block, data_type):
    """Regression for BUG-1.

    With ``update_outer_scale=True`` the per-tensor outer scale is computed
    from ``data.abs().max()``.  A single NaN element makes that amax NaN, and
    because the outer scale is shared, *every* block's ``inner_scale_raw``
    would become NaN — bypassing the per-block NaN defense in
    ``_calculate_inner_scales``.  ``compute_dynamic_outer_scale`` must
    ``nan_to_num`` the reduced amax so the NaN spike degrades to "that block
    goes to 0" instead of "the whole tensor is NaN".
    """
    block_size = 16
    x = prepare_data((128, 64), data_type)
    x_nan = x.clone()
    x_nan[0, 7] = float("nan")

    outer_scale = torch.empty(1, dtype=torch.float32, device=x_nan.device)
    data_lp, scales = convert_to_amdfp4(
        x_nan, block_size=block_size, axis=-1, is_2d_block=is_2d_block,
        outer_scale=outer_scale, update_outer_scale=True,
    )

    assert torch.isfinite(outer_scale).all(), (
        "dynamic outer scale became NaN — torch.clamp does not strip NaN; "
        "compute_dynamic_outer_scale must nan_to_num the amax"
    )
    assert torch.isfinite(scales.float()).all(), (
        "NaN poisoned the inner scales via the per-tensor outer scale"
    )

    x_dq = convert_from_amdfp4(
        data_lp, scales, output_dtype=data_type,
        block_size=block_size, axis=-1, is_2d_block=is_2d_block,
        outer_scale=outer_scale,
    )
    assert torch.isfinite(x_dq).all(), (
        "NaN contaminated the dequant output despite the defense layer"
    )


# A small placeholder that ensures ``is_cdna4`` import survives — used
# by call sites that detect CDNA4 to enable hardware ASM fast paths.
def test_amdfp4_is_cdna4_helper_callable():
    assert isinstance(is_cdna4(), bool)
