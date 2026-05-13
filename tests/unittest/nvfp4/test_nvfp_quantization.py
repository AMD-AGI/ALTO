# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    _OUTER_SCALE_DIVZERO_FLOOR,
    E4M3_EPS,
    F4_E2M1_MAX,
    F8E4M3_MAX,
    OUTER_SCALE_EPS,
    convert_to_nvfp4,
    convert_from_nvfp4,
    compute_dynamic_outer_scale,
    compute_outer_block_scale_shape,
    convert_to_nvfp4_outer_block,
    convert_from_nvfp4_outer_block,
    _outer_scale_and_expanded_axis_last,
    _qdq,
    is_cdna4,
)

from .utils import (
    prepare_data,
    convert_to_nvfp4_pytorch,
    convert_from_nvfp4_pytorch,
    convert_to_nvfp4_outer_block_pytorch,
    convert_from_nvfp4_outer_block_pytorch,
)


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("use_outer_scale", [False, True])
@pytest.mark.parametrize("compile", [False])
def test_nvfp4_quantization(tensor_shape, axis, is_2d_block, data_type,
                            use_sr, use_outer_scale, compile):
    block_size = 16
    device = torch.device("cuda")

    if compile:
        quant_func = torch.compile(
            torch.ops.alto.convert_to_nvfp4, fullgraph=True)
        dequant_func = torch.compile(
            torch.ops.alto.convert_from_nvfp4, fullgraph=True)
    else:
        quant_func = torch.ops.alto.convert_to_nvfp4
        dequant_func = torch.ops.alto.convert_from_nvfp4

    if use_outer_scale:
        outer_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    else:
        outer_scale = None

    x = prepare_data(tensor_shape, data_type)

    data_lp_ref, scales_ref = convert_to_nvfp4_pytorch(
        x, block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale,
    )
    x_dq_ref = convert_from_nvfp4_pytorch(
        data_lp_ref, scales_ref,
        output_dtype=data_type, block_size=block_size, axis=axis,
        is_2d_block=is_2d_block, outer_scale=outer_scale,
    )

    data_lp, scales = quant_func(
        x, block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale, update_outer_scale=False,
        use_sr=use_sr,
    )
    # Both reference and kernel apply the same float32->float8_e4m3fn->float32
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
            x_dq_cross = convert_from_nvfp4_pytorch(
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


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_nvfp4_dynamic_outer_scale(tensor_shape, axis, data_type):
    """Verify that ``update_outer_scale=True`` refreshes the caller's scale
    buffer in place, producing the same quantized output as explicitly
    pre-computing the scale via :func:`compute_dynamic_outer_scale`."""
    block_size = 16

    x = prepare_data(tensor_shape, data_type)

    # Path 1: manual pre-compute, pass in, no update.
    outer_scale_manual = compute_dynamic_outer_scale(x)
    data_lp_manual, scales_manual = convert_to_nvfp4(
        x, block_size=block_size, axis=axis,
        outer_scale=outer_scale_manual, update_outer_scale=False,
    )

    # Path 2: caller-owned buffer, dynamically refreshed in-place.
    outer_scale_dyn = torch.empty(1, dtype=torch.float32, device=x.device)
    data_lp_dyn, scales_dyn = convert_to_nvfp4(
        x, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn, update_outer_scale=True,
    )

    assert torch.equal(outer_scale_manual, outer_scale_dyn), (
        f"Dynamic outer_scale mismatch: "
        f"{outer_scale_manual.item()} vs {outer_scale_dyn.item()}"
    )
    assert torch.equal(data_lp_manual, data_lp_dyn)
    assert torch.equal(scales_manual, scales_dyn)

    x_dq = convert_from_nvfp4(
        data_lp_dyn, scales_dyn,
        output_dtype=data_type, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn,
    )
    assert x_dq.shape == x.shape
    assert x_dq.dtype == data_type

    x_dq_ref = convert_from_nvfp4_pytorch(
        data_lp_dyn, scales_dyn,
        output_dtype=data_type, block_size=block_size, axis=axis,
        outer_scale=outer_scale_dyn,
    )
    is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
    if is_hip:
        assert torch.equal(x_dq_ref, x_dq)
    else:
        assert torch.allclose(x_dq_ref, x_dq)


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["zeros", "large"])
def test_nvfp4_special_values(tensor_shape, axis, data_type, pattern):
    """Verify quantization correctness on edge-case inputs.
    - zeros: all-zero tensor, exercises the E4M3 lower clamp on the
             stored block scale.
    - large: 5000.0 everywhere, exceeds F8E4M3_MAX * F4_E2M1_MAX (2688),
             exercises FP4 saturation and scale clamp to F8E4M3_MAX.
    """
    block_size = 16

    x = prepare_data(tensor_shape, data_type, pattern=pattern)

    data_lp_ref, scales_ref = convert_to_nvfp4_pytorch(
        x, block_size=block_size, axis=axis,
    )

    data_lp, scales = convert_to_nvfp4(
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

    x_dq = convert_from_nvfp4(
        data_lp, scales,
        output_dtype=data_type, block_size=block_size, axis=axis,
    )
    x_dq_ref = convert_from_nvfp4_pytorch(
        data_lp, scales,
        output_dtype=data_type, block_size=block_size, axis=axis,
    )
    assert torch.equal(x_dq_ref, x_dq), (
        f"Dequant mismatch for pattern={pattern}"
    )


@pytest.mark.parametrize("is_2d_block", [False, True])
def test_nvfp4_zero_tensor_with_outer_scale(is_2d_block):
    """All-zero tensor on the outer-scale branch must round-trip to 0 with
    no NaN/Inf, and the effective per-block divisor
    ``inner_scale * outer_scale`` must stay in FP32 normal range.

    Complements ``test_nvfp4_special_values["zeros"]`` which only covers
    the non-outer-scale branch.
    """
    block_size = 16
    axis = -1
    data_type = torch.bfloat16
    x = prepare_data((128, 64), data_type, pattern="zeros")

    outer_scale_buf = torch.empty(1, dtype=torch.float32, device=x.device)
    data_lp, scales = convert_to_nvfp4(
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
        f"_OUTER_SCALE_DIVZERO_FLOOR * E4M3_EPS would be flushed to zero under FTZ."
    )

    x_dq = convert_from_nvfp4(
        data_lp, scales, output_dtype=data_type,
        block_size=block_size, axis=axis, is_2d_block=is_2d_block,
        outer_scale=outer_scale_buf,
    )
    assert torch.isfinite(scales).all(), "stored block scale has NaN/Inf"
    assert torch.isfinite(x_dq).all(), "outer_scale+zero dequant produced NaN/Inf"
    assert (x_dq == 0).all(), "outer_scale+zero dequant must be exactly 0"
    assert (data_lp == 0).all(), "outer_scale+zero packed FP4 must be all zero bins"


@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
def test_nvfp4_non_aligned_m_no_nan_inf(data_type):
    """Regression for the pre-existing edge-tile bug on non-aligned M.

    Historically ``convert_to_nvfp4`` / ``convert_from_nvfp4`` could emit
    garbage scales or NaN/Inf outputs when the leading dimension was not a
    multiple of the Triton tile size (64), even though it was still divisible
    by the quant block size (16).  The kernel now uses masked load/store on
    edge tiles, so a shape such as (150, 128) must remain finite throughout.
    """
    x = prepare_data((150, 128), data_type)
    data_lp, scales = convert_to_nvfp4(
        x,
        block_size=16,
        axis=-1,
        is_2d_block=False,
        update_outer_scale=False,
    )
    x_dq = convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=16,
        axis=-1,
        is_2d_block=False,
    )
    assert torch.isfinite(scales).all(), "non-aligned M produced Inf/NaN scales"
    assert torch.isfinite(x_dq).all(), "non-aligned M produced Inf/NaN dequant output"


@pytest.mark.parametrize("is_2d_inner,is_2d_outer,expected_inner,expected_outer,label", [
    (False, False, (128, 16), (128, 2), "inner=1x16 outer=1x128"),
    (False, True,  (128, 16), (1, 2),   "inner=1x16 outer=128x128"),
    (True,  False, (8, 16),   (128, 2), "inner=16x16 outer=1x128"),
    (True,  True,  (8, 16),   (1, 2),   "inner=16x16 outer=128x128"),
])
def test_nvfp4_outer_block_scale_shapes(
    is_2d_inner, is_2d_outer, expected_inner, expected_outer, label
):
    """Outer-scale shape helper must match the requested four layouts."""
    inner_shape, outer_shape = compute_outer_block_scale_shape(
        (128, 256),
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        inner_block_size=16,
        outer_block_size=128,
    )
    assert inner_shape == expected_inner, label
    assert outer_shape == expected_outer, label


@pytest.mark.parametrize("kwargs,match", [
    (
        dict(shape=(128, 64), is_2d_inner=False, is_2d_outer=False,
             inner_block_size=16, outer_block_size=128),
        "last dimension",
    ),
    (
        dict(shape=(128, 192), is_2d_inner=False, is_2d_outer=False,
             inner_block_size=16, outer_block_size=128),
        "last dimension",
    ),
    (
        dict(shape=(64, 256), is_2d_inner=False, is_2d_outer=True,
             inner_block_size=16, outer_block_size=128),
        "2D outer block",
    ),
    (
        dict(shape=(16,), is_2d_inner=False, is_2d_outer=False,
             inner_block_size=16, outer_block_size=128),
        "at least 2D",
    ),
    (
        dict(shape=(128, 256), is_2d_inner=False, is_2d_outer=False,
             inner_block_size=16, outer_block_size=100),
        "must be a multiple",
    ),
])
def test_nvfp4_outer_block_scale_shape_rejects_invalid_inputs(kwargs, match):
    """Outer-block scaling is fail-fast: no padding or ragged tails."""
    with pytest.raises(ValueError, match=match):
        compute_outer_block_scale_shape(**kwargs)


def test_nvfp4_outer_block_size_equal_inner_block_is_valid():
    """outer_block_size=inner_block_size is a legal degenerate layout."""
    inner_shape, outer_shape = compute_outer_block_scale_shape(
        (128, 256),
        is_2d_inner=False,
        is_2d_outer=False,
        inner_block_size=16,
        outer_block_size=16,
    )
    assert inner_shape == (128, 16)
    assert outer_shape == (128, 16)


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_outer_block_roundtrip_finite(is_2d_inner, is_2d_outer, axis, dtype):
    """All four outer-scale layouts should round-trip with finite outputs."""
    x = prepare_data((128, 256), dtype)
    data_lp, inner_scales, outer_scales = convert_to_nvfp4_outer_block(
        x,
        block_size=16,
        outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
    )
    x_dq = convert_from_nvfp4_outer_block(
        data_lp,
        inner_scales,
        outer_scales,
        output_dtype=x.dtype,
        block_size=16,
        outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )
    assert x_dq.shape == x.shape
    assert x_dq.dtype == dtype
    assert torch.isfinite(inner_scales).all()
    assert torch.isfinite(outer_scales).all()
    assert torch.isfinite(x_dq).all()


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
def test_nvfp4_qdq_routes_to_outer_block(is_2d_inner, is_2d_outer):
    """The shared QDQ helper should expose the outer-block round-trip path."""
    x = prepare_data((128, 256), torch.bfloat16)
    x_qdq = _qdq(
        x,
        axis=-1,
        is_2d_block=is_2d_inner,
        use_outer_scale=False,
        use_outer_block_scale=True,
        is_2d_outer=is_2d_outer,
        outer_block_size=128,
    )
    assert x_qdq.shape == x.shape
    assert x_qdq.dtype == x.dtype
    assert torch.isfinite(x_qdq).all()


def test_nvfp4_qdq_rejects_pts_and_outer_block_together():
    """Tensor-wise and outer-block scaling are alternative second-level scales."""
    x = prepare_data((128, 256), torch.bfloat16)
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        _qdq(
            x,
            axis=-1,
            is_2d_block=False,
            use_outer_scale=True,
            use_outer_block_scale=True,
        )


def test_nvfp4_qdq_rejects_outer_block_return_raw():
    """DGE needs raw FP4 plus scales; outer-block raw support is not wired yet."""
    x = prepare_data((128, 256), torch.bfloat16)
    with pytest.raises(RuntimeError, match="return_raw=True"):
        _qdq(
            x,
            axis=-1,
            is_2d_block=False,
            use_outer_scale=False,
            use_outer_block_scale=True,
            return_raw=True,
        )


def _qdq_with_tensorwise_and_outer(
    x: torch.Tensor,
    *,
    is_2d_inner: bool,
    is_2d_outer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pts = compute_dynamic_outer_scale(x)
    data_lp_pts, scales_pts = convert_to_nvfp4(
        x,
        block_size=16,
        axis=-1,
        is_2d_block=is_2d_inner,
        outer_scale=pts,
        update_outer_scale=False,
        use_sr=False,
    )
    x_pts = convert_from_nvfp4(
        data_lp_pts,
        scales_pts,
        output_dtype=x.dtype,
        block_size=16,
        axis=-1,
        is_2d_block=is_2d_inner,
        outer_scale=pts,
    )
    data_lp_ob, inner_scales_ob, outer_scales = convert_to_nvfp4_outer_block(
        x,
        block_size=16,
        outer_block_size=128,
        axis=-1,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
    )
    x_ob = convert_from_nvfp4_outer_block(
        data_lp_ob,
        inner_scales_ob,
        outer_scales,
        output_dtype=x.dtype,
        block_size=16,
        outer_block_size=128,
        axis=-1,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )
    return x_pts, x_ob


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
def test_nvfp4_outer_block_improves_heterogeneous_regions_vs_pts(is_2d_inner, is_2d_outer):
    """Outer scale should preserve low-energy regions in mixed-energy tensors.

    The 256×256 tensor is built from four 128×128 regions with different
    magnitudes.  Tensor-wise scaling is dominated by the high-energy quadrant,
    while outer-block scaling gives each 1×128 or 128×128 region its own FP32
    range scale.  We do not require bitwise equality between layouts; we only
    require the low-energy regions to improve without hurting total error.
    """
    dtype = torch.bfloat16
    torch.manual_seed(2026)
    low_a = (torch.randn((128, 128), device="cuda") * 0.05).to(dtype)
    high = (torch.randn((128, 128), device="cuda") * 5000.0).to(dtype)
    medium = (torch.randn((128, 128), device="cuda") * 5.0).to(dtype)
    low_b = (torch.randn((128, 128), device="cuda") * 0.05).to(dtype)
    x = torch.cat([
        torch.cat([low_a, high], dim=-1),
        torch.cat([medium, low_b], dim=-1),
    ], dim=0)

    x_pts, x_ob = _qdq_with_tensorwise_and_outer(
        x,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )

    low_ref = torch.cat([x[:128, :128].reshape(-1), x[128:, 128:].reshape(-1)]).float()
    low_pts = torch.cat([x_pts[:128, :128].reshape(-1), x_pts[128:, 128:].reshape(-1)]).float()
    low_ob = torch.cat([x_ob[:128, :128].reshape(-1), x_ob[128:, 128:].reshape(-1)]).float()
    low_pts_err = (low_ref - low_pts).pow(2).mean()
    low_ob_err = (low_ref - low_ob).pow(2).mean()

    total_pts_err = (x.float() - x_pts.float()).pow(2).mean()
    total_ob_err = (x.float() - x_ob.float()).pow(2).mean()
    assert low_ob_err < low_pts_err * 0.5, (
        f"outer-block low-region MSE ({low_ob_err.item():.4e}) should be "
        f"lower than tensor-wise MSE ({low_pts_err.item():.4e})"
    )
    assert total_ob_err < total_pts_err * 1.1, (
        f"outer-block total MSE ({total_ob_err.item():.4e}) should not "
        f"regress vs tensor-wise MSE ({total_pts_err.item():.4e})"
    )


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
def test_nvfp4_outer_block_improves_clean_blocks_with_sparse_outlier(is_2d_inner, is_2d_outer):
    """Outer scale should help clean regions when a sparse outlier sets PTS.

    This is closer to a localized-outlier pattern: most values are low-energy,
    but one column in the right outer block contains large values.  The clean
    left outer block should reconstruct better with its own outer scale, while
    total error should remain in the same ballpark.
    """
    dtype = torch.bfloat16
    torch.manual_seed(2027)
    x = (torch.randn((128, 256), device="cuda") * 0.05).to(dtype)
    x[:, 192] = (torch.randn((128,), device="cuda") * 5000.0).to(dtype)

    x_pts, x_ob = _qdq_with_tensorwise_and_outer(
        x,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )

    low_ref = x[:, :128].float()
    low_pts_err = (low_ref - x_pts[:, :128].float()).pow(2).mean()
    low_ob_err = (low_ref - x_ob[:, :128].float()).pow(2).mean()
    total_pts_err = (x.float() - x_pts.float()).pow(2).mean()
    total_ob_err = (x.float() - x_ob.float()).pow(2).mean()
    assert low_ob_err < low_pts_err * 0.75, (
        f"outer-block clean-region MSE ({low_ob_err.item():.4e}) should be "
        f"lower than tensor-wise MSE ({low_pts_err.item():.4e})"
    )
    assert total_ob_err < total_pts_err * 1.25, (
        f"outer-block total MSE ({total_ob_err.item():.4e}) should stay close "
        f"to tensor-wise MSE ({total_pts_err.item():.4e})"
    )


# ---------------------------------------------------------------------------
# Outer-block scale uses an FP32 floor.
#
# The outer scale is a free FP32 number with no E4M3 round-trip; clamping it
# at ``E4M3_EPS`` (~1.95e-3) instead of FP32 tiny would inflate small-block
# scales and push inner E4M3 scales into the precision-poor ``< 1`` region.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_2d_outer", [False, True])
def test_outer_scale_tracks_amax_below_e4m3_eps(is_2d_outer):
    """Small-amax blocks must produce an outer scale below ``E4M3_EPS``."""
    target_amax = 1.0
    expected_scale = target_amax / (F8E4M3_MAX * F4_E2M1_MAX)  # ~3.7e-4
    assert expected_scale < E4M3_EPS, (
        "precondition: expected outer scale must be below the E4M3 floor"
    )

    dtype = torch.bfloat16
    if is_2d_outer:
        x = torch.full((128, 128), target_amax, device="cuda", dtype=dtype)
    else:
        x = torch.full((1, 128), target_amax, device="cuda", dtype=dtype)
    x[..., 0] = -target_amax

    outer_scale, _ = _outer_scale_and_expanded_axis_last(
        x,
        is_2d_inner=False,
        is_2d_outer=is_2d_outer,
        inner_block_size=16,
        outer_block_size=128,
    )
    assert outer_scale.dtype == torch.float32
    assert (outer_scale < E4M3_EPS * 0.5).all(), (
        f"outer_scale {outer_scale.flatten().tolist()} should track "
        f"amax/2688 ≈ {expected_scale:.3e}, well below E4M3_EPS={E4M3_EPS:.3e}"
    )
    torch.testing.assert_close(
        outer_scale, torch.full_like(outer_scale, expected_scale),
        atol=expected_scale * 1e-2, rtol=1e-2,
    )


@pytest.mark.parametrize("is_2d_outer", [False, True])
def test_outer_scale_dead_block_avoids_div_by_zero(is_2d_outer):
    """All-zero outer blocks must produce a finite, positive scale."""
    if is_2d_outer:
        x = torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16)
    else:
        x = torch.zeros((1, 128), device="cuda", dtype=torch.bfloat16)

    outer_scale, expanded = _outer_scale_and_expanded_axis_last(
        x,
        is_2d_inner=False,
        is_2d_outer=is_2d_outer,
        inner_block_size=16,
        outer_block_size=128,
    )
    assert (outer_scale >= OUTER_SCALE_EPS).all()
    assert torch.isfinite(outer_scale).all()
    assert torch.isfinite(expanded).all()
    assert torch.isfinite(x.float() / expanded).all()


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(True, True, id="inner_16x16_outer_128x128"),
])
def test_outer_block_roundtrip_does_not_regress_on_small_amax(
    is_2d_inner, is_2d_outer,
):
    """Small-amax data must round-trip with comparable relative MSE to normal-amax data."""
    dtype = torch.bfloat16
    torch.manual_seed(2026)
    x_small = (torch.randn((128, 256), device="cuda") * 1e-3).to(dtype)
    x_normal = torch.randn((128, 256), device="cuda", dtype=dtype)

    def _roundtrip(x):
        data_lp, inner_scales, outer_scales = convert_to_nvfp4_outer_block(
            x,
            block_size=16,
            outer_block_size=128,
            axis=-1,
            is_2d_inner=is_2d_inner,
            is_2d_outer=is_2d_outer,
            use_sr=False,
        )
        return convert_from_nvfp4_outer_block(
            data_lp, inner_scales, outer_scales,
            output_dtype=x.dtype,
            block_size=16,
            outer_block_size=128,
            axis=-1,
            is_2d_inner=is_2d_inner,
            is_2d_outer=is_2d_outer,
        )

    x_small_qdq = _roundtrip(x_small)
    x_normal_qdq = _roundtrip(x_normal)

    rel_err_small = (
        (x_small.float() - x_small_qdq.float()).pow(2).mean()
        / x_small.float().pow(2).mean().clamp(min=1e-30)
    )
    rel_err_normal = (
        (x_normal.float() - x_normal_qdq.float()).pow(2).mean()
        / x_normal.float().pow(2).mean().clamp(min=1e-30)
    )
    # An over-aggressive floor would push inner E4M3 into the ``< 1`` region
    # for small-amax data and inflate this ratio by ~10x.
    assert rel_err_small < rel_err_normal * 3.0, (
        f"small-amax relative MSE ({rel_err_small.item():.3e}) regressed vs "
        f"normal-amax relative MSE ({rel_err_normal.item():.3e})"
    )


# ---------------------------------------------------------------------------
# ``precomputed_outer_scale`` plumbing.
#
# 2D outer-block scale grids are axis-invariant on the same tensor; callers
# can compute the scale once and feed it back via ``precomputed_outer_scale``
# to skip a redundant outer-amax reduction on the axis=0 QDQ.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
@pytest.mark.parametrize("axis", [-1, -2])
def test_outer_block_precomputed_scale_matches_recompute(
    is_2d_inner, is_2d_outer, axis,
):
    """Reusing a captured outer scale must produce identical packed FP4 + scales.

    Axis-invariance is what lets the linear path share an outer scale across
    the axis=-1 and axis=0 QDQ calls.  Run the QDQ once, capture the outer
    scale, run it again with ``precomputed_outer_scale`` set, and require
    bitwise equality of every output.
    """
    x = prepare_data((128, 256), torch.bfloat16)

    data_lp_a, inner_a, outer_a = convert_to_nvfp4_outer_block(
        x,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
    )
    data_lp_b, inner_b, outer_b = convert_to_nvfp4_outer_block(
        x,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
        precomputed_outer_scale=outer_a,
    )
    torch.testing.assert_close(outer_a, outer_b, atol=0, rtol=0)
    assert torch.equal(data_lp_a, data_lp_b)
    torch.testing.assert_close(inner_a, inner_b, atol=0, rtol=0)


def test_outer_block_precomputed_scale_validates_shape():
    """Wrong-shape ``precomputed_outer_scale`` must raise a clear ValueError."""
    x = prepare_data((128, 256), torch.bfloat16)
    bad_outer = torch.zeros((1, 1), device=x.device, dtype=torch.float32)
    with pytest.raises(ValueError, match="precomputed_outer_scale"):
        convert_to_nvfp4_outer_block(
            x,
            block_size=16, outer_block_size=128,
            axis=-1,
            is_2d_inner=False, is_2d_outer=False,
            use_sr=False,
            precomputed_outer_scale=bad_outer,
        )


def test_qdq_precomputed_outer_scale_routes_through():
    """``_qdq(precomputed_outer_scale=...)`` must produce identical output."""
    x = prepare_data((128, 256), torch.bfloat16)
    dq_a, outer_scale = _qdq(
        x,
        axis=-1,
        is_2d_block=False,
        use_outer_scale=False,
        use_outer_block_scale=True,
        is_2d_outer=True,
        outer_block_size=128,
        return_outer_scale=True,
    )
    dq_b = _qdq(
        x,
        axis=-1,
        is_2d_block=False,
        use_outer_scale=False,
        use_outer_block_scale=True,
        is_2d_outer=True,
        outer_block_size=128,
        precomputed_outer_scale=outer_scale,
    )
    assert torch.equal(dq_a, dq_b)


def test_qdq_rejects_precomputed_outer_scale_without_outer_block():
    """``precomputed_outer_scale`` is meaningful only when outer-block is on."""
    x = prepare_data((128, 256), torch.bfloat16)
    junk = torch.zeros((1,), device=x.device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="precomputed_outer_scale"):
        _qdq(
            x,
            axis=-1,
            is_2d_block=False,
            use_outer_scale=False,
            use_outer_block_scale=False,
            precomputed_outer_scale=junk,
        )


def test_qdq_rejects_return_outer_scale_without_outer_block():
    x = prepare_data((128, 256), torch.bfloat16)
    with pytest.raises(RuntimeError, match="return_outer_scale"):
        _qdq(
            x,
            axis=-1,
            is_2d_block=False,
            use_outer_scale=False,
            use_outer_block_scale=False,
            return_outer_scale=True,
        )


def test_qdq_rejects_return_raw_and_return_outer_scale_together():
    x = prepare_data((128, 256), torch.bfloat16)
    with pytest.raises(RuntimeError, match="cannot return both"):
        _qdq(
            x,
            axis=-1,
            is_2d_block=False,
            use_outer_scale=False,
            use_outer_block_scale=True,
            is_2d_outer=True,
            outer_block_size=128,
            return_raw=True,
            return_outer_scale=True,
        )


# ---------------------------------------------------------------------------
# Outer-amax reductions are minimized when the scale grid is axis-invariant.
#
# Monkey-patch the only function that performs the reduction
# (``_outer_scale_and_expanded_axis_last``) and verify the call count matches
# what we expect for each forward / backward axis-sharing configuration.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_2dblock_x,use_2dblock_w,use_outer_2dblock_x,use_outer_2dblock_w,"
    "use_hadamard,expected_calls,id_label",
    [
        # 2D outer, no Hadamard: axis=0 QDQ on x, w, and grad_output all reuse
        # the axis=-1 outer scale.  3 reductions = fwd_x + fwd_w + bwd_g.
        (False, False, True, True, False, 3, "C2_no_hadamard"),
        # 2D outer with Hadamard: the rotated x and grad_output are different
        # tensors than the fprop ones, so only w can reuse.  5 reductions.
        (False, False, True, True, True, 5, "C2_hadamard_blocks_x_share"),
        # 1D outer: scale grid is not axis-invariant; nothing can be shared.
        # 6 reductions = fwd (x_-1, w_-1, x_0, w_0) + bwd (g_-1, g_0).
        (False, False, False, False, False, 6, "C1_no_share"),
        # 2D inner block: axis=0 dq views are aliased to the fprop views, so
        # no extra QDQ is issued in the first place.  3 reductions.
        (True, True, True, True, False, 3, "C4_inner_2d_aliasing"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_nvfp4_linear_outer_block_reuses_fwd_outer_scale(
    use_2dblock_x, use_2dblock_w,
    use_outer_2dblock_x, use_outer_2dblock_w,
    use_hadamard, expected_calls, id_label,
):
    """Outer-amax reductions per forward+backward must match the expected count."""
    from alto.kernels.fp4.nvfp4 import nvfp_quantization as quant_mod
    from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_scaled_mm

    real_fn = quant_mod._outer_scale_and_expanded_axis_last
    calls = []

    def _counted(*args, **kwargs):
        calls.append(1)
        return real_fn(*args, **kwargs)

    quant_mod._outer_scale_and_expanded_axis_last = _counted
    try:
        # Shapes satisfy the 1D wgrad-axis alignment contract
        # (x.shape[0] % 16 == 0, weight.shape[0] % 16 == 0).
        torch.manual_seed(0)
        x = torch.randn(128, 256, dtype=torch.bfloat16,
                        device="cuda", requires_grad=True)
        w = torch.randn(128, 256, dtype=torch.bfloat16,
                        device="cuda", requires_grad=True)
        y = _to_nvfp4_then_scaled_mm(
            x, w,
            use_2dblock_x=use_2dblock_x,
            use_2dblock_w=use_2dblock_w,
            use_sr_grad=False,
            use_outer_scale=False,
            use_hadamard=use_hadamard,
            use_dge=False,
            use_outer_block_scale=True,
            use_outer_2dblock_x=use_outer_2dblock_x,
            use_outer_2dblock_w=use_outer_2dblock_w,
            outer_block_size=128,
        )
        y.sum().backward()
    finally:
        quant_mod._outer_scale_and_expanded_axis_last = real_fn

    assert len(calls) == expected_calls, (
        f"{id_label}: expected {expected_calls} outer-amax reductions, "
        f"saw {len(calls)}"
    )


# ---------------------------------------------------------------------------
# Outer-block kernel must agree bit-for-bit with the PyTorch reference under
# deterministic rounding.  This catches axis / outer-amax indexing bugs that
# a self-roundtrip test would miss (errors that cancel between quant and
# dequant).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_convert_to_nvfp4_outer_block_bit_equal_to_pytorch_reference(
    is_2d_inner, is_2d_outer, axis, dtype,
):
    """Quant kernel and PyTorch reference must agree on all outputs bitwise."""
    x = prepare_data((128, 256), dtype)

    data_lp_k, inner_k, outer_k = convert_to_nvfp4_outer_block(
        x,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
    )
    data_lp_p, inner_p, outer_p = convert_to_nvfp4_outer_block_pytorch(
        x,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )

    assert data_lp_k.shape == data_lp_p.shape
    assert inner_k.shape == inner_p.shape
    assert outer_k.shape == outer_p.shape
    assert data_lp_k.dtype == data_lp_p.dtype == torch.uint8
    assert torch.equal(data_lp_k, data_lp_p)
    torch.testing.assert_close(inner_k, inner_p, atol=0, rtol=0)
    torch.testing.assert_close(outer_k, outer_p, atol=0, rtol=0)


@pytest.mark.parametrize("is_2d_inner,is_2d_outer", [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True,  id="inner_1x16_outer_128x128"),
    pytest.param(True,  False, id="inner_16x16_outer_1x128"),
    pytest.param(True,  True,  id="inner_16x16_outer_128x128"),
])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_convert_from_nvfp4_outer_block_bit_equal_to_pytorch_reference(
    is_2d_inner, is_2d_outer, axis, dtype,
):
    """Dequant kernel and PyTorch reference must agree bit-for-bit."""
    x = prepare_data((128, 256), dtype)
    data_lp, inner_scales, outer_scales = convert_to_nvfp4_outer_block(
        x,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
        use_sr=False,
    )

    dq_k = convert_from_nvfp4_outer_block(
        data_lp, inner_scales, outer_scales,
        output_dtype=dtype,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )
    dq_p = convert_from_nvfp4_outer_block_pytorch(
        data_lp, inner_scales, outer_scales,
        output_dtype=dtype,
        block_size=16, outer_block_size=128,
        axis=axis,
        is_2d_inner=is_2d_inner,
        is_2d_outer=is_2d_outer,
    )
    torch.testing.assert_close(dq_k, dq_p, atol=0, rtol=0)


def test_outer_block_close_to_pts_when_outer_covers_whole_tensor():
    """Whole-tensor 2D outer-block QDQ tracks the PTS round-trip within slack.

    The two paths are not algebraically identical: the PTS path applies two
    E4M3 rounds (one on ``amax/6``, one on ``block_scale/pts``), while the
    outer-block path normalizes the input in bf16 and runs a single E4M3
    round on the inner scale.  This test only requires that the resulting
    round-trip outputs stay close, catching catastrophic divergences (axis
    bugs, wrong outer-amax math) without over-constraining the implementation.

    1D outer cases are excluded because ``outer_block_size=N`` produces one
    scale per row, not a global scale.
    """
    M, N = 128, 128
    x = prepare_data((M, N), torch.bfloat16)

    data_lp_ob, inner_ob, outer_ob = convert_to_nvfp4_outer_block(
        x,
        block_size=16,
        outer_block_size=max(M, N),
        axis=-1,
        is_2d_inner=True,
        is_2d_outer=True,
        use_sr=False,
    )
    dq_ob = convert_from_nvfp4_outer_block(
        data_lp_ob, inner_ob, outer_ob,
        output_dtype=torch.bfloat16,
        block_size=16,
        outer_block_size=max(M, N),
        axis=-1,
        is_2d_inner=True,
        is_2d_outer=True,
    )

    pts = compute_dynamic_outer_scale(x)
    data_lp_pts, scales_pts = convert_to_nvfp4(
        x,
        block_size=16,
        axis=-1,
        is_2d_block=True,
        outer_scale=pts,
        update_outer_scale=False,
        use_sr=False,
    )
    dq_pts = convert_from_nvfp4(
        data_lp_pts, scales_pts,
        output_dtype=torch.bfloat16,
        block_size=16,
        axis=-1,
        is_2d_block=True,
        outer_scale=pts,
    )

    # ``prepare_data`` is heavy-tailed (~0.5% entries scaled by ~100×), so a
    # whole-tensor outer scale is dominated by outliers and leaves inner E4M3
    # in the precision-poor ``< 1`` region for the bulk of the tensor.  Allow
    # a generous slack on the absolute round-trip MSE; the binding check is
    # the cross-path agreement.
    sig_pow = x.float().pow(2).mean().clamp(min=1e-30)
    err_ob = (x.float() - dq_ob.float()).pow(2).mean() / sig_pow
    err_pts = (x.float() - dq_pts.float()).pow(2).mean() / sig_pow
    err_xx = (dq_ob.float() - dq_pts.float()).pow(2).mean() / sig_pow

    assert err_ob < 5e-2, f"outer-block round-trip MSE too high: {err_ob.item():.3e}"
    assert err_pts < 5e-2, f"PTS round-trip MSE too high: {err_pts.item():.3e}"
    assert err_xx < 1e-1, (
        f"outer-block / PTS disagreement {err_xx.item():.3e} larger than "
        f"either path's reconstruction error (ob={err_ob.item():.3e}, "
        f"pts={err_pts.item():.3e})"
    )
