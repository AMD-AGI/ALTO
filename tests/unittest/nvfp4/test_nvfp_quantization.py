# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    _OUTER_SCALE_DIVZERO_FLOOR,
    convert_to_nvfp4,
    convert_from_nvfp4,
    compute_dynamic_outer_scale,
    compute_outer_block_scale_shape,
    convert_to_nvfp4_outer_block,
    convert_from_nvfp4_outer_block,
    _qdq,
    is_cdna4,
)

from .utils import (
    prepare_data,
    convert_to_nvfp4_pytorch,
    convert_from_nvfp4_pytorch,
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
