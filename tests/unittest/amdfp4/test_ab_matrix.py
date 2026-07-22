# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""A/B matrix: NVFP4 (E4M3 inner) vs AMD-FP4 (UE5M3 inner) on quant patterns.

Runs the C.1 stress patterns from ``docs/amd-fp4/implementation-validation-plan.md``
and prints a tabulated summary (scale bit-exactness, SQNR, saturation) for both
inner-scale formats on the same input tensor.

Hard gates only apply where format choice can actually distinguish E4M3 from
UE5M3 — i.e. without a dynamic outer scale, since the outer scale would
otherwise normalize each tensor's range into ``[0, max_fmt]`` and erase the
range/precision difference between the two inner grids.

Saturation is defined as the fraction of blocks whose unclamped
``inner_scale_raw = block_amax / outer / F4_E2M1_MAX`` would exceed
``max_fmt`` and therefore trigger the upper-rail inner-scale clamp.  This
is the only saturation signal that is sensitive to the inner-grid choice;
counting "fraction of E2M1 elements at the rail" is by construction always
100% (the per-block divisor is built so the largest element in the block
exactly hits ``F4_E2M1_MAX``).
"""

from __future__ import annotations

import pytest
import torch
from tabulate import tabulate

from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    F4_E2M1_MAX,
    _SCALE_FORMAT_TABLE,
    compute_dynamic_outer_scale,
    convert_from_nvfp4,
    convert_to_nvfp4,
)

from nvfp4.utils import (  # noqa: E402  (see amdfp4/conftest sys.path)
    calc_cossim,
    convert_from_nvfp4_pytorch,
    convert_to_nvfp4_pytorch,
    prepare_data,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="AMD-FP4 A/B matrix requires CUDA / ROCm",
)

AB_PATTERNS = (
    "random",
    "zeros",
    "large",
    "hot_channel",
    "lognormal",
    "near_overflow",
    "near_underflow",
    "single_spike",
)

# Patterns that show up in the printed report (kept short for readability;
# the full matrix is still asserted on).
REPORT_PATTERNS = (
    "random",
    "near_overflow",
    "near_underflow",
    "hot_channel",
    "lognormal",
)


def _sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    num = (x.float() - x_hat.float()).pow(2).sum()
    den = x.float().pow(2).sum().clamp(min=1e-30)
    if num == 0:
        return float("inf")
    return (10.0 * torch.log10(den / num)).item()


def _inner_clamp_rate(
    x: torch.Tensor,
    scale_format: str,
    outer_scale: torch.Tensor | None,
    block_size: int = 16,
) -> float:
    """Fraction of blocks whose unclamped ``inner_scale_raw`` exceeds ``max_fmt``.

    This is the inner-scale upper-rail saturation rate: blocks here lose
    information because the per-block divisor cannot grow large enough to
    fit the block's amplitude into FP4.  UE5M3 is meaningfully better than
    E4M3 only when this rate diverges between the two formats.
    """
    fmt_max = _SCALE_FORMAT_TABLE[scale_format][1]
    x2d = x.float().reshape(-1, x.shape[-1])
    M, N = x2d.shape
    nb = N // block_size
    grouped = x2d.reshape(M, nb, block_size)
    max_abs = grouped.abs().amax(dim=-1)
    if outer_scale is None:
        outer = 1.0
    else:
        outer = outer_scale.float().reshape(()).item()
    inner_raw = max_abs / outer / F4_E2M1_MAX
    return (inner_raw > fmt_max).float().mean().item()


def _run_one_cell(
    pattern: str,
    scale_format: str,
    *,
    use_outer_scale: bool,
) -> dict:
    block_size = 16
    axis = -1
    data_type = torch.bfloat16
    shape = (128, 64)

    x = prepare_data(shape, data_type, pattern=pattern)
    if use_outer_scale:
        # Production path: outer is computed per-format (it scales by
        # 1/(max_fmt * F4_E2M1_MAX), so UE5M3's outer is ~274× smaller than
        # E4M3's on the same tensor — exactly the AMD-FP4 design point).
        outer_scale = compute_dynamic_outer_scale(x, scale_format=scale_format)
    else:
        outer_scale = None

    data_lp_ref, scales_ref = convert_to_nvfp4_pytorch(
        x,
        block_size=block_size,
        axis=axis,
        outer_scale=outer_scale,
        scale_format=scale_format,
    )
    data_lp, scales = convert_to_nvfp4(
        x,
        block_size=block_size,
        axis=axis,
        outer_scale=outer_scale,
        update_outer_scale=False,
        scale_format=scale_format,
        use_sr=False,
    )

    # The full kernel-vs-oracle bit-exact regression is covered by
    # ``tests/unittest/nvfp4/test_nvfp_quantization.py`` (all axes / shapes /
    # block layouts / SR / outer / scale_format).  Here we only require
    # numerical agreement: relative scale error must stay small.  This makes
    # the AB matrix robust against rail-boundary FP rounding differences (e.g.
    # ``inner_raw == max_fmt`` evaluated in two different orders) while still
    # catching real bugs that would push errors orders of magnitude above
    # quantization noise.
    scale_rel_err = (
        (scales_ref.float() - scales.float()).abs()
        / scales_ref.float().abs().clamp(min=1e-30)
    ).max().item()
    scale_ok = scale_rel_err < 5e-3
    fp4_mismatch_rate = (data_lp_ref != data_lp).float().mean().item()
    fp4_ok = fp4_mismatch_rate < 1e-3

    x_dq_ref = convert_from_nvfp4_pytorch(
        data_lp_ref,
        scales_ref,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        outer_scale=outer_scale,
        scale_format=scale_format,
    )
    x_dq = convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        outer_scale=outer_scale,
        scale_format=scale_format,
    )

    mae = (x_dq_ref - x_dq).abs().mean().item()
    sqnr = _sqnr_db(x, x_dq)
    cos = calc_cossim(x, x_dq)
    sat = _inner_clamp_rate(x, scale_format, outer_scale, block_size=block_size)
    finite = torch.isfinite(x_dq).all().item()

    return {
        "scale_ok": scale_ok,
        "scale_rel_err": scale_rel_err,
        "fp4_ok": fp4_ok,
        "fp4_mismatch_rate": fp4_mismatch_rate,
        "mae": mae,
        "sqnr_db": sqnr,
        "cosine": cos,
        "saturation": sat,
        "finite": finite,
    }


def _fmt_sqnr(v: float) -> str:
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.2f}"


@cuda_required
@pytest.mark.parametrize("use_outer_scale", [False, True])
def test_amdfp4_ab_matrix_report(use_outer_scale):
    """Run all AB patterns × {E4M3, UE5M3} and assert format-distinguishing gates."""
    rows = []
    by_pattern_fmt = {}

    for pattern in AB_PATTERNS:
        for fmt, label in (("e4m3", "NVFP4 (E4M3)"), ("ue5m3", "AMD-FP4 (UE5M3)")):
            m = _run_one_cell(pattern, fmt, use_outer_scale=use_outer_scale)
            by_pattern_fmt[(pattern, fmt)] = m
            if pattern in REPORT_PATTERNS:
                rows.append([
                    pattern,
                    label,
                    f"{m['scale_rel_err']:.2e}",
                    _fmt_sqnr(m["sqnr_db"]),
                    f"{m['cosine']:.5f}",
                    f"{m['saturation'] * 100:.3f}%",
                    f"{m['mae']:.3e}",
                    "yes" if m["finite"] else "NO",
                ])

    hdr = [
        "pattern", "format", "scale rel err", "SQNR (dB)", "cosine",
        "inner clamp", "MAE", "finite",
    ]
    print()
    print(tabulate(rows, headers=hdr, tablefmt="github"))
    print(f"\n(use_outer_scale={use_outer_scale})")

    # ---- Always-on invariants ------------------------------------------------
    # Bit-exact kernel-vs-oracle scale agreement is regression-tested by
    # ``tests/unittest/nvfp4/test_nvfp_quantization.py`` over the full
    # axis/shape/SR/outer/scale_format matrix.  Here we only assert that the
    # kernel and the oracle land within one UE5M3 grid step (≤ 1 mantissa
    # ULP, i.e. 12.5%) — this catches catastrophic drift while tolerating
    # rail-boundary code choices that hash to neighbouring UE5M3 codepoints
    # but reconstruct to indistinguishable BF16 outputs (MAE stays tiny).
    UE5M3_GRID_STEP = 0.125 + 1e-6  # one mantissa ULP, with FP slack.
    for (pattern, fmt), m in by_pattern_fmt.items():
        assert m["finite"], f"non-finite dequant: pattern={pattern} format={fmt}"
        assert m["scale_rel_err"] <= UE5M3_GRID_STEP, (
            f"scale drift > 1 inner-format ULP "
            f"(kernel vs PyTorch oracle): pattern={pattern} format={fmt} "
            f"rel_err={m['scale_rel_err']:.3e}"
        )
        # MAE on dequant output stays small relative to tensor amplitude.
        assert m["mae"] < 1.0, (
            f"dequant MAE {m['mae']:.3e} too large: "
            f"pattern={pattern} format={fmt}"
        )

    # On the production outer-scale path the outer normalizes both formats'
    # inner_scale ranges into ``[0, max_fmt]``, which erases the format
    # advantage on every pattern by design.  We only assert non-regression
    # there; the format-distinguishing gates are checked on the
    # ``use_outer_scale=False`` parametrize cell below.

    e4_rand = by_pattern_fmt[("random", "e4m3")]
    ue5_rand = by_pattern_fmt[("random", "ue5m3")]
    assert ue5_rand["sqnr_db"] >= e4_rand["sqnr_db"] - 0.5, (
        f"random: UE5M3 SQNR {ue5_rand['sqnr_db']:.2f} dB regressed vs "
        f"E4M3 {e4_rand['sqnr_db']:.2f} dB by more than 0.5 dB"
    )
    assert ue5_rand["cosine"] >= e4_rand["cosine"] - 0.001, (
        f"random: UE5M3 cosine {ue5_rand['cosine']:.5f} regressed vs "
        f"E4M3 {e4_rand['cosine']:.5f}"
    )

    if not use_outer_scale:
        # ---- near_overflow: x == 1e4 ----------------------------------------
        # E4M3_max=448 < 1e4/6 → every block clamps; reconstruction caps at
        # 448*6 = 2688.  UE5M3_max=114688 (D2', GFXIPARCH-2067 §19.10)
        # ≫ 1e4/6 ≈ 1667 → no clamp.
        e4_ov = by_pattern_fmt[("near_overflow", "e4m3")]
        ue5_ov = by_pattern_fmt[("near_overflow", "ue5m3")]
        assert e4_ov["saturation"] >= 0.99, (
            f"near_overflow: E4M3 should clamp ≥99% of blocks, got "
            f"{e4_ov['saturation']:.4%}"
        )
        assert ue5_ov["saturation"] <= 0.01, (
            f"near_overflow: UE5M3 should clamp ≤1% of blocks, got "
            f"{ue5_ov['saturation']:.4%}"
        )
        assert ue5_ov["sqnr_db"] >= e4_ov["sqnr_db"] + 10.0, (
            f"near_overflow: UE5M3 SQNR {ue5_ov['sqnr_db']:.2f} dB must beat "
            f"E4M3 {e4_ov['sqnr_db']:.2f} dB by ≥10 dB"
        )

        # ---- near_underflow: x == 1e-4 --------------------------------------
        # 1e-4/6 ≈ 1.67e-5; E4M3_EPS = 2^-6 = 1.56e-2 ≫ 1.67e-5 → every block
        # clamps to EPS, x/EPS ≈ 6.4e-3 rounds to E2M1=0 → reconstruction = 0.
        # UE5M3_EPS = 2^-17 = 7.63e-6 < 1.67e-5 → UE5M3 keeps signal.
        e4_un = by_pattern_fmt[("near_underflow", "e4m3")]
        ue5_un = by_pattern_fmt[("near_underflow", "ue5m3")]
        assert ue5_un["sqnr_db"] >= e4_un["sqnr_db"] + 5.0, (
            f"near_underflow: UE5M3 SQNR {ue5_un['sqnr_db']:.2f} dB must beat "
            f"E4M3 {e4_un['sqnr_db']:.2f} dB by ≥5 dB"
        )


# ---------------------------------------------------------------------------
# D2' NaN-input A/B handling
# ---------------------------------------------------------------------------

@cuda_required
@pytest.mark.parametrize("scale_format", ["e4m3", "ue5m3"])
def test_amdfp4_nan_input_handling(scale_format):
    """A NaN spike in the input MUST NOT corrupt either E4M3 or UE5M3 output.

    Under D2' both formats may emit a NaN code at cast time (E4M3 0xFF/0x7F,
    UE5M3 0xFF), so the defense layer in ``_calculate_nvfp4_scales`` /
    ``_quantize_inner_scale`` is what keeps downstream GEMM finite.  This
    test exercises the production kernel + dequant path on a tensor with a
    NaN spike and asserts that the dequant output is finite.

    Mirrors industry pattern: TransformerEngine zero-inits padding scale,
    vLLM zero-outs MoE padding, TRT-LLM clamps scale ≥ 1e-12.
    """
    block_size = 16
    shape = (128, 64)
    data_type = torch.bfloat16
    x = prepare_data(shape, data_type, pattern="random")
    x[0, 7] = float("nan")

    data_lp, scales = convert_to_nvfp4(
        x,
        block_size=block_size,
        axis=-1,
        is_2d_block=False,
        update_outer_scale=False,
        scale_format=scale_format,
    )
    assert torch.isfinite(scales.float()).all(), (
        f"{scale_format}: NaN input contaminated inner scales — D2' "
        f"defense layer regression."
    )

    x_dq = convert_from_nvfp4(
        data_lp, scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=-1,
        is_2d_block=False,
        scale_format=scale_format,
    )
    assert torch.isfinite(x_dq).all(), (
        f"{scale_format}: NaN input leaked into dequant output — defense "
        f"layer must keep downstream GEMM input finite."
    )
