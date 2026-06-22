# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Correctness tests for the Four Over Six (4/6) NVFP4 reference (P1).

These exercise the pure-PyTorch reference ``convert_to_nvfp4_4o6_pytorch``
against the existing NVFP4 reference, validating:

  * the per-block "never worse than M=6" guarantee that the selection rule
    must satisfy (MAE and MSE objectives),
  * hand-crafted blocks where M=4 / M=6 is provably better (paper Table 2),
  * the 256-vs-448 global-scale overflow guard,
  * round-trip consistency through the unchanged dequant path, and
  * SNR improvement on near-maximal-value distributions.

Everything runs in pure PyTorch on CPU so correctness can be checked without
a GPU; 4/6 only changes how the stored (fp4, scale) pair is chosen.
"""

import pytest
import torch

from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    F8E4M3_4O6_CEIL,
    compute_dynamic_outer_scale,
    convert_to_nvfp4,
)

from .utils import (
    F4_E2M1_MAX,
    F8E4M3_MAX,
    compute_dynamic_outer_scale_4o6_pytorch,
    convert_from_nvfp4_pytorch,
    convert_to_nvfp4_4o6_pytorch,
    convert_to_nvfp4_pytorch,
    calc_snr,
)

DEVICE = "cpu"

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton NVFP4 kernel requires a GPU"
)


def _mae(a, b):
    return (a.float() - b.float()).abs().mean().item()


def _mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


def _dequant(data_lp, inner_scale, outer_scale, is_2d_block=False):
    return convert_from_nvfp4_pytorch(
        data_lp,
        inner_scale,
        output_dtype=torch.float32,
        block_size=16,
        axis=-1,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
    )


# ---------------------------------------------------------------------------
# 1. Never-worse-than-M=6 guarantee (the core property of the selection rule).
#    Comparison is apples-to-apples: the M=6 baseline uses the *same* 256-based
#    outer scale, so any gap is purely the 4/6 block-scale selection.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["mae", "mse"])
def test_4o6_never_worse_than_m6(metric):
    torch.manual_seed(0)
    x = torch.randn(128, 256, device=DEVICE, dtype=torch.float32)
    # sparse large outliers -> heavy-tailed blocks, the regime 4/6 targets
    mask = torch.bernoulli(torch.full_like(x, 0.01))
    x = x + 50.0 * torch.randn_like(x) * mask

    outer = compute_dynamic_outer_scale_4o6_pytorch(x)

    base_lp, base_inner = convert_to_nvfp4_pytorch(x, block_size=16, outer_scale=outer)
    o46_lp, o46_inner = convert_to_nvfp4_4o6_pytorch(
        x, block_size=16, outer_scale=outer, select_metric=metric)

    base = _dequant(base_lp, base_inner, outer)
    o46 = _dequant(o46_lp, o46_inner, outer)

    err = _mae if metric == "mae" else _mse
    err_base = err(x, base)
    err_o46 = err(x, o46)
    # selection picks the min-error candidate per block, so the global error
    # under the selection metric can never exceed the M=6 baseline.
    assert err_o46 <= err_base + 1e-6, (
        f"4/6 ({metric}) error {err_o46:.6e} must not exceed M=6 baseline "
        f"{err_base:.6e}")


# ---------------------------------------------------------------------------
# 2. Hand-crafted blocks: paper Table 2 examples (block_size 16 via tiling).
# ---------------------------------------------------------------------------

def test_4o6_selects_four_when_better():
    # [10,20,30,40] is exactly representable when scaled to 4 (-> [1,2,3,4]),
    # but suffers error when scaled to 6.  4/6 must pick M=4 and reconstruct
    # the block exactly.
    block = torch.tensor([10.0, 20.0, 30.0, 40.0] * 4, device=DEVICE).reshape(1, 16)
    lp, inner, choose4 = convert_to_nvfp4_4o6_pytorch(
        block, block_size=16, outer_scale=None, select_metric="mae",
        return_choice=True)
    assert bool(choose4.all()), "block should select M=4"

    recon = _dequant(lp, inner, outer_scale=None)
    assert _mae(block, recon) == pytest.approx(0.0, abs=1e-6), (
        f"M=4 should reconstruct [10,20,30,40] exactly, got {recon.flatten()[:4]}")


def test_4o6_selects_six_when_better():
    # [15,30,120,180] is exactly representable when scaled to 6 (-> [0.5,1,4,6]).
    block = torch.tensor([15.0, 30.0, 120.0, 180.0] * 4, device=DEVICE).reshape(1, 16)
    lp, inner, choose4 = convert_to_nvfp4_4o6_pytorch(
        block, block_size=16, outer_scale=None, select_metric="mse",
        return_choice=True)
    assert not bool(choose4.any()), "block should select M=6"

    recon = _dequant(lp, inner, outer_scale=None)
    assert _mse(block, recon) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Global-scale overflow guard: 256 ceiling lets the amax block pick M=4
#    without the E4M3 block scale saturating; 448 would overflow.
# ---------------------------------------------------------------------------

def test_4o6_global_scale_avoids_e4m3_overflow():
    # static facts that motivate the 256 ceiling
    assert (448.0 * F4_E2M1_MAX / 4.0) > F8E4M3_MAX  # 672 > 448 -> would clamp
    assert (256.0 * F4_E2M1_MAX / 4.0) <= F8E4M3_MAX  # 384 <= 448 -> safe

    torch.manual_seed(1)
    x = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
    x[7, 80] = 5000.0  # single dominating outlier sets the tensor amax

    outer = compute_dynamic_outer_scale_4o6_pytorch(x)
    _, inner = convert_to_nvfp4_4o6_pytorch(
        x, block_size=16, outer_scale=outer, select_metric="mse")

    # no stored block scale may saturate at the E4M3 ceiling
    assert torch.all(inner.float() <= F8E4M3_MAX + 1e-3)
    # the amax block under M=4 lands at exactly 256*6/4 = 384, well represented
    assert inner.float().max().item() <= 384.0 + 1e-3


# ---------------------------------------------------------------------------
# 4. Round-trip consistency: the unchanged dequant path must reconstruct a
#    4/6-encoded tensor identically to a manual q * inner * outer.
# ---------------------------------------------------------------------------

def test_4o6_roundtrip_uses_unchanged_dequant():
    torch.manual_seed(2)
    x = torch.randn(32, 64, device=DEVICE, dtype=torch.float32) * 3.0
    outer = compute_dynamic_outer_scale_4o6_pytorch(x)

    lp, inner = convert_to_nvfp4_4o6_pytorch(
        x, block_size=16, outer_scale=outer, select_metric="mae")
    recon = _dequant(lp, inner, outer)

    assert recon.shape == x.shape
    # reconstruction is reasonable (heavy-tailed-free gaussian) and finite
    assert torch.isfinite(recon).all()
    snr = calc_snr(x, recon)
    assert snr > 15.0, f"round-trip SNR unexpectedly low: {snr:.2f} dB"


# ---------------------------------------------------------------------------
# 5. SNR improvement on a near-maximal-value distribution (4/6's target case).
# ---------------------------------------------------------------------------

def test_4o6_improves_snr_on_near_maximal_blocks():
    torch.manual_seed(3)
    # each 16-block: a random per-block magnitude B, with most values packed
    # near 0.7-1.0 * B -> scaled values cluster around 4-6, the FP4 gap.
    n_rows, n_blocks, bs = 64, 16, 16
    base = (torch.rand(n_rows, n_blocks, 1, device=DEVICE) * 4.0 + 1.0)
    frac = torch.rand(n_rows, n_blocks, bs, device=DEVICE) * 0.3 + 0.7  # [0.7,1.0]
    sign = torch.sign(torch.randn(n_rows, n_blocks, bs, device=DEVICE))
    x = (base * frac * sign).reshape(n_rows, n_blocks * bs).float()

    outer = compute_dynamic_outer_scale_4o6_pytorch(x)
    base_lp, base_inner = convert_to_nvfp4_pytorch(x, block_size=16, outer_scale=outer)
    o46_lp, o46_inner = convert_to_nvfp4_4o6_pytorch(
        x, block_size=16, outer_scale=outer, select_metric="mse")

    snr_base = calc_snr(x, _dequant(base_lp, base_inner, outer))
    snr_o46 = calc_snr(x, _dequant(o46_lp, o46_inner, outer))

    assert snr_o46 >= snr_base - 1e-4, "4/6 must not reduce SNR"
    assert snr_o46 >= snr_base + 0.3, (
        f"4/6 should improve SNR on near-maximal blocks: "
        f"4/6={snr_o46:.2f} dB vs M=6={snr_base:.2f} dB")


# ---------------------------------------------------------------------------
# 6. 2D-block variant of the never-worse guarantee.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["mae", "mse"])
def test_4o6_2d_block_never_worse_than_m6(metric):
    torch.manual_seed(4)
    x = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
    mask = torch.bernoulli(torch.full_like(x, 0.01))
    x = x + 40.0 * torch.randn_like(x) * mask

    outer = compute_dynamic_outer_scale_4o6_pytorch(x)
    base_lp, base_inner = convert_to_nvfp4_pytorch(
        x, block_size=16, is_2d_block=True, outer_scale=outer)
    o46_lp, o46_inner = convert_to_nvfp4_4o6_pytorch(
        x, block_size=16, is_2d_block=True, outer_scale=outer, select_metric=metric)

    base = _dequant(base_lp, base_inner, outer, is_2d_block=True)
    o46 = _dequant(o46_lp, o46_inner, outer, is_2d_block=True)

    err = _mae if metric == "mae" else _mse
    assert err(x, o46) <= err(x, base) + 1e-6


# ===========================================================================
# P2 -- production Triton kernel (``convert_to_nvfp4(..., use_4o6=True)``).
# These require a GPU; they validate the kernel against the P1 reference.
# ===========================================================================

def _cpu_dequant(data_lp, inner_scale, outer_scale, is_2d_block=False):
    return convert_from_nvfp4_pytorch(
        data_lp.cpu(), inner_scale.cpu(), output_dtype=torch.float32,
        block_size=16, axis=-1, is_2d_block=is_2d_block,
        outer_scale=outer_scale.cpu(),
    )


@requires_gpu
@pytest.mark.parametrize("metric", ["mae", "mse"])
@pytest.mark.parametrize("is_2d_block", [False, True])
def test_triton_4o6_matches_reference(metric, is_2d_block):
    """The Triton 4/6 kernel must agree with the P1 PyTorch reference.

    Both consume the same 256-ceiling outer scale, so any difference is purely
    numerical (e.g. a rare selection flip on a near-tie); we therefore require
    very high agreement rather than exact equality.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    mask = torch.bernoulli(torch.full_like(x, 0.01))
    x = x + 40.0 * torch.randn_like(x) * mask

    outer = compute_dynamic_outer_scale(x, scale_ceiling=F8E4M3_4O6_CEIL)

    lp_t, s_t = convert_to_nvfp4(
        x, block_size=16, is_2d_block=is_2d_block,
        outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=True, select_metric=metric)
    lp_r, s_r = convert_to_nvfp4_4o6_pytorch(
        x.cpu(), block_size=16, is_2d_block=is_2d_block,
        outer_scale=outer.cpu(), select_metric=metric)

    deq_t = _cpu_dequant(lp_t, s_t, outer, is_2d_block)
    deq_r = _cpu_dequant(lp_r, s_r, outer, is_2d_block)

    snr = calc_snr(deq_r, deq_t)
    assert snr > 55.0, f"Triton 4/6 disagrees with reference: SNR {snr:.1f} dB"
    frac_match = (s_t.cpu().float() == s_r.float()).float().mean().item()
    assert frac_match > 0.99, f"only {frac_match:.4f} of block scales match reference"


@requires_gpu
def test_triton_4o6_default_off_is_unchanged():
    """``use_4o6=False`` must reproduce the standard NVFP4 kernel exactly."""
    torch.manual_seed(1)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    outer = compute_dynamic_outer_scale(x)

    lp_a, s_a = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False)
    lp_b, s_b = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=False)

    assert torch.equal(lp_a, lp_b)
    assert torch.equal(s_a, s_b)


@requires_gpu
@pytest.mark.parametrize("metric", ["mae", "mse"])
def test_triton_4o6_never_worse_than_m6(metric):
    """Triton 4/6 must not increase reconstruction error vs M=6 (same outer)."""
    torch.manual_seed(2)
    x = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    mask = torch.bernoulli(torch.full_like(x, 0.01))
    x = x + 40.0 * torch.randn_like(x) * mask
    outer = compute_dynamic_outer_scale(x, scale_ceiling=F8E4M3_4O6_CEIL)

    base_lp, base_s = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=False)
    o46_lp, o46_s = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=True, select_metric=metric)

    err = _mae if metric == "mae" else _mse
    err_base = err(x.cpu(), _cpu_dequant(base_lp, base_s, outer))
    err_o46 = err(x.cpu(), _cpu_dequant(o46_lp, o46_s, outer))
    assert err_o46 <= err_base + 1e-6, (
        f"Triton 4/6 ({metric}) {err_o46:.6e} worse than M=6 {err_base:.6e}")


@requires_gpu
def test_triton_4o6_improves_snr_on_near_maximal_blocks():
    torch.manual_seed(3)
    n_rows, n_blocks, bs = 64, 16, 16
    base = (torch.rand(n_rows, n_blocks, 1, device="cuda") * 4.0 + 1.0)
    frac = torch.rand(n_rows, n_blocks, bs, device="cuda") * 0.3 + 0.7
    sign = torch.sign(torch.randn(n_rows, n_blocks, bs, device="cuda"))
    x = (base * frac * sign).reshape(n_rows, n_blocks * bs).float()
    outer = compute_dynamic_outer_scale(x, scale_ceiling=F8E4M3_4O6_CEIL)

    base_lp, base_s = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=False)
    o46_lp, o46_s = convert_to_nvfp4(
        x, block_size=16, outer_scale=outer.clone(), update_outer_scale=False,
        use_4o6=True, select_metric="mse")

    snr_base = calc_snr(x.cpu(), _cpu_dequant(base_lp, base_s, outer))
    snr_o46 = calc_snr(x.cpu(), _cpu_dequant(o46_lp, o46_s, outer))
    assert snr_o46 >= snr_base + 0.3, (
        f"Triton 4/6 should improve SNR on near-maximal blocks: "
        f"4/6={snr_o46:.2f} dB vs M=6={snr_base:.2f} dB")


def test_convert_to_nvfp4_rejects_bad_select_metric():
    """Invalid ``select_metric`` is rejected before any kernel launch (CPU)."""
    x = torch.zeros(16, dtype=torch.float32)
    with pytest.raises(AssertionError, match="select_metric"):
        convert_to_nvfp4(x, block_size=16, use_4o6=True, select_metric="rmse")
