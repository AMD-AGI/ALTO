# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Unit tests for the midmax scale-selection logic in the mxfp4 quantization kernel.

Baseline reference: ROCm/tensorcast tcast/number.py _decode(), which defines
  emax    = 2^ebits - 1 - bias           (for E2M1: emax = 2)
  maxfloat = 2^emax * (2 - 2^-mbits)    (for E2M1: 6.0)
  midmax  = (2^(emax+1) - maxfloat)/2 + maxfloat  (for E2M1: 7.0)

The kernel uses midmax to decide whether to bump the uint8 scale exponent by 1:
  bump if amax_normalized > 7.0  (i.e. amax_normalized > midmax)
"""

import pytest
import torch

from alto.kernels.fp4.mxfp4.mxfp_quantization import (
    convert_to_mxfp4,
    convert_from_mxfp4,
    is_cdna4,
)
from .utils import (
    prepare_data,
    convert_to_mxfp4_pytorch,
    convert_from_mxfp4_pytorch,
)

# E2M1 constants (baseline from tensorcast number.py)
_EBITS, _MBITS, _BIAS = 2, 1, 1
_EMAX = 2**_EBITS - 1 - _BIAS          # 2
_MAXFLOAT = 2**_EMAX * (2.0 - 2**-_MBITS)   # 6.0
_MIDMAX = (2**(_EMAX + 1) - _MAXFLOAT) / 2.0 + _MAXFLOAT  # 7.0
_TARGET_MAX_POW2 = 2  # matches kernel constant


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_block(value: float, block_size: int = 32, dtype=torch.float32) -> torch.Tensor:
    """Return a 1-D tensor of length block_size filled with `value`, on CUDA."""
    return torch.full((block_size,), value, dtype=dtype, device="cuda")


def _midmax_scale_ref(amax: float, block_size: int = 32) -> int:
    """
    Pure-Python reference for the midmax uint8 scale given a block's amax.
    Mirrors the triton kernel logic in _calculate_scales with USE_MIDMAX=True.
    """
    import struct

    amax_f32 = float(amax)
    bits = struct.unpack("I", struct.pack("f", abs(amax_f32)))[0]
    f32_exp = (bits >> 23) & 0xFF
    if f32_exp >= 0xFF:
        f32_exp = 0xFE  # cap NaN/Inf

    scale = f32_exp - _TARGET_MAX_POW2

    # normalize amax into [2^target_max_pow2, 2^(target_max_pow2+1))
    mantissa = bits & 0x7FFFFF
    norm_bits = mantissa | ((127 + _TARGET_MAX_POW2) << 23)
    amax_scaled = struct.unpack("f", struct.pack("I", norm_bits))[0]

    if amax_scaled > _MIDMAX:
        scale += 1

    scale = max(scale, 1)  # clamp to minimum normal
    return scale


# ---------------------------------------------------------------------------
# 1. Baseline constant sanity
# ---------------------------------------------------------------------------

def test_e2m1_midmax_value():
    """E2M1 midmax must equal 7.0 per the tensorcast baseline formula."""
    assert _MAXFLOAT == 6.0
    assert _MIDMAX == 7.0


# ---------------------------------------------------------------------------
# 2. Scale correctness: midmax vs round-even on crafted inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amax,expect_bump", [
    (6.0, False),   # exactly maxfloat — amax_normalized == 6.0, no bump
    (6.9, False),   # below midmax — no bump
    (7.0, False),   # at midmax (not strictly greater) — no bump
    (7.1, True),    # just above midmax — bump
    (8.0, True),    # well above midmax — bump
    (12.0, True),   # large value — bump
])
def test_midmax_scale_bump(amax, expect_bump):
    """
    Verify the pure-Python reference bumps the scale exactly when
    amax_normalized > 7.0 (strict greater-than, matching the kernel).
    """
    ref_scale = _midmax_scale_ref(amax)
    ref_no_bump = _midmax_scale_ref(6.0)
    ref_bump = _midmax_scale_ref(7.1)
    if expect_bump:
        assert ref_scale == ref_bump, f"amax={amax} expected bump, got scale={ref_scale}"
    else:
        assert ref_scale == ref_no_bump, f"amax={amax} expected no bump, got scale={ref_scale}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_midmax_scale_matches_ref_random(dtype):
    """
    Triton kernel (use_midmax=True) scales must match the pure-Python reference
    on random data.
    """
    torch.manual_seed(42)
    x = torch.randn(128, 64, dtype=dtype, device="cuda")
    _, scales_triton = convert_to_mxfp4(x, use_midmax=True)

    # build reference scales block-by-block
    x_f32 = x.float()
    block_size = 32
    M, N = 128, 64
    scales_ref = torch.zeros(M, N // block_size, dtype=torch.uint8, device="cpu")
    for m in range(M):
        for nb in range(N // block_size):
            block = x_f32[m, nb * block_size:(nb + 1) * block_size]
            amax = block.abs().max().item()
            scales_ref[m, nb] = _midmax_scale_ref(amax)

    assert torch.all(scales_triton.cpu() == scales_ref).item(), \
        "Triton midmax scales differ from pure-Python reference"


# ---------------------------------------------------------------------------
# 3. Kernel output: use_midmax=True vs use_midmax=False differ appropriately
# ---------------------------------------------------------------------------

def test_midmax_differs_from_round_even_on_outliers():
    """
    For blocks whose amax == 7.0 (exactly E2M1 midmax), round-even rounds up
    but midmax does not (strict > comparison), so scales must differ.
    """
    block_size = 32
    x = torch.full((4, 64), 7.0, dtype=torch.float32, device="cuda")

    _, scales_midmax = convert_to_mxfp4(x, use_midmax=True)
    _, scales_round_even = convert_to_mxfp4(x, use_midmax=False)

    assert not torch.equal(scales_midmax, scales_round_even), \
        "Expected scales to differ at amax==7.0 (midmax boundary)"
    # midmax should be strictly lower (no bump) than round-even (bumps at 7.0)
    assert torch.all(scales_midmax < scales_round_even).item(), \
        "midmax scales should be lower than round-even at amax==7.0"


def test_midmax_false_matches_pytorch_ref():
    """
    With use_midmax=False the triton kernel must match the existing pytorch
    reference implementation (which implements round-even only).
    """
    x = prepare_data((128, 64), torch.float32)
    _, scales_triton = convert_to_mxfp4(x, use_midmax=False)
    _, scales_ref = convert_to_mxfp4_pytorch(x)
    assert torch.all(scales_triton == scales_ref).item(), \
        "Triton round-even scales differ from pytorch reference"


# ---------------------------------------------------------------------------
# 4. Quantize → dequantize roundtrip with use_midmax=True
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tensor_shape", [(128, 64), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_midmax_roundtrip(tensor_shape, axis, is_2d_block, dtype):
    """
    Roundtrip (quant → dequant) with use_midmax=True should reconstruct
    the input with reasonable accuracy (MAE within one E2M1 quantum of the
    block scale).
    """
    x = prepare_data(tensor_shape, dtype)
    data_lp, scales = convert_to_mxfp4(x, axis=axis, is_2d_block=is_2d_block, use_midmax=True)
    x_dq = convert_from_mxfp4(data_lp, scales, output_dtype=dtype, axis=axis, is_2d_block=is_2d_block)

    mae = (x.float() - x_dq.float()).abs().mean().item()
    # E2M1 has 4 representable values per octave; 25% relative error is generous
    amax = x.float().abs().max().item()
    assert mae < 0.25 * amax + 1e-3, f"Roundtrip MAE {mae:.4f} too large (amax={amax:.4f})"


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

def test_midmax_all_zeros():
    """A block of all zeros must not produce NaN/Inf scales or outputs."""
    x = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
    data_lp, scales = convert_to_mxfp4(x, use_midmax=True)
    assert not torch.any(torch.isnan(scales.float())), "NaN in scales for zero input"
    assert torch.all(scales >= 1).item(), "scale below minimum-normal clamp"

    x_dq = convert_from_mxfp4(data_lp, scales, output_dtype=torch.float32)
    assert not torch.any(torch.isnan(x_dq)), "NaN in dequantized output for zero input"
    assert torch.all(x_dq == 0).item(), "Zero input should dequantize to zeros"


def test_midmax_large_values():
    """Very large values (near FP32 max) must not produce inf/nan scales."""
    x = torch.full((32, 32), 1e30, dtype=torch.float32, device="cuda")
    data_lp, scales = convert_to_mxfp4(x, use_midmax=True)
    assert torch.all(scales < 255).item(), "scale hit 0xFF (inf/nan exponent)"
    assert not torch.any(torch.isnan(scales.float()))


def test_midmax_negative_values():
    """Negative-valued blocks should produce the same scales as their positive counterpart."""
    torch.manual_seed(7)
    x_pos = torch.abs(torch.randn(64, 64, dtype=torch.float32, device="cuda"))
    x_neg = -x_pos

    _, scales_pos = convert_to_mxfp4(x_pos, use_midmax=True)
    _, scales_neg = convert_to_mxfp4(x_neg, use_midmax=True)
    assert torch.all(scales_pos == scales_neg).item(), \
        "Negating all values should not change midmax scales"


def test_midmax_single_outlier_block():
    """
    A single block containing one value just above midmax (7.1) surrounded by
    small values should bump only that block's scale.
    """
    block_size = 32
    x = torch.ones(1, 2 * block_size, dtype=torch.float32, device="cuda") * 0.5
    # First block: push amax above midmax
    x[0, :block_size] = 7.1

    _, scales = convert_to_mxfp4(x, use_midmax=True)
    scale_bumped_block = scales[0, 0].item()
    scale_small_block = scales[0, 1].item()
    assert scale_bumped_block > scale_small_block, \
        "Block with amax > midmax should have a higher scale than a block with small values"


# ---------------------------------------------------------------------------
# 6. Scale minimum clamp (all-zero block)
# ---------------------------------------------------------------------------

def test_midmax_scale_minimum_clamp():
    """
    The kernel clamps scales to minimum 1 (= 2^1 in FP32 representation).
    An all-zero block exercises this path.
    """
    x = torch.zeros(32, 64, dtype=torch.float32, device="cuda")
    _, scales = convert_to_mxfp4(x, use_midmax=True)
    assert torch.all(scales >= 1).item(), "All scales must be >= 1 (minimum-normal clamp)"


# ---------------------------------------------------------------------------
# 7. Consistency: use_midmax=True gives same result across dtypes (f32 vs bf16)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_asm", [False])
def test_midmax_dtype_consistency(use_asm):
    """
    Quantizing the same tensor in float32 and bfloat16 with use_midmax=True
    should yield scales that are close (within ±1) due to bf16 precision loss.
    """
    if use_asm and not is_cdna4():
        pytest.skip("ASM mode only on CDNA4")

    torch.manual_seed(0)
    x_f32 = torch.randn(64, 64, dtype=torch.float32, device="cuda")
    x_bf16 = x_f32.to(torch.bfloat16)

    _, scales_f32 = convert_to_mxfp4(x_f32, use_midmax=True, use_asm=use_asm)
    _, scales_bf16 = convert_to_mxfp4(x_bf16, use_midmax=True, use_asm=use_asm)

    diff = (scales_f32.int() - scales_bf16.int()).abs()
    assert diff.max().item() <= 1, \
        f"Scale difference between f32 and bf16 exceeded ±1: max={diff.max().item()}"


# ---------------------------------------------------------------------------
# 8. ASM path (CDNA4 only)
# ---------------------------------------------------------------------------

def test_midmax_asm_matches_non_asm():
    """On CDNA4, use_asm=True with use_midmax=True must match use_asm=False."""
    if not is_cdna4():
        pytest.skip("ASM path requires CDNA4 hardware")

    x = prepare_data((128, 64), torch.float32)
    data_lp_asm, scales_asm = convert_to_mxfp4(x, use_midmax=True, use_asm=True)
    data_lp_ref, scales_ref = convert_to_mxfp4(x, use_midmax=True, use_asm=False)

    assert torch.all(scales_asm == scales_ref).item(), "ASM/non-ASM scales differ under midmax"
    assert torch.all(data_lp_asm == data_lp_ref).item(), "ASM/non-ASM fp4 values differ under midmax"
