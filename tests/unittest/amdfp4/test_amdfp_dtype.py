# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Bit-level oracle tests for the UE5M3 PyTorch primitives (AMD-FP4 inner scale).

UE5M3 is the inner-scale dtype of the AMD-FP4 recipe; this file owns the
dtype-layer regression so ``tests/unittest/amdfp4/`` is the single home
for every AMD-FP4 layer (dtype primitive → quant op → linear/grouped GEMM
→ A/B matrix).

Coverage (D2', GFXIPARCH-2067 §19.10 aligned):

* Constants vs. closed-form ground truth (UE5M3_MAX = 114688 at code 0xFE).
* All 256 uint8 codes decode to the expected fp32 value (manual table);
  code 0xFF decodes to NaN (verified with isnan, NaN ≠ NaN bypass).
* Round-trip: every code re-encodes to itself (identity-on-grid),
  including the 0xFF NaN code.
* Idempotency: ``quantize(quantize(x)) == quantize(x)`` (NaN positions
  match between successive quantizations).
* Saturation: NaN / +Inf / -Inf / overflow → 0xFF (NaN); finite negatives
  → 0x00; ``UE5M3_MAX`` exact → 0xFE (max normal).
* Round-to-nearest-even spot checks on between-grid values.

The CUDA path is not exercised here; that's the job of
``test_amdfp_triton_pytorch_parity.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from alto.kernels.fp4.fp4_common import (
    UE5M3_EPS,
    UE5M3_EXP_BIAS,
    UE5M3_MAX,
    UE5M3_MIN_NORMAL,
    UE5M3_NAN_CODE,
    UE5M3_NUM_EXP_BITS,
    UE5M3_NUM_MAN_BITS,
    f32_to_ue5m3_uint8,
    quantize_to_ue5m3,
    ue5m3_uint8_to_f32,
)


# ---------------------------------------------------------------------------
# Ground-truth 256-code value table.
# Independently computed from the documented spec, used as the oracle.
# ---------------------------------------------------------------------------

def _build_ue5m3_value_table() -> list[float]:
    """Return ``table[i]`` = exact fp32 value for UE5M3 code ``i``.

    Per D2' / GFXIPARCH-2067 §19.10, code 0xFF is reserved for NaN.
    """
    table: list[float] = []
    for code in range(256):
        if code == UE5M3_NAN_CODE:
            table.append(float("nan"))
            continue
        exp = code >> UE5M3_NUM_MAN_BITS
        mant = code & ((1 << UE5M3_NUM_MAN_BITS) - 1)
        if exp == 0:
            if mant == 0:
                value = 0.0
            else:
                value = (mant / (1 << UE5M3_NUM_MAN_BITS)) * (2.0 ** (1 - UE5M3_EXP_BIAS))
        else:
            value = (1.0 + mant / (1 << UE5M3_NUM_MAN_BITS)) * (
                2.0 ** (exp - UE5M3_EXP_BIAS)
            )
        table.append(value)
    return table


_VALUE_TABLE = _build_ue5m3_value_table()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants_match_closed_form():
    assert UE5M3_NUM_EXP_BITS == 5
    assert UE5M3_NUM_MAN_BITS == 3
    assert UE5M3_EXP_BIAS == 15
    # Max normal at code 0xFE: (1 + 6/8) * 2^16 = 1.75 * 65536 = 114688
    assert UE5M3_MAX == 114688.0
    assert UE5M3_MAX == 1.75 * (2.0 ** 16)
    # Min normal: 2^-14
    assert UE5M3_MIN_NORMAL == math.ldexp(1.0, -14)
    # Smallest subnormal: 2^-17
    assert UE5M3_EPS == math.ldexp(1.0, -17)
    # NaN code reserved per GFXIPARCH-2067 §19.10
    assert UE5M3_NAN_CODE == 0xFF


def test_dynamic_range_increase_vs_e4m3():
    """Sanity: ``UE5M3_MAX / E4M3_MAX`` is the AMD-FP4 dynamic-range gain.

    Closed form (D2', spec-aligned): ``(1.75 * 2^16) / (1.75 * 2^8) = 256``.
    """
    e4m3_max = 448.0
    ratio = UE5M3_MAX / e4m3_max
    assert abs(ratio - (1.75 / 1.75) * 256.0) < 1e-9
    assert ratio == 256.0
    assert ratio > 250.0


def test_value_table_endpoints():
    """Spot-check the most important entries of the manual decode table."""
    assert _VALUE_TABLE[0x00] == 0.0
    assert _VALUE_TABLE[0x01] == math.ldexp(1.0, -17)            # smallest subnormal
    assert _VALUE_TABLE[0x07] == 7.0 * math.ldexp(1.0, -17)      # largest subnormal
    assert _VALUE_TABLE[0x08] == math.ldexp(1.0, -14)            # smallest normal
    # Code 0x78 = exp=15, mant=0 -> (1 + 0) * 2^(15-15) = 1.0
    assert _VALUE_TABLE[0x78] == 1.0
    # Code 0xFE = exp=31, mant=6 -> (1 + 6/8) * 2^16 = 114688 (max normal)
    assert _VALUE_TABLE[0xFE] == 114688.0
    assert _VALUE_TABLE[0xFE] == UE5M3_MAX
    # Code 0xFF: NaN (D2', spec-aligned).
    assert math.isnan(_VALUE_TABLE[0xFF])


# ---------------------------------------------------------------------------
# Decode (uint8 -> fp32) bit-exact on all 256 codes
# ---------------------------------------------------------------------------

def test_decode_all_256_codes_bit_exact():
    codes = torch.arange(256, dtype=torch.uint8)
    decoded = ue5m3_uint8_to_f32(codes)
    expected = torch.tensor(_VALUE_TABLE, dtype=torch.float32)

    # Code 0xFF is NaN under D2'; ``torch.equal`` returns False for NaN
    # positions, so we mask 0xFF out and assert NaN-ness separately.
    finite_mask = torch.arange(256, dtype=torch.uint8) != UE5M3_NAN_CODE
    assert torch.equal(decoded[finite_mask], expected[finite_mask]), (
        "UE5M3 decode mismatch on the finite 0x00-0xFE grid: "
        f"first diff at code 0x{int((decoded[finite_mask] != expected[finite_mask]).nonzero()[0].item()):02X}"
    )
    # Code 0xFF MUST decode to NaN.
    assert torch.isnan(decoded[UE5M3_NAN_CODE]).item(), (
        f"Code 0xFF must decode to NaN under D2'; got {float(decoded[UE5M3_NAN_CODE])}"
    )


# ---------------------------------------------------------------------------
# Round-trip: every code re-encodes to itself
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip_all_codes():
    """``encode(decode(code)) == code`` for every 8-bit pattern, including 0xFF.

    For the NaN code 0xFF the round-trip is: decode -> NaN, then re-encode
    classifies NaN as a "special" input (D2') and emits 0xFF again.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    decoded = ue5m3_uint8_to_f32(codes)
    re_encoded = f32_to_ue5m3_uint8(decoded)
    assert torch.equal(codes, re_encoded), (
        "Round-trip (decode + encode) is not identity on the 256-code grid; "
        f"first diff at code 0x{int((codes != re_encoded).nonzero()[0].item()):02X}, "
        f"re-encoded as 0x{int(re_encoded[(codes != re_encoded).nonzero()[0].item()].item()):02X}"
    )


# ---------------------------------------------------------------------------
# Idempotency: quantize(quantize(x)) == quantize(x) bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(8,), (128,), (1024,), (4, 256), (3, 17, 31)])
def test_idempotency_random(size):
    """``quantize(quantize(x)) == quantize(x)`` bit-for-bit on finite values.

    Under D2' a value > UE5M3_MAX encodes to the 0xFF NaN code, which decodes
    to NaN.  We therefore check finite-vs-finite equality and NaN-mask
    equality separately (since ``torch.equal`` returns False on NaN).
    """
    torch.manual_seed(0)
    # Mix positive normal range with subnormal and saturation candidates.
    x = torch.randn(size, dtype=torch.float32).abs() * 10.0
    x[..., 0] = 0.0                       # exact zero
    x[..., -1] = UE5M3_MAX * 0.5          # mid-range
    if x.numel() >= 3:
        x.view(-1)[1] = UE5M3_EPS * 0.5   # below smallest subnormal -> rounds to 0 or eps
        x.view(-1)[2] = UE5M3_MAX * 2.0   # overflow -> NaN code (D2')
    q1 = quantize_to_ue5m3(x)
    q2 = quantize_to_ue5m3(q1)
    nan1 = torch.isnan(q1)
    nan2 = torch.isnan(q2)
    assert torch.equal(nan1, nan2), (
        "quantize_to_ue5m3 NaN mask not preserved across two passes; "
        f"shape={tuple(size)}, diff at idx={int((nan1 ^ nan2).nonzero()[0].item())}"
    )
    finite_mask = ~nan1
    assert torch.equal(q1[finite_mask], q2[finite_mask]), (
        "quantize_to_ue5m3 is not idempotent on finite values; "
        f"shape={tuple(size)}, max diff="
        f"{float((q1[finite_mask] - q2[finite_mask]).abs().max()):.3e}"
    )


# ---------------------------------------------------------------------------
# Saturation / clamp policy
# ---------------------------------------------------------------------------

def test_saturation_nan_inf_negative_large():
    """D2' saturation policy (GFXIPARCH-2067 §19.10):

    * NaN, ±Inf, finite values strictly above ``UE5M3_MAX`` -> 0xFF (NaN)
    * Finite negatives                                       -> 0x00
    * 0.0                                                    -> 0x00
    * ``UE5M3_MAX`` exact                                    -> 0xFE (max normal)
    """
    x = torch.tensor(
        [float("nan"), float("inf"), -float("inf"),
         -1.0, -1e30, 0.0,
         UE5M3_MAX, UE5M3_MAX * 2.0, 1e30],
        dtype=torch.float32,
    )
    codes = f32_to_ue5m3_uint8(x)
    expected = torch.tensor(
        # NaN  +Inf  -Inf  -1.0 -1e30  0.0  MAX  2*MAX  1e30
        [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0xFF],
        dtype=torch.uint8,
    )
    assert torch.equal(codes, expected), (
        f"Saturation policy violated; codes={[hex(c) for c in codes.tolist()]}, "
        f"expected={[hex(c) for c in expected.tolist()]}"
    )


def test_zero_clean():
    """All-zero input must encode to 0x00 and decode back to exact 0.0."""
    x = torch.zeros(64, dtype=torch.float32)
    codes = f32_to_ue5m3_uint8(x)
    assert torch.equal(codes, torch.zeros(64, dtype=torch.uint8))
    assert torch.equal(ue5m3_uint8_to_f32(codes), x)


# ---------------------------------------------------------------------------
# D2' spec-aligned NaN / max-normal contract
# ---------------------------------------------------------------------------

def test_decode_0xff_is_nan():
    """Code 0xFF MUST decode to a quiet FP32 NaN (GFXIPARCH-2067 §19.10)."""
    codes = torch.tensor([UE5M3_NAN_CODE], dtype=torch.uint8)
    decoded = ue5m3_uint8_to_f32(codes)
    assert torch.isnan(decoded).all().item(), (
        f"0xFF must decode to NaN, got {decoded.tolist()}"
    )


def test_max_normal_at_0xfe_decodes_114688():
    """Code 0xFE MUST decode to UE5M3_MAX = 114688.0 (max normal)."""
    codes = torch.tensor([0xFE], dtype=torch.uint8)
    decoded = ue5m3_uint8_to_f32(codes)
    assert decoded.item() == 114688.0
    assert decoded.item() == UE5M3_MAX


def test_encode_max_value_lands_on_0xfe():
    """Encoding the exact max-normal value MUST land on 0xFE, NOT on 0xFF."""
    x = torch.tensor([UE5M3_MAX], dtype=torch.float32)
    code = f32_to_ue5m3_uint8(x).item()
    assert code == 0xFE, (
        f"Expected encode(UE5M3_MAX) == 0xFE under D2', got 0x{code:02X}"
    )


def test_overflow_above_max_encodes_to_nan():
    """Any finite value strictly above UE5M3_MAX MUST encode to 0xFF (NaN)."""
    x = torch.tensor(
        [UE5M3_MAX * 2.0, 1e6, 1e30, float("inf")],
        dtype=torch.float32,
    )
    codes = f32_to_ue5m3_uint8(x).tolist()
    assert codes == [0xFF] * 4, (
        f"Overflow-above-MAX must encode to 0xFF, got {[hex(c) for c in codes]}"
    )


def test_quantize_to_ue5m3_propagates_nan():
    """``quantize_to_ue5m3(NaN)`` MUST emit NaN (D2' round-trip)."""
    x = torch.tensor(
        [float("nan"), 1.0, UE5M3_MAX * 2.0, 0.0],
        dtype=torch.float32,
    )
    q = quantize_to_ue5m3(x)
    assert torch.isnan(q[0]).item(), "NaN input must produce NaN output"
    assert q[1].item() == 1.0, "1.0 should round-trip cleanly"
    assert torch.isnan(q[2]).item(), "Overflow must produce NaN output (D2')"
    assert q[3].item() == 0.0, "0.0 must round-trip to 0.0"


# ---------------------------------------------------------------------------
# Round-to-nearest-even spot checks on between-grid inputs
# ---------------------------------------------------------------------------

def test_rtne_tie_to_even():
    """Inputs exactly between two UE5M3 grid points must round to the
    representation with an even least-significant mantissa bit.

    Grid near 1.0:
        code 0x77 = exp=14, mant=7 -> (1+7/8)*2^-1 = 0.9375
        code 0x78 = exp=15, mant=0 -> 1.0
        code 0x79 = exp=15, mant=1 -> (1+1/8)*2^0 = 1.125

    Halfway between 1.0 and 1.125 is 1.0625.  Both neighbors are equidistant
    -> tie -> round to even mantissa LSB -> code 0x78 (mant=0, even).
    """
    x = torch.tensor([0.9375, 1.0, 1.0625, 1.125], dtype=torch.float32)
    codes = f32_to_ue5m3_uint8(x)
    # 0.9375 and 1.0 and 1.125 are exact grid points -> identity.
    # 1.0625 is the tie -> 0x78 (mant=0 is even).
    expected = torch.tensor([0x77, 0x78, 0x78, 0x79], dtype=torch.uint8)
    assert torch.equal(codes, expected), (
        f"RTNE tie-to-even broken: codes={codes.tolist()}, expected={expected.tolist()}"
    )


def test_rtne_no_tie_rounds_to_nearest():
    """1.05 is closer to 1.0 than to 1.125 -> rounds down."""
    x = torch.tensor([1.05, 1.1], dtype=torch.float32)
    codes = f32_to_ue5m3_uint8(x)
    assert codes[0].item() == 0x78, f"1.05 should round to 1.0, got code 0x{int(codes[0].item()):02X}"
    # 1.1 is closer to 1.125 than to 1.0 (diff 0.025 vs 0.1) -> rounds up.
    assert codes[1].item() == 0x79, f"1.1 should round to 1.125, got code 0x{int(codes[1].item()):02X}"


def test_rtne_subnormal_boundary():
    """Just-below-MIN_NORMAL inputs should land on subnormal grid points,
    not silently snap to 0 or to the smallest normal."""
    # smallest subnormal = 2^-17; smallest normal = 2^-14 = 8 * 2^-17
    # Pick an input that's exactly between subnormals:
    #   2^-17 (mant=1) and 2 * 2^-17 (mant=2)
    # The midpoint is 1.5 * 2^-17; with RTNE tie -> mant=2 (even)
    midpoint = 1.5 * math.ldexp(1.0, -17)
    just_below = math.ldexp(1.0, -17) * 1.4    # closer to mant=1
    just_above = math.ldexp(1.0, -17) * 1.6    # closer to mant=2
    x = torch.tensor([midpoint, just_below, just_above], dtype=torch.float32)
    codes = f32_to_ue5m3_uint8(x)
    # midpoint -> 0x02 (mant=2 is even); just_below -> 0x01; just_above -> 0x02
    expected = torch.tensor([0x02, 0x01, 0x02], dtype=torch.uint8)
    assert torch.equal(codes, expected), (
        f"Subnormal RTNE broken: codes={codes.tolist()}, expected={expected.tolist()}"
    )


def test_rtne_below_smallest_subnormal_rounds_to_zero():
    """Inputs below half the smallest subnormal must round to 0."""
    x = torch.tensor(
        [UE5M3_EPS * 0.4, UE5M3_EPS * 0.49, math.ldexp(1.0, -30)],
        dtype=torch.float32,
    )
    codes = f32_to_ue5m3_uint8(x)
    assert torch.all(codes == 0).item(), (
        f"Inputs < 0.5 * UE5M3_EPS should round to 0, got codes={codes.tolist()}"
    )


# ---------------------------------------------------------------------------
# Coverage of the full encoding map on a dense sweep
# ---------------------------------------------------------------------------

def test_encode_grid_values_identity():
    """Every UE5M3 grid value (including 0xFF -> NaN -> 0xFF), when fed
    back to the encoder, must produce its own code -- a stronger version
    of `roundtrip_all_codes` that exercises the encoder's path selection
    (normal vs denormal vs saturate vs special) on real fp32 numerics."""
    codes = torch.arange(256, dtype=torch.uint8)
    values = ue5m3_uint8_to_f32(codes)
    re_encoded = f32_to_ue5m3_uint8(values)
    diffs = (codes != re_encoded).nonzero(as_tuple=False).flatten().tolist()
    assert not diffs, (
        f"Encoder is not identity on grid values; offending codes: "
        + ", ".join(f"0x{c:02X} (got 0x{int(re_encoded[c].item()):02X})" for c in diffs[:8])
    )
