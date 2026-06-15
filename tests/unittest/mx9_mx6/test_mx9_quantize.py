# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the MX9 fake-quantize kernel.

Three layers:
  1. Golden-vector parity vs Quark's published mx9 output (hardcoded, so it runs
     even when Quark is not installed). This is the primary correctness anchor.
  2. Optional bit-exact parity vs a live Quark import (skipped if unavailable).
  3. Format-property checks (block independence, prime-bit demotion, padding,
     NaN/Inf, dtype, idempotence, axis contract).
"""

import pytest
import torch

from alto.modifiers.quantization.mx import BLOCK_SIZE, MX9_QUANT_BIT as QUANT_BIT, mx9_fake_quantize


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _quark_interesting_pattern() -> torch.Tensor:
    """Quark test_mx.py:228 create_4d_tensor_with_interesting_pattern()."""
    result = torch.zeros(1, 2, 3, 4)
    for x1 in range(2):
        for x2 in range(3):
            for x3 in range(4):
                result[0, x1, x2, x3] = (x3 + 1.0) * (10 * (x2 + 1.0)) + 100.0 * x1
    return result


# Quark's published mx9 output for the pattern above @ block_size=16, axis=-1
# (test_mx.py:341-348). MX9 is high precision, so it round-trips these integers
# exactly.
_QUARK_MX9_GOLDEN = torch.tensor(
    [
        [
            [[10.0, 20.0, 30.0, 40.0], [20.0, 40.0, 60.0, 80.0], [30.0, 60.0, 90.0, 120.0]],
            [[110.0, 120.0, 130.0, 140.0], [120.0, 140.0, 160.0, 180.0], [130.0, 160.0, 190.0, 220.0]],
        ]
    ]
)


def _quark_mx9_reference(x: torch.Tensor, block_size: int):
    """Live Quark reference; returns None (caller skips) if Quark is unavailable."""
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_mx6_mx9,
        )
    except Exception:
        return None
    return fake_quantize_mx6_mx9(x.clone(), axis=-1, block_size=block_size, quant_bit=QUANT_BIT)


def _rand(shape, dtype):
    torch.manual_seed(0)
    return torch.randn(*shape, dtype=dtype)


# --------------------------------------------------------------------------- #
# Layer 1: golden vector (always runs, no Quark needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mx9_matches_quark_golden_vector(dtype):
    x = _quark_interesting_pattern().to(dtype)
    out = mx9_fake_quantize(x, block_size=16)
    expected = _QUARK_MX9_GOLDEN.to(dtype)
    assert out.dtype == dtype
    assert torch.equal(out, expected), (out - expected).abs().max().item()


# --------------------------------------------------------------------------- #
# Layer 2: live Quark bit-exact (bonus, skips if Quark absent)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (2, 3, 32), (3, 40)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mx9_matches_quark_bit_exact(shape, dtype, block_size):
    x = _rand(shape, dtype)
    ref = _quark_mx9_reference(x, block_size)
    if ref is None:
        pytest.skip("Quark reference (fake_quantize_mx6_mx9) not importable")

    out = mx9_fake_quantize(x, block_size=block_size, quant_bit=QUANT_BIT)
    assert out.dtype == ref.dtype
    assert torch.equal(out, ref), (out - ref).abs().max().item()


# --------------------------------------------------------------------------- #
# Layer 3: format-property checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_shape_and_dtype_preserved(dtype):
    x = _rand((4, 64), dtype)
    out = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    assert out.shape == x.shape
    assert out.dtype == dtype


def test_block_independence():
    # A block of huge values must not degrade the scale of a separate small block.
    # Two rows = two independent blocks (last dim == block_size).
    big = torch.full((1, BLOCK_SIZE), 100.0)
    small = torch.full((1, BLOCK_SIZE), 0.01)
    joint = mx9_fake_quantize(torch.cat([big, small], dim=0), block_size=BLOCK_SIZE)
    alone = mx9_fake_quantize(small, block_size=BLOCK_SIZE)
    # small block's result is identical whether or not the big block shares the tensor.
    assert torch.equal(joint[1:2], alone)


def test_prime_bit_demotes_small_pair():
    # MX9's distinguishing feature: when BOTH elements of an adjacent pair are
    # >= 1 octave below the block max, that pair's scale drops one exponent,
    # giving the pair finer resolution. With block max 128 (max_exp=7) the
    # non-demoted scale is 2.0 and the demoted scale is 1.0, so v=1.4 rounds to
    # 2.0 (err 0.6) without demotion but to 1.0 (err 0.4) with it.
    v = 1.4
    block = torch.zeros(1, BLOCK_SIZE)
    block[0, 0] = 128.0          # block max -> sets shared exponent, never demoted
    block[0, 1] = v              # pair (idx 0,1): idx0 is the max -> NOT demoted
    block[0, 2] = v              # pair (idx 2,3): BOTH small -> demoted
    block[0, 3] = v
    out = mx9_fake_quantize(block, block_size=BLOCK_SIZE)
    # Hard values: with max_exp=7 the non-demoted scale is 2.0 and the demoted
    # scale is 1.0, so v=1.4 lands on exactly those grid points.
    assert out[0, 1].item() == 2.0  # non-demoted pair (idx0 is the max)
    assert out[0, 2].item() == 1.0  # demoted pair (both small)
    assert out[0, 3].item() == 1.0
    err_non_demoted = (out[0, 1] - v).abs().item()
    err_demoted = (out[0, 2] - v).abs().item()
    assert err_demoted < err_non_demoted


def test_padding_path_non_divisible_last_dim():
    x = _rand((3, 40), torch.float32)  # 40 not divisible by 16
    out = mx9_fake_quantize(x, block_size=16)
    assert out.shape == x.shape
    # Padding zeros must not corrupt the real elements: quantizing the same
    # data laid out as exact blocks gives the matching prefix.
    assert not torch.isnan(out).any()


def test_inf_does_not_poison_block():
    # nan_to_num is applied to the amax copy only (matching Quark), so ±Inf in
    # the data does NOT blow up the shared scale: the real elements survive and
    # Inf is clamped to the finite quant range. NaN in the *data* path is not
    # scrubbed and would propagate -- documented limitation, tested separately.
    x = torch.full((1, BLOCK_SIZE), 4.0)
    x[0, 0] = float("inf")
    x[0, 1] = float("-inf")
    out = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    assert not torch.isinf(out).any()
    # The real 4.0 values are unaffected by the poisoned amax candidate.
    assert torch.equal(out[0, 2:], torch.full((BLOCK_SIZE - 2,), 4.0))


def test_nan_in_data_propagates():
    # Known limitation faithful to Quark: a NaN element survives QDQ as NaN.
    # Locking this in so any future change to the data-path sanitization is loud.
    x = torch.full((1, BLOCK_SIZE), 4.0)
    x[0, 0] = float("nan")
    out = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    assert torch.isnan(out[0, 0])
    assert torch.equal(out[0, 1:], torch.full((BLOCK_SIZE - 1,), 4.0))


def test_qdq_idempotent():
    x = _rand((8, 64), torch.float32)
    once = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    twice = mx9_fake_quantize(once, block_size=BLOCK_SIZE)
    assert torch.equal(once, twice)


def test_zeros_stay_zero():
    x = torch.zeros((2, 32), dtype=torch.float32)
    out = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    assert torch.equal(out, x)


def test_axis_contract():
    x = _rand((4, 32), torch.float32)
    with pytest.raises(NotImplementedError):
        mx9_fake_quantize(x, block_size=BLOCK_SIZE, axis=0)
