# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the MX6 fake-quantize wrapper.

MX6 shares Quark's ``fake_quantize_mx6_mx9`` algorithm with MX9, but uses
``quant_bit=5``.  These tests focus on proving that the wrapper selects the MX6
bit width and still preserves the basic QDQ invariants.
"""

import pytest
import torch

from alto.kernels.mx6.format import BLOCK_SIZE, QUANT_BIT
from alto.kernels.mx6.quantize import mx6_fake_quantize


def _quark_mx6_reference(x: torch.Tensor, block_size: int):
    """Live Quark reference; returns None if Quark is unavailable."""
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_mx6_mx9,
        )
    except Exception:
        return None
    return fake_quantize_mx6_mx9(x.clone(), axis=-1, block_size=block_size, quant_bit=QUANT_BIT)


def _rand(shape, dtype):
    torch.manual_seed(6)
    return torch.randn(*shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (2, 3, 32), (3, 40)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mx6_matches_quark_bit_exact(shape, dtype, block_size):
    x = _rand(shape, dtype)
    ref = _quark_mx6_reference(x, block_size)
    if ref is None:
        pytest.skip("Quark reference (fake_quantize_mx6_mx9) not importable")

    out = mx6_fake_quantize(x, block_size=block_size, quant_bit=QUANT_BIT)
    assert out.dtype == ref.dtype
    assert torch.equal(out, ref), (out - ref).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_shape_and_dtype_preserved(dtype):
    x = _rand((4, 64), dtype)
    out = mx6_fake_quantize(x, block_size=BLOCK_SIZE)
    assert out.shape == x.shape
    assert out.dtype == dtype


def test_padding_path_non_divisible_last_dim():
    x = _rand((3, 40), torch.float32)
    out = mx6_fake_quantize(x, block_size=BLOCK_SIZE)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_axis_contract():
    x = _rand((4, 32), torch.float32)
    with pytest.raises(NotImplementedError):
        mx6_fake_quantize(x, block_size=BLOCK_SIZE, axis=0)
