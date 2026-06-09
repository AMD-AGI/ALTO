# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the MX9 dispatch wiring in ``ModelPatcher.patch_fake_quantize``.

The MX9 kernel itself is covered by ``test_mx9_quantize.py``. This file tests the
*wiring* one layer up: after ``patch_fake_quantize()`` replaces
``compressed_tensors...forward.fake_quantize``, a ``QuantizationArgs`` carrying
``format == "mx9"`` must route to ``mx9_fake_quantize``, while plain int8 args
(no ``format``) must fall through to the original implementation untouched.

Also checks that ``registry_patch.inject_format_field()`` makes the ``format``
field survive pydantic validation (otherwise the recipe value is silently
dropped and dispatch never fires).

Run with:
    pytest tests/unittest/mx9_mx6/test_mx9_dispatch.py
"""

import pytest
import torch

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle import forward as forward_module

from alto.kernels.mx9.format import BLOCK_SIZE
from alto.kernels.mx9.quantize import mx9_fake_quantize


def _mx9_args() -> QuantizationArgs:
    return QuantizationArgs.model_validate(
        {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": True,
            "format": "mx9",
        }
    )


def _int8_args() -> QuantizationArgs:
    return QuantizationArgs.model_validate(
        {
            "num_bits": 8,
            "type": "int",
            "symmetric": True,
            "strategy": "tensor",
        }
    )


# --------------------------------------------------------------------------- #
# format field injection (registry_patch)
# --------------------------------------------------------------------------- #
def test_format_field_survives_validation():
    """Without inject_format_field(), pydantic drops the unknown ``format`` key
    and dispatch can never see it."""
    args = _mx9_args()
    assert getattr(args, "format", None) == "mx9"


def test_format_field_defaults_none_for_plain_int8():
    args = _int8_args()
    assert getattr(args, "format", None) is None


# --------------------------------------------------------------------------- #
# dispatch routing
# --------------------------------------------------------------------------- #
def test_mx9_args_dispatch_to_mx9_kernel():
    """format == "mx9" must route fake_quantize to mx9_fake_quantize bit-exact."""
    torch.manual_seed(0)
    x = torch.randn(3, 40, dtype=torch.float32)
    args = _mx9_args()

    patched_out = forward_module.fake_quantize(
        x=x,
        scale=torch.ones(1),
        zero_point=None,
        args=args,
        g_idx=None,
        global_scale=None,
    )
    expected = mx9_fake_quantize(x, block_size=BLOCK_SIZE)

    assert torch.equal(patched_out, expected)


def test_int8_args_fall_through_to_original():
    """Plain int8 (no format) must NOT touch the mx9 kernel; output must match a
    direct INT8 per-tensor QDQ, never the mx9 result."""
    torch.manual_seed(1)
    x = torch.randn(3, 40, dtype=torch.float32)
    args = _int8_args()

    # per-tensor symmetric int8 scale (what the original kernel would use)
    qmax = 127.0
    scale = (x.abs().max() / qmax).clamp_min(1e-12).reshape(1)
    zp = torch.zeros(1, dtype=torch.int32)

    patched_out = forward_module.fake_quantize(
        x=x,
        scale=scale,
        zero_point=zp,
        args=args,
        g_idx=None,
        global_scale=None,
    )

    expected_int8 = (torch.round(x / scale).clamp(-128, 127)) * scale
    assert torch.allclose(patched_out, expected_int8, atol=1e-5)

    # and it must differ from the mx9 path (sanity: int8 did not silently route to mx9)
    mx9_out = mx9_fake_quantize(x, block_size=BLOCK_SIZE)
    assert not torch.equal(patched_out, mx9_out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
