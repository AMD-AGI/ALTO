# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the MX6 dispatch wiring in ``ModelPatcher.patch_fake_quantize``.

The MX6 kernel itself is covered by ``test_mx6_quantize.py``. This file tests the
*wiring* one layer up: after ``patch_fake_quantize()`` replaces
``compressed_tensors...forward.fake_quantize``, a ``QuantizationArgs`` carrying
``format == "mx6"`` must route to ``mx6_fake_quantize``, while plain int8 args
(no ``format``) must fall through to the original implementation untouched.

Also checks that ``registry_patch.inject_format_field()`` makes the ``format``
field survive pydantic validation (otherwise the recipe value is silently
dropped and dispatch never fires).

Run with:
    pytest tests/unittest/mx9_mx6/test_mx6_dispatch.py
"""

import pytest
import torch

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle import forward as forward_module

from alto.kernels.mx6.format import BLOCK_SIZE
from alto.kernels.mx6.quantize import mx6_fake_quantize


def _mx6_args() -> QuantizationArgs:
    return QuantizationArgs.model_validate(
        {
            "num_bits": 5,
            "type": "int",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": True,
            "format": "mx6",
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
    args = _mx6_args()
    assert getattr(args, "format", None) == "mx6"


def test_format_field_defaults_none_for_plain_int8():
    args = _int8_args()
    assert getattr(args, "format", None) is None


# --------------------------------------------------------------------------- #
# dispatch routing
# --------------------------------------------------------------------------- #
def test_mx6_args_dispatch_to_mx6_kernel():
    """format == "mx6" must route fake_quantize to mx6_fake_quantize bit-exact."""
    torch.manual_seed(6)
    x = torch.randn(3, 40, dtype=torch.float32)
    args = _mx6_args()

    patched_out = forward_module.fake_quantize(
        x=x,
        scale=torch.ones(1),
        zero_point=None,
        args=args,
        g_idx=None,
        global_scale=None,
    )
    expected = mx6_fake_quantize(x, block_size=BLOCK_SIZE)

    assert torch.equal(patched_out, expected)


def test_int8_args_fall_through_to_original():
    """Plain int8 (no format) must NOT touch the mx6 kernel; output must match a
    direct INT8 per-tensor QDQ, never the mx6 result."""
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

    # and it must differ from the mx6 path (sanity: int8 did not silently route to mx6)
    mx6_out = mx6_fake_quantize(x, block_size=BLOCK_SIZE)
    assert not torch.equal(patched_out, mx6_out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
