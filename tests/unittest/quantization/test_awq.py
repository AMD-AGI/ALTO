# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from alto.modifiers.quantization.awq import _pseudo_quantize_tensor


def test_pseudo_quantize_stacked_weights_without_group_size():
    weight = torch.randn(2, 3, 4)

    quantized = _pseudo_quantize_tensor(weight)
    expected = torch.stack([_pseudo_quantize_tensor(part) for part in weight])

    assert quantized.shape == weight.shape
    assert torch.equal(quantized, expected)
