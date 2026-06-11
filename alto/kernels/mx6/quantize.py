# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX6 fake-quantize wrapper.

Quark implements MX6 and MX9 with the same ``fake_quantize_mx6_mx9`` function.
The only format difference in this emulation path is ``quant_bit``:

- MX6: ``quant_bit=5``
- MX9: ``quant_bit=8``

The shared-prime-bit, block reshape, exponent, scale, clamp, and dequant math is
therefore reused from the MX9 implementation.
"""

import torch

from alto.kernels.mx9.quantize import mx9_fake_quantize
from .format import BLOCK_SIZE, QUANT_BIT


def mx6_fake_quantize(
    input_tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    quant_bit: int = QUANT_BIT,
    axis: int = -1,
) -> torch.Tensor:
    """对输入执行 MX6 block-wise fake quantization（QDQ）。"""
    return mx9_fake_quantize(
        input_tensor=input_tensor,
        block_size=block_size,
        quant_bit=quant_bit,
        axis=axis,
    )
