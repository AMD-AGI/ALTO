# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX packed-quantization Triton kernels for the MX6 and MX9 block formats.

Fake-quant reference / emulation lives in ``alto.modifiers.quantization.mx``.
This package provides GPU pack/unpack only.
"""

from ._mx_common import convert_to_mx, convert_from_mx
from .mx6_quantization import QUANT_BIT as _MX6_QUANT_BIT, convert_to_mx6, convert_from_mx6
from .mx9_quantization import QUANT_BIT as _MX9_QUANT_BIT, convert_to_mx9, convert_from_mx9

MX_QUANT_BIT = {"mx6": _MX6_QUANT_BIT, "mx9": _MX9_QUANT_BIT}

__all__ = [
    "MX_QUANT_BIT",
    "convert_to_mx",
    "convert_from_mx",
    "convert_to_mx6",
    "convert_from_mx6",
    "convert_to_mx9",
    "convert_from_mx9",
]
