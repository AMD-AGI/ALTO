# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Package-level entrypoints for NVFP4 kernels.

This subpackage primarily exposes the public quantize/dequantize APIs and the
NVFP4-specific dynamic per-tensor scale helper. Low-level Triton helpers remain
available as compatibility aliases but are intentionally omitted from
``__all__``.
"""

from .nvfp_linear import NVFP4LinearFunction, _to_nvfp4_then_linear
from .nvfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_SCALE_FORMATS,
    _calculate_nvfp4_scales,
    _pack_fp4,
    _unpack_fp4,
    compute_dynamic_per_tensor_scale,
    convert_from_nvfp4,
    convert_to_nvfp4,
    is_cdna4,
)

__all__ = (
    "BLOCK_SIZE_DEFAULT",
    "NVFP4LinearFunction",
    "SUPPORTED_SCALE_FORMATS",
    "_to_nvfp4_then_linear",
    "compute_dynamic_per_tensor_scale",
    "convert_from_nvfp4",
    "convert_to_nvfp4",
    "is_cdna4",
)
