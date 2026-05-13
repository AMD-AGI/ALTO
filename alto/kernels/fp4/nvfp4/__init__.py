# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Package-level entrypoints for NVFP4 kernels.

This subpackage primarily exposes the public quantize/dequantize APIs and the
NVFP4-specific dynamic outer-scale helper.  Low-level Triton helpers remain
available as compatibility aliases but are intentionally omitted from
``__all__``.

Naming convention used throughout this subpackage:

* ``inner_scale`` -- per-block scale stored next to the packed FP4 data
  (NVFP4 spec ``s_block``).
* ``outer_scale`` -- outer-level scale that sits above ``inner_scale``
  (NVFP4 spec ``s_global``); today a per-tensor FP32 scalar, named
  agnostically so a future outer-blockwise layout reuses the same
  surface.

See the docstring at the top of ``nvfp_quantization.py`` for the full
rationale and div-by-zero floor reasoning.
"""

from .nvfp_linear import NVFP4LinearFunction
from .nvfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_SCALE_FORMATS,
    _calculate_nvfp4_scales,
    _pack_fp4,
    _unpack_fp4,
    compute_dynamic_outer_scale,
    compute_outer_block_scale_shape,
    convert_from_nvfp4,
    convert_from_nvfp4_outer_block,
    convert_to_nvfp4,
    convert_to_nvfp4_outer_block,
    is_cdna4,
)

__all__ = (
    "BLOCK_SIZE_DEFAULT",
    "NVFP4LinearFunction",
    "SUPPORTED_SCALE_FORMATS",
    "compute_dynamic_outer_scale",
    "compute_outer_block_scale_shape",
    "convert_from_nvfp4",
    "convert_from_nvfp4_outer_block",
    "convert_to_nvfp4",
    "convert_to_nvfp4_outer_block",
    "is_cdna4",
)
