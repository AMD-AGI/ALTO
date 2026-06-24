# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 kernels.

The AMD-FP4 recipe is **NVFP4-style E2M1 storage with a UE5M3 inner-block
scale and an FP32 per-tensor outer scale**, aligned with GFXIPARCH-2067
§19.10 / OCP E5M3.  In other words: same micro-block layout as NVFP4,
strictly wider inner-scale dynamic range (max normal 114688 vs 448 for
E4M3) at the same mantissa precision.

This sub-package is a peer of :mod:`alto.kernels.fp4.nvfp4`; both build
on the shared body in :mod:`alto.kernels.fp4.outer_scaled_fp4`.  The split
exists so:

* recipe-level call sites (``convert_to_amdfp4``, ``AMDFP4LinearFunction``,
  ``_quantize_then_amdfp4_scaled_grouped_mm``) pin ``scale_format`` to
  ``"ue5m3"`` exactly once and never expose the dtype switch to user code;
* dispatch / tracing tooling that keys on the ATen op id sees AMD-FP4 as
  a distinct operator (``alto::convert_to_amdfp4``) and can route /
  benchmark it independently from NVFP4.
"""

from .amdfp_grouped_gemm import (
    ALIGN_SIZE_M,
    AMDFP4GroupedGEMM,
    _quantize_then_amdfp4_scaled_grouped_mm,
    amdfp4_grouped_gemm,
)
from .amdfp_linear import AMDFP4LinearFunction, _to_amdfp4_then_scaled_mm
from .amdfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    convert_from_amdfp4,
    convert_to_amdfp4,
)

__all__ = (
    "ALIGN_SIZE_M",
    "AMDFP4GroupedGEMM",
    "AMDFP4LinearFunction",
    "BLOCK_SIZE_DEFAULT",
    "_quantize_then_amdfp4_scaled_grouped_mm",
    "_to_amdfp4_then_scaled_mm",
    "amdfp4_grouped_gemm",
    "convert_from_amdfp4",
    "convert_to_amdfp4",
)
