# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 grouped GEMM surface (UE5M3 inner scale).

Like :mod:`alto.kernels.fp4.amdfp4.amdfp_linear`, this is a thin layer
on top of the NVFP4 grouped-GEMM machinery: the autograd function and
all loop / native-dispatch helpers are reused as-is and the
``scale_format`` literal is hard-pinned to ``"ue5m3"`` at the AMD-FP4
boundary.
"""

from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import (
    ALIGN_SIZE_M,
    NVFP4GroupedGEMM,
)

from .functional import (
    _quantize_then_amdfp4_scaled_grouped_mm,
    amdfp4_grouped_gemm,
)


# Re-export the autograd Function under an AMD-FP4 name; same alias
# rationale as ``AMDFP4LinearFunction`` -- the underlying
# ``torch.autograd.Function`` already routes ``scale_format``, so we
# avoid creating a parallel marker subclass that would only complicate
# isinstance checks at the dispatch layer.
AMDFP4GroupedGEMM = NVFP4GroupedGEMM


__all__ = (
    "ALIGN_SIZE_M",
    "AMDFP4GroupedGEMM",
    "_quantize_then_amdfp4_scaled_grouped_mm",
    "amdfp4_grouped_gemm",
)
