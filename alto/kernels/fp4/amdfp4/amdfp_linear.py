# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 linear autograd surface.

The autograd function itself is identical to the NVFP4 path —
``NVFP4LinearFunction`` already accepts a ``scale_format`` parameter that
selects E4M3 vs UE5M3 inner scales.  This module re-exposes it under an
AMD-FP4 name and provides a thin ``_to_amdfp4_then_scaled_mm`` helper
that pins ``scale_format='ue5m3'`` so user code never spells the dtype
literal.
"""

from __future__ import annotations

import torch

from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _to_nvfp4_then_scaled_mm,
)
from alto.kernels.fp4.nvfp4.nvfp_quantization import OUTER_BLOCK_SIZE_DEFAULT


# Re-export the autograd function under an AMD-FP4 name.  We deliberately
# do NOT subclass: ``torch.autograd.Function`` works through a custom
# metaclass + class methods, and a marker subclass would create a
# *separate* autograd record key without changing behaviour.  Keep this
# as a simple alias so isinstance-checks against the NVFP4 type still
# match (the dispatch layer benefits from this single source of truth).
AMDFP4LinearFunction = NVFP4LinearFunction


def _to_amdfp4_then_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
    use_outer_block_scale: bool = False,
    use_outer_2dblock_x: bool = False,
    use_outer_2dblock_w: bool = False,
    outer_block_size: int = OUTER_BLOCK_SIZE_DEFAULT,
) -> torch.Tensor:
    """AMD-FP4 (UE5M3 inner scale) variant of ``_to_nvfp4_then_scaled_mm``.

    Mirrors the NVFP4 helper exactly -- including the outer-block knobs --
    except ``scale_format`` is hard-pinned to ``"ue5m3"`` so the AMD-FP4
    dispatch layer / training recipes can address the AMD-FP4 path by op
    surface alone.  Outer-block scaling shares the NVFP4 code path; only the
    inner-block scale grid differs (UE5M3 vs E4M3), which the outer-scale
    denominator accounts for.
    """
    return _to_nvfp4_then_scaled_mm(
        a,
        b,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
        use_outer_block_scale=use_outer_block_scale,
        use_outer_2dblock_x=use_outer_2dblock_x,
        use_outer_2dblock_w=use_outer_2dblock_w,
        outer_block_size=outer_block_size,
        scale_format="ue5m3",
    )


__all__ = (
    "AMDFP4LinearFunction",
    "_to_amdfp4_then_scaled_mm",
)
