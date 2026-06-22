# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
from dataclasses import dataclass
import torch


@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class TrainingOpConfig:
    precision: Literal["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4", "amdfp4"]
    """
    Quantization recipe family.  ``"nvfp4"`` and ``"amdfp4"`` are peers with
    the same micro-block layout but different inner-scale grids:

    * ``"nvfp4"``  — E4M3 inner scale by default (NVFP4 spec ``s_block``).
    * ``"amdfp4"`` — UE5M3 inner scale (GFXIPARCH-2067 §19.10 / OCP E5M3,
      max normal 114688, NaN at 0xFF).  See ``inner_scale_format`` below
      for how this selection is forwarded to the kernels.
    """
    use_2dblock_x: bool
    use_2dblock_w: bool
    use_hadamard: bool
    use_sr_grad: bool
    use_dge: bool

    clip_mode: Literal["none", "static", "dynamic"] = "none"
    """
    clipping mode applied in MXFP4/NVFP4 quantization.
    * none: no clipping
    * static: apply a static clipping scale 3/4 before mxfp4 quantization (16/17 for nvfp4, but not implemented)
    * dynamic: dynamic clipped scale based on the mean and std of the input tensor (only supported for mxfp4)
    """

    two_level_scaling: Literal["none", "tensorwise", "blockwise"] = "none"
    """
    apply an extra scaling factor besides the default blockwise scales of MXFP4/NVFP4.
    * none: no extra scaling
    * tensorwise: apply a global scale factor to the entire tensor (for NVFP4 only)
    * blockwise:
      * MXFP4: apply a blockwise scale factor containing shared mantissa to each block
      * NVFP4: not implemented
    """

    inner_scale_format: Literal["e4m3", "ue5m3"] = "e4m3"
    """
    Per-block inner-scale dtype used by NVFP4 / AMD-FP4 (orthogonal to ``precision``).

    * ``"e4m3"`` — NVFP4 default: signed FP8 E4M3, max 448.
    * ``"ue5m3"`` — AMD-FP4: unsigned 8-bit (5 exp + 3 mant, NaN code 0xFF
      per GFXIPARCH-2067 §19.10, no sign bit, no Inf encoding), max normal
      114688 at code 0xFE; ~256× wider dynamic range than E4M3 at the same
      mantissa precision.

    Selected when ``precision in {"nvfp4", "amdfp4"}``; ignored for
    MXFP4 / MXFP8.  When ``precision == "amdfp4"`` this field is forced to
    ``"ue5m3"`` in :meth:`__post_init__` (AMD-FP4 = NVFP4 spec + UE5M3
    inner scale).

    Locked design choice (see ``docs/amd-fp4/agent-handoff.md`` D3): keep the
    ``precision`` literal carrying the recipe and let this orthogonal field
    carry the inner-scale dtype, so additional dtypes (E8M3, UE4M4, …) can
    be added by extending this Literal alone.
    """

    def __post_init__(self) -> None:
        # Pin the AMD-FP4 recipe to its UE5M3 inner-scale dtype.  Allowed
        # caller-side aliases:
        #   - ``inner_scale_format`` left at its dataclass default ``"e4m3"``
        #     (i.e. caller didn't touch it; we silently override),
        #   - ``inner_scale_format="ue5m3"`` explicitly (no-op).
        # Any other value is rejected so AMD-FP4 cannot accidentally be
        # routed to the E4M3 grid by an out-of-date caller.
        if self.precision == "amdfp4":
            if self.inner_scale_format == "e4m3":
                # Not a frozen dataclass, so plain assignment is fine even with
                # ``slots=True`` (the slot descriptor accepts writes).
                self.inner_scale_format = "ue5m3"
            elif self.inner_scale_format != "ue5m3":
                raise ValueError(
                    f"precision='amdfp4' requires inner_scale_format='ue5m3'; "
                    f"got inner_scale_format={self.inner_scale_format!r}"
                )


torch.serialization.add_safe_globals([TrainingOpConfig])
