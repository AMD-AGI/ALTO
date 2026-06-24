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
    Quantization recipe family, and the **single source of truth** for the
    FP4 inner-scale grid.  ``"nvfp4"`` and ``"amdfp4"`` are peers with the
    same micro-block layout but different inner-scale grids:

    * ``"nvfp4"``  — E4M3 inner scale (NVFP4 spec ``s_block``).
    * ``"amdfp4"`` — UE5M3 inner scale (GFXIPARCH-2067 §19.10 / OCP E5M3,
      max normal 114688, NaN at 0xFF; ~256× wider dynamic range than E4M3
      at the same mantissa precision).

    The inner-scale dtype is fully determined by ``precision`` — there is no
    separate user-facing knob.  Dispatch selects the matching kernel family
    directly from this field (``nvfp4`` → E4M3 ops, ``amdfp4`` → UE5M3 ops).
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


torch.serialization.add_safe_globals([TrainingOpConfig])
