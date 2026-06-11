# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
from dataclasses import dataclass
import torch


@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class TrainingOpConfig:
    precision: Literal["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"]
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

    use_midmax: bool = False


torch.serialization.add_safe_globals([TrainingOpConfig])
