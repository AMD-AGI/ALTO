# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
from dataclasses import dataclass
import torch


@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class TrainingOpConfig:
    precision: Literal["mxfp4", "mxfp8", "nvfp4"]
    use_2dblock_x: bool
    use_2dblock_w: bool
    use_hadamard: bool
    use_sr_grad: bool
    use_dge: bool

    # MXFP4-specific: apply a static clipping scale 3/4 before fp4 quantization
    # for nvfp4, try 16/17 instead of 4/3
    use_static_clip: bool = False

    # this option adds an extra scaling factor besides the default blockwise scales.
    # NVFP4: none or tensorwise
    # MXFP4: none or blockwise (shared mantissa)
    # MXFP8: none
    two_level_scaling: Literal["none", "tensorwise", "blockwise"] = "none"


torch.serialization.add_safe_globals([TrainingOpConfig])
