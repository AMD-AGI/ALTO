# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
from dataclasses import dataclass
import torch


TrainingPrecision = Literal["bf16", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"]


@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class PrecisionScheduleEntry:
    precision: TrainingPrecision
    start_step: int
    end_step: int | None = None


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

    mxfp4_scale_selection: Literal["default", "mse_4_6_shifted", "mse_4_6_strict"] = "default"
    """
    MXFP4-only block scale selection.
    * default: keep the existing E8M0 scale calculation.
    * mse_4_6_shifted: compare qmax=6 with a qmax=4-like candidate
      derived from scales_6 + 1.
    * mse_4_6_strict: independently calculate qmax=6 and qmax=4 scales,
      then choose the candidate with lower reconstruction MSE.
    """

    precision_schedule: tuple[PrecisionScheduleEntry, ...] = ()
    """
    Optional step-based precision schedule. Step ranges are [start_step, end_step).
    If no entry matches the current training step, `precision` is used.
    """

torch.serialization.add_safe_globals([PrecisionScheduleEntry, TrainingOpConfig])
