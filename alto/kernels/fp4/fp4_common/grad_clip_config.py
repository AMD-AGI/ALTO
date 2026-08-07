# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass


@dataclass
class GradClipConfig:
    clip_grad_output: bool = False
    grad_output_max_norm: float | None = None
    grad_output_clip_value: float | None = None

    clip_grad_weight: bool = False
    grad_weight_max_norm: float | None = None
    grad_weight_clip_value: float | None = None
