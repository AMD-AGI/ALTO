# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
import torch


@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class TrainingOpConfig:
    precision: str
    use_2dblock_x: bool
    use_2dblock_w: bool
    use_hadamard: bool
    use_sr_grad: bool
    use_dge: bool


torch.serialization.add_safe_globals([TrainingOpConfig])
