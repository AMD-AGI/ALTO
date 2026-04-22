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
    # NVFP4-specific: apply a two-level (per-tensor × per-block) scale on top
    # of the E4M3 block scales.  Ignored by precisions whose scale scheme
    # already absorbs the global dynamic range (e.g. MXFP4's E8M0 scales),
    # hence the default of ``False`` which preserves current behaviour for
    # every non-NVFP4 scheme.
    use_per_tensor_scale: bool = False


torch.serialization.add_safe_globals([TrainingOpConfig])
