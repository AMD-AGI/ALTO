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

    two_level_scaling: Literal["none", "tensorwise", "blockwise", "outer_block"] = "none"
    """
    apply an extra scaling factor besides the default blockwise scales of MXFP4/NVFP4.
    * none: no extra scaling
    * tensorwise: apply a global scale factor to the entire tensor (for NVFP4 only)
    * blockwise:
      * MXFP4: apply a blockwise scale factor containing shared mantissa to each block
      * NVFP4: not implemented
    * outer_block: apply a per-super-block FP32 scale before NVFP4 E4M3 block scales
    """
    use_outer_2dblock_x: bool = False
    use_outer_2dblock_w: bool = False
    outer_block_size: int = 128

    align_x_forward_wgrad: bool = False
    """
    paper-recipe X̂ alignment: when ``True`` and outer-block scaling is active
    on a 1D inner-block X (``use_2dblock_x=False``), reuse the forward's
    axis=-1 QDQ of X as the wgrad-axis view instead of running an independent
    axis=0 QDQ.  Matches TetraJet-v2 (arXiv 2510.27527, §D.1-(b)) and is
    mutually exclusive with ``use_hadamard`` (Hadamard rotates X on the M
    axis before axis=0 QDQ — reusing the unrotated forward view would break
    the rotation invariance ``(Hx)ᵀ(Hg) = xᵀg``).
    """
    use_dx_rht: bool = False
    """
    paper-recipe dX RHT: when ``True``, apply a Random Hadamard Transform on
    the N axis to both ``grad_output`` and ``weight`` before the dX GEMM in
    backward (``dX = grad_output @ weight``).  Forward keeps the unrotated
    weight QDQ; an *additional* axis=-1 weight QDQ on ``H^T @ W`` is saved
    for the rotated dX path.  Independent of ``use_hadamard`` (which is the
    M-axis wgrad RHT).  Cf. TetraJet-v2 Table 8(c) "dX RHT" column.
    """


torch.serialization.add_safe_globals([TrainingOpConfig])
