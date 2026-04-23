# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync
from .cg_backward import mxfp4_grouped_gemm


def _quantize_then_mxfp_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_dge: bool,
    use_hadamard: bool,
) -> torch.Tensor:
    m_indices = create_indices_from_offsets_nosync(offs)
    return mxfp4_grouped_gemm(
        A,
        B,
        m_indices,
        trans_weights=False,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_dge=use_dge,
        use_hadamard=use_hadamard,
    )
