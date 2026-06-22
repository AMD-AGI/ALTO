# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 grouped-GEMM Python entrypoints.

Mirrors :mod:`alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.functional` but
pins ``scale_format='ue5m3'`` so neither the AMD-FP4 dispatch layer nor
training recipes need to thread the dtype literal through call sites.
"""

from __future__ import annotations

import torch

from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import (
    _quantize_then_nvfp4_scaled_grouped_mm,
    nvfp4_grouped_gemm,
)


def amdfp4_grouped_gemm(
    inputs: torch.Tensor,          # [M_total, K]
    expert_weights: torch.Tensor,  # [E, N, K] if trans_weights else [E, K, N]
    expert_indices: torch.Tensor,  # [M_total]  int32
    *,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = False,
    use_sr_grad: bool = True,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """AMD-FP4 (UE5M3 inner scale) Grouped GEMM with full autograd support.

    Same parameter contract as
    :func:`alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.nvfp4_grouped_gemm`
    minus ``scale_format``, which is hard-pinned to ``"ue5m3"``.
    """
    return nvfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
        scale_format="ue5m3",
    )


def _quantize_then_amdfp4_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,               # [E, K, N] per dispatch convention
    offs: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """AMD-FP4 dispatch-side variant of
    :func:`alto.kernels.fp4.nvfp4.nvfp_grouped_gemm._quantize_then_nvfp4_scaled_grouped_mm`.

    ``scale_format`` is hard-pinned to ``"ue5m3"`` so the MoE dispatch
    code path can route AMD-FP4 traffic by op surface alone.
    """
    return _quantize_then_nvfp4_scaled_grouped_mm(
        A,
        B,
        offs=offs,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
        scale_format="ue5m3",
    )
