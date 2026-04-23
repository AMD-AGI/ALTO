# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified autograd core for NVFP4 Grouped GEMM."""

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.dge import dge_bwd
from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.fp4.nvfp4.nvfp_quantization import _qdq, convert_from_nvfp4
from alto.kernels.hadamard_transform import HadamardTransform
from .autotune import ALIGN_SIZE_M
from .utils import (
    _check_grouped_axis0_qdq_contract,
    _check_grouped_loop_contract,
    _resolve_expert_indices,
    _use_cdna4_grouped_backend,
)
from .cg_forward import _nvfp4_grouped_mm_2d_3d
from .cg_backward import _nvfp4_grouped_wgrad


@torch.compiler.allow_in_graph
class NVFP4GroupedGEMM(torch.autograd.Function):
    """Unified NVFP4 grouped GEMM autograd implementation.

    Public entry modes:
      * expert_indices-based API
      * offs-based dispatch API

    Backend selection:
      * CDNA4 -> Triton grouped backend
      * otherwise -> Python loop fallback
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: Optional[torch.Tensor],
        offs: Optional[torch.Tensor],
        trans_weights: bool,
        use_2dblock_x: bool,
        use_2dblock_w: bool,
        use_sr_grad: bool,
        use_per_tensor_scale: bool,
        hadamard_transform: Optional[HadamardTransform] = None,
        use_dge: bool = False,
    ) -> torch.Tensor:
        M_total = inputs.shape[0]
        original_dtype = inputs.dtype
        expert_weights = unwrap_weight_wrapper(expert_weights)
        expert_indices = _resolve_expert_indices(expert_indices=expert_indices, offs=offs, M_total=M_total)
        num_experts = expert_weights.shape[0]
        _check_grouped_axis0_qdq_contract(M_total, where="NVFP4GroupedGEMM.forward")
        num_groups = None
        if not _use_cdna4_grouped_backend():
            _check_grouped_loop_contract(M_total, where="NVFP4GroupedGEMM.forward")
            num_groups = M_total // ALIGN_SIZE_M

        x_dq = _qdq(inputs, axis=-1, is_2d_block=use_2dblock_x, use_per_tensor_scale=use_per_tensor_scale)
        quant_axis_w = -1 if trans_weights else -2
        w_dq = _qdq(expert_weights, axis=quant_axis_w, is_2d_block=use_2dblock_w, use_per_tensor_scale=use_per_tensor_scale)

        w_for_fprop = w_dq.transpose(-2, -1).contiguous() if trans_weights else w_dq
        y = _nvfp4_grouped_mm_2d_3d(
            x_dq,
            w_for_fprop,
            expert_indices=expert_indices,
            num_groups=num_groups,
            output_dtype=original_dtype,
        )

        requant_axis_w = -2 if trans_weights else -1
        w_raw_fp4 = None
        w_raw_scales = None
        w_bwd = (
            _qdq(
                expert_weights,
                axis=requant_axis_w,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
                return_raw=use_dge,
            ) if not use_2dblock_w else w_dq
        )
        if use_dge and not use_2dblock_w:
            w_bwd, w_raw_fp4, w_raw_scales = w_bwd
        elif use_dge:
            _, w_raw_fp4, w_raw_scales = _qdq(
                expert_weights,
                axis=quant_axis_w,
                is_2d_block=True,
                use_per_tensor_scale=use_per_tensor_scale,
                return_raw=True,
            )

        if not use_2dblock_x:
            x_for_wgrad = hadamard_transform(inputs, left_mul=True) if hadamard_transform is not None else inputs
            x_bwd = _qdq(
                x_for_wgrad,
                axis=0,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
            )
        else:
            x_bwd = x_dq

        if use_dge:
            ctx.save_for_backward(x_bwd, w_bwd, expert_indices, w_raw_fp4, w_raw_scales)
        else:
            ctx.save_for_backward(x_bwd, w_bwd, expert_indices)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.use_sr_grad = use_sr_grad
        ctx.use_per_tensor_scale = use_per_tensor_scale
        ctx.trans_weights = trans_weights
        ctx.num_experts = num_experts
        ctx.num_groups = num_groups
        ctx.original_dtype = original_dtype
        ctx.hadamard_transform = hadamard_transform
        ctx.use_dge = use_dge
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_dge:
            x_bwd, w_bwd, expert_indices, w_raw_fp4, w_raw_scales = ctx.saved_tensors
        else:
            x_bwd, w_bwd, expert_indices = ctx.saved_tensors

        g_dq = _qdq(
            grad_output,
            axis=-1,
            is_2d_block=ctx.use_2dblock_x,
            use_per_tensor_scale=ctx.use_per_tensor_scale,
            use_sr=ctx.use_sr_grad,
        )

        w_for_dgrad = w_bwd if ctx.trans_weights else w_bwd.transpose(-2, -1).contiguous()
        grad_inputs = _nvfp4_grouped_mm_2d_3d(
            g_dq,
            w_for_dgrad,
            expert_indices=expert_indices,
            num_groups=ctx.num_groups,
            output_dtype=ctx.original_dtype,
        )

        grad_output_for_wgrad = (
            ctx.hadamard_transform(grad_output, left_mul=True)
            if (not ctx.use_2dblock_x and ctx.hadamard_transform is not None)
            else grad_output
        )
        grad_weights = _nvfp4_grouped_wgrad(
            grad_output=grad_output_for_wgrad,
            x_bwd=x_bwd,
            expert_indices=expert_indices,
            num_experts=ctx.num_experts,
            num_groups=ctx.num_groups if ctx.num_groups is not None else expert_indices.shape[0] // ALIGN_SIZE_M,
            use_sr_grad=ctx.use_sr_grad,
            use_per_tensor_scale=ctx.use_per_tensor_scale,
            use_2dblock_x=ctx.use_2dblock_x,
            trans_weights=ctx.trans_weights,
            output_dtype=ctx.original_dtype,
        )

        if ctx.use_dge:
            requant_axis_w = -2 if ctx.trans_weights else -1
            quant_axis_w = -1 if ctx.trans_weights else -2
            w_fp4_values = convert_from_nvfp4(
                w_raw_fp4,
                torch.ones_like(w_raw_scales),
                output_dtype=ctx.original_dtype,
                axis=quant_axis_w if ctx.use_2dblock_w else requant_axis_w,
                is_2d_block=ctx.use_2dblock_w,
                per_tensor_scale=None,
            )
            grad_weights = grad_weights * dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)

        return grad_inputs, grad_weights, None, None, None, None, None, None, None, None, None
