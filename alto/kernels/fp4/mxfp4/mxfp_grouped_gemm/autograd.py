# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch/torchtitan licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

"""MXFP4 Grouped GEMM autograd core.

``MXFP4GroupedGEMM`` used to live in ``cg_backward.py`` alongside the Triton
kernels themselves.  Hosting it in its own module mirrors the layered layout
used by the NVFP4 grouped package (``autograd.py`` / ``functional.py`` /
``cg_forward.py`` / ``cg_backward.py``) and keeps ``cg_backward.py`` focused
on the primitives the autograd class consumes.
"""

from typing import Optional

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_forward import cg_grouped_gemm_forward
from alto.kernels.blockwise_fp8.grouped_gemm.cg_backward import (
    cg_grouped_gemm_backward_inputs,
    cg_grouped_gemm_backward_weights,
)
from alto.kernels.dge import dge_bwd
from alto.kernels.fp4.mxfp4.mxfp_quantization import BLOCK_SIZE_DEFAULT, is_cdna4
from alto.kernels.hadamard_transform.transform import HadamardTransform

from .cg_backward import (
    mxfp4_grouped_gemm_backward_inputs,  # noqa: F401 — import registers torchtitan::mxfp4_grouped_gemm_backward_inputs
    mxfp4_grouped_gemm_backward_weights,  # noqa: F401 — import registers torchtitan::mxfp4_grouped_gemm_backward_weights
)
from .cg_forward import mxfp4_grouped_gemm_forward  # noqa: F401 — import registers torchtitan::mxfp4_grouped_gemm_forward


class MXFP4GroupedGEMM(torch.autograd.Function):
    """Autograd function for contiguous MXFP4 grouped GEMM."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=True,
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_sr_grad=False,
        hadamard_transform: Optional[HadamardTransform] = None,
        use_dge=False,
    ):
        """Forward pass for contiguous grouped GEMM."""
        original_dtype = inputs.dtype

        inputs_mxfp4, input_scales = torch.ops.torchtitan.convert_to_mxfp4(
            inputs,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )
        if trans_weights:
            quant_axis_w = -1
            requant_axis_w = -2
        else:
            quant_axis_w = -2
            requant_axis_w = -1

        expert_weights_mxfp4, expert_weight_scales = torch.ops.torchtitan.convert_to_mxfp4(
            expert_weights,
            axis=quant_axis_w,
            is_2d_block=use_2dblock_w,
        )

        if is_cdna4():
            res = torch.ops.torchtitan.mxfp4_grouped_gemm_forward(
                inputs=inputs_mxfp4,
                expert_weights=expert_weights_mxfp4,
                expert_indices=expert_indices,
                input_scales=input_scales,
                weight_scales=expert_weight_scales,
                trans_weights=trans_weights,
                use_2dblock_x=use_2dblock_x,
                use_2dblock_w=use_2dblock_w,
                output_dtype=original_dtype,
            )
        else:
            x_dq = torch.ops.torchtitan.convert_from_mxfp4(
                inputs_mxfp4,
                input_scales,
                original_dtype,
                axis=-1,
                is_2d_block=use_2dblock_x,
            ).contiguous()
            w_dq = torch.ops.torchtitan.convert_from_mxfp4(
                expert_weights_mxfp4,
                expert_weight_scales,
                original_dtype,
                axis=quant_axis_w,
                is_2d_block=use_2dblock_w,
            )
            if not trans_weights:
                w_dq = w_dq.transpose(-2, -1)
            w_dq = w_dq.contiguous()
            res = cg_grouped_gemm_forward(x_dq, w_dq, expert_indices)

        if not use_2dblock_w:
            expert_weights_mxfp4, expert_weight_scales = torch.ops.torchtitan.convert_to_mxfp4(
                expert_weights,
                axis=requant_axis_w,
                is_2d_block=False,
            )
            if not is_cdna4():
                w_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    expert_weights_mxfp4,
                    expert_weight_scales,
                    original_dtype,
                    axis=requant_axis_w,
                    is_2d_block=False,
                )
                if not trans_weights:
                    w_dq = w_dq.transpose(-2, -1)
                w_dq = w_dq.contiguous()
        if not use_2dblock_x:
            if hadamard_transform is not None:
                inputs = hadamard_transform(inputs, left_mul=True)
            inputs_mxfp4, input_scales = torch.ops.torchtitan.convert_to_mxfp4(
                inputs,
                axis=0,
                is_2d_block=False,
            )
            if not is_cdna4():
                x_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    inputs_mxfp4,
                    input_scales,
                    original_dtype,
                    axis=0,
                    is_2d_block=use_2dblock_x,
                ).contiguous()

        # Save for backward
        if is_cdna4():
            ctx.save_for_backward(inputs_mxfp4, input_scales, expert_weights_mxfp4, expert_weight_scales,
                                  expert_indices)
        else:
            ctx.save_for_backward(x_dq, w_dq, expert_indices, expert_weights_mxfp4)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.trans_weights = trans_weights
        ctx.original_dtype = original_dtype
        ctx.use_sr_grad = use_sr_grad
        ctx.use_dge = use_dge
        ctx.hadamard_transform = hadamard_transform

        return res

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for contiguous grouped GEMM."""
        if is_cdna4():
            inputs, input_scales, expert_weights, expert_weight_scales, expert_indices = ctx.saved_tensors
            expert_weights_mxfp4 = expert_weights
        else:
            x_dq, expert_weights, expert_indices, expert_weights_mxfp4 = ctx.saved_tensors
        # Get number of experts
        num_experts = expert_weights.shape[0]
        if ctx.trans_weights:
            quant_axis_w = -1
            requant_axis_w = -2
        else:
            quant_axis_w = -2
            requant_axis_w = -1

        if ctx.use_2dblock_x:
            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=-1,
                use_sr=ctx.use_sr_grad,
                is_2d_block=True,
            )
            grad_output_mxfp4_m = grad_output_mxfp4
            grad_output_scales_m = grad_output_scales

            if not is_cdna4():
                grad_output_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4,
                    grad_output_scales,
                    ctx.original_dtype,
                    axis=-1,
                    is_2d_block=True,
                ).contiguous()
                grad_output_m_dq = grad_output_dq
        else:
            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=-1,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
            )
            if ctx.hadamard_transform is not None:
                grad_output = ctx.hadamard_transform(grad_output, left_mul=True)
            grad_output_mxfp4_m, grad_output_scales_m = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output,
                axis=0,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
            )

            if not is_cdna4():
                grad_output_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4,
                    grad_output_scales,
                    ctx.original_dtype,
                    axis=-1,
                    is_2d_block=False,
                ).contiguous()
                grad_output_m_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4_m,
                    grad_output_scales_m,
                    ctx.original_dtype,
                    axis=0,
                    is_2d_block=False,
                ).contiguous()

        # Compute gradients
        if is_cdna4():
            grad_inputs = torch.ops.torchtitan.mxfp4_grouped_gemm_backward_inputs(
                grad_output=grad_output_mxfp4,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                go_scales=grad_output_scales,
                expert_weight_scales=expert_weight_scales,
                trans_weights=ctx.trans_weights,
                use_2dblock_x=ctx.use_2dblock_x,
                use_2dblock_w=ctx.use_2dblock_w,
                k_pack_w=not ctx.use_2dblock_w,
                output_dtype=ctx.original_dtype,
            )

            grad_weights = torch.ops.torchtitan.mxfp4_grouped_gemm_backward_weights(
                grad_output=grad_output_mxfp4_m,
                inputs=inputs,
                expert_indices=expert_indices,
                num_experts=num_experts,
                go_scales=grad_output_scales_m,
                input_scales=input_scales,
                trans_weights=ctx.trans_weights,
                use_2dblock_go=ctx.use_2dblock_x,
                use_2dblock_x=ctx.use_2dblock_x,
                k_pack_go=not ctx.use_2dblock_x,
                k_pack_x=not ctx.use_2dblock_x,
                output_dtype=ctx.original_dtype,
            )
        else:
            grad_inputs = cg_grouped_gemm_backward_inputs(
                grad_output=grad_output_dq,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
            )
            grad_weights = cg_grouped_gemm_backward_weights(
                grad_output=grad_output_m_dq,
                inputs=x_dq,
                expert_indices=expert_indices,
                num_experts=num_experts,
            )
            if not ctx.trans_weights:
                grad_weights = grad_weights.transpose(-2, -1)

        if ctx.use_dge:
            scale_shape = list(expert_weights_mxfp4.shape)
            if ctx.use_2dblock_w:
                scale_shape[quant_axis_w] //= BLOCK_SIZE_DEFAULT // 2
                scale_shape[requant_axis_w] //= BLOCK_SIZE_DEFAULT
            else:
                scale_shape[requant_axis_w] //= BLOCK_SIZE_DEFAULT // 2

            # unpack fp4 weights
            scale_ones = expert_weights_mxfp4.new_ones(scale_shape, dtype=torch.uint8) * 127
            w_fp4_values = torch.ops.torchtitan.convert_from_mxfp4(
                expert_weights_mxfp4,
                scale_ones,
                ctx.original_dtype,
                axis=quant_axis_w if ctx.use_2dblock_w else requant_axis_w,
                is_2d_block=ctx.use_2dblock_w,
            )
            grad_weights *= dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)

        # Return grads with arity matching forward's positional inputs:
        #   (inputs, expert_weights, expert_indices, trans_weights,
        #    use_2dblock_x, use_2dblock_w, use_sr_grad, hadamard_transform,
        #    use_dge)
        return grad_inputs, grad_weights, None, None, None, None, None, None, None
