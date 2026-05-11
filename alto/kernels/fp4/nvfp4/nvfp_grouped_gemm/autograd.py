# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified autograd core for NVFP4 Grouped GEMM.

Internal weight-layout contract
-------------------------------

``NVFP4GroupedGEMM.forward`` always receives ``expert_weights`` in the
canonical ``[E, N, K]`` layout (matching ``nn.Linear.weight``-style storage:
output dim first, reduction dim last).  This layout is identical to what the
CDNA4 grouped kernels (``cg_grouped_gemm_forward`` /
``cg_grouped_gemm_backward_inputs`` / ``cg_grouped_gemm_backward_weights``)
consume natively, so no per-step transpose / contiguous copy is needed
anywhere inside the autograd function or its primitives.

The public wrappers in :mod:`functional` are the single place that adapts
``trans_weights=False`` (dispatch-layer ``[E, K, N]``) and
``trans_weights=True`` (user ``[E, N, K]``) into this canonical layout.
"""

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.dge import dge_bwd
from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.fp4.nvfp4.nvfp_quantization import _qdq, convert_from_nvfp4
from alto.kernels.hadamard_transform import HadamardTransform
from .autotune import ALIGN_SIZE_M
from .cg_backward import _nvfp4_grouped_dgrad, _nvfp4_grouped_wgrad
from .cg_forward import _nvfp4_grouped_fprop
from alto.kernels.fp4.fp4_common import (
    check_grouped_loop_contract,
    resolve_expert_indices,
    use_cdna4_grouped_backend,
)
from .utils import check_grouped_axis0_qdq_contract


# Canonical reduction-axis indices for expert_weights in ``[E, N, K]``.
#   fprop GEMM reduces over K (last axis) → quant on axis -1
#   dgrad GEMM reduces over N (second-to-last axis) → quant on axis -2
_FPROP_AXIS_W = -1
_DGRAD_AXIS_W = -2


@torch.compiler.allow_in_graph
class NVFP4GroupedGEMM(torch.autograd.Function):
    """Unified NVFP4 grouped GEMM autograd implementation.

    ``expert_weights`` must arrive in the canonical ``[E, N, K]`` layout; the
    public wrappers in :mod:`functional` handle any upstream transpose.

    Backend selection:
      * CDNA4 -> Triton grouped kernels (fprop, dgrad, wgrad)
      * otherwise -> Python loop fallback
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: Optional[torch.Tensor],
        offs: Optional[torch.Tensor],
        use_2dblock_x: bool,
        use_2dblock_w: bool,
        use_sr_grad: bool,
        use_per_tensor_scale: bool,
        hadamard_transform: Optional[HadamardTransform] = None,
        use_dge: bool = False,
    ) -> torch.Tensor:
        M_bufferlen = inputs.shape[0]
        original_dtype = inputs.dtype
        expert_weights = unwrap_weight_wrapper(expert_weights)
        # Align weight dtype with activation so saved QDQ tensors share a
        # common dtype across forward / backward under AMP autocast.
        if expert_weights.dtype != original_dtype:
            expert_weights = expert_weights.to(dtype=original_dtype)
        expert_indices = resolve_expert_indices(
            expert_indices=expert_indices, offs=offs, M_total=M_bufferlen,
        )
        # M_total is the routed-token count (<= M_bufferlen when MoE wrappers
        # pad the activation buffer); kernels output [M_bufferlen, N] with
        # zeros in padding rows.
        M_total = expert_indices.shape[0]
        num_experts = expert_weights.shape[0]
        # axis=0 QDQ runs on the full buffer, so its alignment check uses
        # M_bufferlen; the loop fallback iterates only over routed rows.
        check_grouped_axis0_qdq_contract(M_bufferlen, location="NVFP4GroupedGEMM.forward")
        num_groups: Optional[int] = None
        if not use_cdna4_grouped_backend():
            check_grouped_loop_contract(M_total, where="NVFP4GroupedGEMM.forward")
            num_groups = M_total // ALIGN_SIZE_M

        x_dq = _qdq(
            inputs, axis=-1, is_2d_block=use_2dblock_x,
            use_per_tensor_scale=use_per_tensor_scale,
        )
        w_dq = _qdq(
            expert_weights, axis=_FPROP_AXIS_W, is_2d_block=use_2dblock_w,
            use_per_tensor_scale=use_per_tensor_scale,
        )
        y = _nvfp4_grouped_fprop(
            x_dq,
            w_dq,
            expert_indices=expert_indices,
            num_groups=num_groups,
            output_dtype=original_dtype,
        )

        w_raw_fp4 = None
        w_raw_scales = None
        if not use_2dblock_w:
            w_bwd = _qdq(
                expert_weights,
                axis=_DGRAD_AXIS_W,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
                return_raw=use_dge,
            )
            if use_dge:
                w_bwd, w_raw_fp4, w_raw_scales = w_bwd
        else:
            # 2D block scaling is axis-invariant, so the fprop view also plays
            # the role of the wgrad-axis view.
            w_bwd = w_dq
            if use_dge:
                _, w_raw_fp4, w_raw_scales = _qdq(
                    expert_weights,
                    axis=_FPROP_AXIS_W,
                    is_2d_block=True,
                    use_per_tensor_scale=use_per_tensor_scale,
                    return_raw=True,
                )

        if not use_2dblock_x:
            x_for_wgrad = (
                hadamard_transform(inputs, left_mul=True)
                if hadamard_transform is not None else inputs
            )
            x_bwd = _qdq(
                x_for_wgrad, axis=0, is_2d_block=False,
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

        # Saved QDQ tensors carry forward's dtype; align them with
        # grad_output to avoid BF16 vs FP32 mismatches under AMP autocast.
        if x_bwd.dtype != grad_output.dtype:
            x_bwd = x_bwd.to(dtype=grad_output.dtype)
        if w_bwd.dtype != grad_output.dtype:
            w_bwd = w_bwd.to(dtype=grad_output.dtype)

        num_groups = ctx.num_groups
        if num_groups is None:
            # Loop fallback was not needed during forward (CDNA4 path) but
            # backward's loop fallback still needs it; recompute from the
            # already-normalized expert_indices to avoid any host sync.
            num_groups = expert_indices.shape[0] // ALIGN_SIZE_M

        g_dq = _qdq(
            grad_output, axis=-1, is_2d_block=ctx.use_2dblock_x,
            use_per_tensor_scale=ctx.use_per_tensor_scale, use_sr=ctx.use_sr_grad,
        )
        grad_inputs = _nvfp4_grouped_dgrad(
            g_dq,
            w_bwd,
            expert_indices=expert_indices,
            num_groups=num_groups,
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
            num_groups=num_groups,
            use_sr_grad=ctx.use_sr_grad,
            use_per_tensor_scale=ctx.use_per_tensor_scale,
            use_2dblock_x=ctx.use_2dblock_x,
            output_dtype=ctx.original_dtype,
        )

        if ctx.use_dge:
            # Recover the exact FP4 bin each weight fell into without paying a
            # second quantization pass.
            dge_axis_w = _FPROP_AXIS_W if ctx.use_2dblock_w else _DGRAD_AXIS_W
            w_fp4_values = convert_from_nvfp4(
                w_raw_fp4,
                torch.ones_like(w_raw_scales),
                output_dtype=ctx.original_dtype,
                axis=dge_axis_w,
                is_2d_block=ctx.use_2dblock_w,
                per_tensor_scale=None,
            )
            grad_weights = grad_weights * dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)

        # Return grads with arity matching forward's positional inputs:
        #   (inputs, expert_weights, expert_indices, offs,
        #    use_2dblock_x, use_2dblock_w, use_sr_grad, use_per_tensor_scale,
        #    hadamard_transform, use_dge)
        return grad_inputs, grad_weights, None, None, None, None, None, None, None, None
