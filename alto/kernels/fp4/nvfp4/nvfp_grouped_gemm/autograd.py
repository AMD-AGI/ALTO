# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 Grouped GEMM autograd function.

Implements the full forward+backward for MoE-style grouped matrix multiplication
using the NVFP4 QDQ path. The structure mirrors NVFP4LinearFunction: keep
axis-specific quantized views so the implementation models what a future fully
native NVFP4 grouped-GEMM backend would consume.

Two logical primitives are used throughout:

* ``_nvfp4_grouped_mm_2d_3d``  : A[M, X]  × B[E, X, Y] -> C[M, Y]
* ``_nvfp4_grouped_wgrad``     : G.T[N,M] × X[M, K]    -> dW[E, N, K]

Execution backends:

* CDNA4 / gfx950: native Triton grouped backend (via grouped BF16 Triton GEMM
  kernels fed from packed-NVFP4 QDQ views)
* non-CDNA4: Python loop fallback
"""

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_forward import cg_grouped_gemm_forward
from alto.kernels.blockwise_fp8.grouped_gemm.cg_backward import (
    cg_grouped_gemm_backward_inputs,
    cg_grouped_gemm_backward_weights,
)
from alto.kernels.dge import dge_bwd
from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync
from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.fp4.nvfp4.nvfp_linear import _qdq
from alto.kernels.fp4.nvfp4.nvfp_quantization import convert_from_nvfp4, is_cdna4
from alto.kernels.hadamard_transform import HadamardTransform

ALIGN_SIZE_M: int = 128
QDQ_AXIS0_BLOCK_SIZE: int = 16

_GROUPED_MM_AVAILABLE: Optional[bool] = None


def _has_native_grouped_mm() -> bool:
    """True when the native Triton grouped backend is available."""
    global _GROUPED_MM_AVAILABLE
    if _GROUPED_MM_AVAILABLE is not None:
        return _GROUPED_MM_AVAILABLE
    _GROUPED_MM_AVAILABLE = bool(torch.cuda.is_available() and is_cdna4())
    return _GROUPED_MM_AVAILABLE


def _check_grouped_loop_contract(M_total: int, *, where: str) -> None:
    torch._check(
        M_total > 0 and M_total % ALIGN_SIZE_M == 0,
        lambda: (
            f"{where} (loop backend) requires M_total ({M_total}) to be a "
            f"positive multiple of ALIGN_SIZE_M ({ALIGN_SIZE_M})."
        ),
    )


def _check_grouped_axis0_qdq_contract(M_total: int, *, where: str) -> None:
    torch._check(
        M_total > 0 and M_total % QDQ_AXIS0_BLOCK_SIZE == 0,
        lambda: (
            f"{where} requires M_total ({M_total}) to be a positive multiple "
            f"of QDQ_AXIS0_BLOCK_SIZE ({QDQ_AXIS0_BLOCK_SIZE}) so the axis=0 "
            "NVFP4 QDQ view is well-defined."
        ),
    )


def _resolve_expert_indices(
    *,
    expert_indices: Optional[torch.Tensor],
    offs: Optional[torch.Tensor],
    M_total: int,
) -> torch.Tensor:
    """Normalize routing information to a contiguous int32 expert-index tensor."""
    if expert_indices is None:
        torch._check(offs is not None, lambda: "Either expert_indices or offs must be provided.")
        expert_indices = create_indices_from_offsets_nosync(offs).contiguous()
    elif expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)
    if not expert_indices.is_contiguous():
        expert_indices = expert_indices.contiguous()
    torch._check(
        expert_indices.shape[0] == M_total,
        lambda: (
            f"expert_indices.shape[0] ({expert_indices.shape[0]}) must match "
            f"M_total ({M_total})."
        ),
    )
    return expert_indices


def _group_ids_from_expert_indices(expert_indices: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Return one expert id per ALIGN_SIZE_M-sized group without host sync."""
    return expert_indices.view(num_groups, ALIGN_SIZE_M)[:, 0].contiguous()


def _nvfp4_grouped_mm_2d_3d(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    expert_indices: torch.Tensor,
    num_groups: Optional[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped GEMM for 2D activations × 3D expert weights."""
    if _has_native_grouped_mm():
        return cg_grouped_gemm_forward(
            A.contiguous(),
            B.transpose(-2, -1).contiguous(),
            expert_indices.contiguous(),
        ).to(output_dtype)

    assert num_groups is not None, "_nvfp4_grouped_mm_2d_3d: loop fallback requires num_groups"
    M_total = A.shape[0]
    _check_grouped_loop_contract(M_total, where="_nvfp4_grouped_mm_2d_3d")
    X = A.shape[1]
    Y = B.shape[2]
    group_ids = _group_ids_from_expert_indices(expert_indices, num_groups)
    A_groups = A.view(num_groups, ALIGN_SIZE_M, X)
    y_groups = torch.zeros((num_groups, ALIGN_SIZE_M, Y), dtype=output_dtype, device=A.device)
    for eid in range(B.shape[0]):
        mask = group_ids == eid
        y_groups[mask] = A_groups[mask] @ B[eid]
    return y_groups.reshape(M_total, Y)


def _nvfp4_grouped_wgrad(
    grad_output: torch.Tensor,
    x_bwd: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    num_groups: int,
    *,
    use_sr_grad: bool,
    use_per_tensor_scale: bool,
    use_2dblock_x: bool,
    trans_weights: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize grad_output and compute grouped dW."""
    if use_2dblock_x:
        g_m_dq = _qdq(
            grad_output, axis=-1, is_2d_block=True,
            use_per_tensor_scale=use_per_tensor_scale, use_sr=use_sr_grad,
        )
    else:
        g_m_dq = _qdq(
            grad_output, axis=0, is_2d_block=False,
            use_per_tensor_scale=use_per_tensor_scale, use_sr=use_sr_grad,
        )

    if _has_native_grouped_mm():
        dw = cg_grouped_gemm_backward_weights(
            g_m_dq.contiguous(),
            x_bwd.contiguous(),
            expert_indices.contiguous(),
            num_experts=num_experts,
        ).to(output_dtype)
        if not trans_weights:
            dw = dw.transpose(-2, -1).contiguous()
        return dw

    N = g_m_dq.shape[1]
    K = x_bwd.shape[1]
    _check_grouped_loop_contract(grad_output.shape[0], where="_nvfp4_grouped_wgrad")
    group_ids = _group_ids_from_expert_indices(expert_indices, num_groups)
    go_groups = g_m_dq.view(num_groups, ALIGN_SIZE_M, N)
    x_groups = x_bwd.view(num_groups, ALIGN_SIZE_M, K)
    dw = torch.zeros((num_experts, N, K), dtype=output_dtype, device=g_m_dq.device)
    for eid in range(num_experts):
        mask = group_ids == eid
        go_e = go_groups[mask].reshape(-1, N)
        x_e = x_groups[mask].reshape(-1, K)
        dw[eid] = go_e.T @ x_e
    if not trans_weights:
        dw = dw.transpose(-2, -1).contiguous()
    return dw


@torch.compiler.allow_in_graph
class NVFP4GroupedGEMMFunction(torch.autograd.Function):
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
        _check_grouped_axis0_qdq_contract(M_total, where="NVFP4GroupedGEMMFunction.forward")
        num_groups = None
        if not _has_native_grouped_mm():
            _check_grouped_loop_contract(M_total, where="NVFP4GroupedGEMMFunction.forward")
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


# Backward-compatible aliases. Both public entry modes route through the same
# autograd implementation.
NVFP4GroupedGEMM = NVFP4GroupedGEMMFunction
NVFP4GroupedGEMMNative = NVFP4GroupedGEMMFunction


def _offs_to_indices(offs: torch.Tensor, M_total: int) -> torch.Tensor:
    """Convert cumulative offsets [E] to per-token expert indices [M_total]."""
    return _resolve_expert_indices(expert_indices=None, offs=offs, M_total=M_total)
