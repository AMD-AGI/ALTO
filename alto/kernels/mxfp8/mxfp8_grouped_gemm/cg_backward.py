# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""MXFP8 contiguous grouped GEMM — backward (dgrad + wgrad) + autograd Function.

Skeleton only; kernel bodies filled in Steps 3-5.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT, is_cdna4
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
)
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_forward import mxfp8_grouped_gemm_forward


# ============ dgrad: grad_input = grad_output @ expert_weights ============

@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N", "K", "GROUP_SIZE_M",
        "USE_2DBLOCK_GO", "USE_2DBLOCK_B",
        "QUANT_BLOCK_SIZE",
        "LHS_FORMAT_ID", "RHS_FORMAT_ID", "USE_DOT_SCALED",
    ],
)
@triton.jit
def _kernel_mxfp8_grouped_gemm_backward_dx(
    grad_output_ptr,  # [M_TOTAL, N], fp8 (V1: e4m3; v2: e5m2)
    b_ptr,            # [num_experts, N, K], fp8 (e4m3)
    grad_input_ptr,   # [M_TOTAL, K], output dtype
    indices_ptr,
    go_s_ptr,
    b_s_ptr,
    stride_gom, stride_gon,
    stride_be, stride_bn, stride_bk,
    stride_gim, stride_gik,
    stride_gosm, stride_gosn,
    stride_bse, stride_bsn, stride_bsk,
    M_TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_2DBLOCK_GO: tl.constexpr,
    USE_2DBLOCK_B: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    LHS_FORMAT_ID: tl.constexpr,  # V1: 0 (e4m3); v2: 1 (e5m2) for grad_output
    RHS_FORMAT_ID: tl.constexpr,  # 0 (e4m3)
    USE_DOT_SCALED: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = ALIGN_SIZE_M,
):
    """dgrad kernel: dX = GO @ W (N is reduction dim). To be implemented in Step 3."""
    # TODO(Step 3): port from mxfp4 cg_backward._kernel_mxfp4_grouped_gemm_backward_dx
    pass


# ============ wgrad: grad_weights = grad_output^T @ inputs (per expert) ============

@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N", "K", "NUM_EXPERTS", "GROUP_SIZE_M",
        "USE_2DBLOCK_GO", "USE_2DBLOCK_A",
        "QUANT_BLOCK_SIZE",
        "LHS_FORMAT_ID", "RHS_FORMAT_ID", "USE_DOT_SCALED",
    ],
)
@triton.jit
def _kernel_mxfp8_grouped_gemm_backward_dw(
    grad_output_ptr,   # [M_TOTAL, N], fp8 (V1: e4m3; v2: e5m2), quantized along M-axis
    inputs_ptr,        # [M_TOTAL, K], fp8 (e4m3), quantized along M-axis
    grad_weights_ptr,  # [num_experts, N, K], output dtype
    indices_ptr,
    go_s_ptr,
    a_s_ptr,
    stride_gom, stride_gon,
    stride_am, stride_ak,
    stride_gbe, stride_gbn, stride_gbk,
    stride_gosm, stride_gosn,
    stride_asm, stride_ask,
    M_TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    USE_2DBLOCK_GO: tl.constexpr,
    USE_2DBLOCK_A: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    LHS_FORMAT_ID: tl.constexpr,  # V1: 0 (e4m3); v2: 1 (e5m2) for grad_output
    RHS_FORMAT_ID: tl.constexpr,  # 0 (e4m3)
    USE_DOT_SCALED: tl.constexpr,
):
    """wgrad kernel: dW = GO^T @ X (M is reduction dim). To be implemented in Step 4."""
    # TODO(Step 4): port from mxfp4 cg_backward._kernel_mxfp4_grouped_gemm_backward_dw
    pass


# =============== triton_op wrappers ===============

@triton_op("alto::mxfp8_grouped_gemm_backward_inputs", mutates_args={})
def mxfp8_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    go_scales: torch.Tensor,
    expert_weight_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    lhs_format_id: int = 0,  # V1: e4m3 (v2: 1=e5m2 for grad_output)
    rhs_format_id: int = 0,  # e4m3
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """dgrad wrapper. To be implemented in Step 3."""
    raise NotImplementedError("Step 3: implement dgrad wrapper")


@triton_op("alto::mxfp8_grouped_gemm_backward_weights", mutates_args={})
def mxfp8_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    go_scales: torch.Tensor,
    input_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_go: bool = False,
    use_2dblock_x: bool = False,
    lhs_format_id: int = 0,  # V1: e4m3 (v2: 1=e5m2 for grad_output)
    rhs_format_id: int = 0,  # e4m3
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """wgrad wrapper. To be implemented in Step 4."""
    raise NotImplementedError("Step 4: implement wgrad wrapper")


# =============== Autograd Function ===============

@torch.compiler.allow_in_graph
class MXFP8GroupedGEMM(torch.autograd.Function):
    """Autograd Function for mxfp8 grouped GEMM. To be implemented in Step 5."""

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
        fwd_format="e4m3",
        bwd_grad_format="e4m3",
    ):
        raise NotImplementedError("Step 5: implement autograd forward")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Step 5: implement autograd backward")
