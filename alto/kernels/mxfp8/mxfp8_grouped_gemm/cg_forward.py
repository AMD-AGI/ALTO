# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""MXFP8 contiguous grouped GEMM — forward pass.

Skeleton only; kernel body filled in Step 2.
"""

import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N",
        "K",
        "GROUP_SIZE_M",
        "USE_2DBLOCK_A",
        "USE_2DBLOCK_B",
        "QUANT_BLOCK_SIZE",
        "LHS_FORMAT_ID",
        "RHS_FORMAT_ID",
        "USE_DOT_SCALED",
    ],
)
@triton.jit
def _kernel_mxfp8_grouped_gemm_forward(
    # Pointers to matrices
    a_ptr,        # [M_total, K], fp8 (e4m3)
    b_ptr,        # [num_experts, N, K], fp8 (e4m3)
    c_ptr,        # [M_total, N], output dtype
    indices_ptr,  # [M_total], int32
    a_s_ptr,      # A scales, uint8 E8M0
    b_s_ptr,      # B scales, uint8 E8M0
    # Matrix strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bse, stride_bsn, stride_bsk,
    # Matrix dimensions
    M_TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    # Tile params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_2DBLOCK_A: tl.constexpr,
    USE_2DBLOCK_B: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    LHS_FORMAT_ID: tl.constexpr,  # 0=e4m3, 1=e5m2
    RHS_FORMAT_ID: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = ALIGN_SIZE_M,
    SUPER_GROUP_M: tl.constexpr = 32,
):
    """Forward grouped GEMM kernel: Y = X @ W^T per expert. To be implemented in Step 2."""
    # TODO(Step 2): port from mxfp4 cg_forward._kernel_mxfp4_grouped_gemm_forward,
    # remove all packing, parameterize dot_scaled dtype strings via LHS/RHS_FORMAT_ID,
    # add CDNA3 fallback via _dequantize_fp8 + tl.dot when USE_DOT_SCALED=False.
    pass


@triton_op("alto::mxfp8_grouped_gemm_forward", mutates_args={})
def mxfp8_grouped_gemm_forward(
    inputs: torch.Tensor,            # [M_total, K], fp8
    expert_weights: torch.Tensor,    # [num_experts, N, K], fp8
    expert_indices: torch.Tensor,    # [M_total], int32
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    lhs_format_id: int = 0,          # 0=e4m3, 1=e5m2
    rhs_format_id: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Wrapper: launches forward kernel. To be implemented in Step 2."""
    raise NotImplementedError("Step 2: implement forward wrapper")
