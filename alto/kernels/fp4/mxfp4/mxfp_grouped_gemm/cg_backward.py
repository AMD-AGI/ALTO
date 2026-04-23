# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch/torchtitan licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

import torch
import triton
from torch.library import triton_op, wrap_triton
import triton.language as tl

from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import (
    STANDARD_CONFIGS,
    ALIGN_SIZE_M,
)
from alto.kernels.fp4.mxfp4.mxfp_quantization import BLOCK_SIZE_DEFAULT

# ============ Triton kernel for contiguous grouped GEMM backward inputs ============


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N",
        "K",
        "GROUP_SIZE_M",
        "K_PACK_B",
        "USE_2DBLOCK_GO",
        "USE_2DBLOCK_B",
        "QUANT_BLOCK_SIZE",
    ],
)
@triton.jit
def _kernel_mxfp4_grouped_gemm_backward_dx(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    b_ptr,  # expert weights [num_experts, N, K]
    grad_input_ptr,  # [M_TOTAL, K]
    # Pointer to indices array
    indices_ptr,  # [M_TOTAL]
    go_s_ptr,
    b_s_ptr,
    stride_gom,
    stride_gon,
    stride_be,
    stride_bn,
    stride_bk,
    stride_gim,
    stride_gik,
    stride_gosm,
    stride_gosn,
    stride_bse,
    stride_bsn,
    stride_bsk,
    M_TOTAL,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    K_PACK_B: tl.constexpr,
    USE_2DBLOCK_GO: tl.constexpr,
    USE_2DBLOCK_B: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Computes the gradient with respect to the inputs (backward pass).
    Performs: grad_input = grad_output @ expert_weights
    """
    if USE_2DBLOCK_GO:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    N_PACKED: tl.constexpr = N // 2
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 2
    K_PACKED: tl.constexpr = K // 2

    # number of MXFP blocks inside a tile
    n_rep_n: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    # total number of scales
    Ns: tl.constexpr = (N // QUANT_BLOCK_SIZE)

    pid = tl.program_id(0)

    # number of tiles per matrix dimension
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    # 2D tile index from linear
    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K
    k_pack_start = tile_k * PACKED_BLOCK_SIZE_K

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start
        offs_k_pack = tl.arange(0, PACKED_BLOCK_SIZE_K) + k_pack_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K
        mask_k_pack = offs_k_pack < K_PACKED

        if USE_2DBLOCK_GO:
            offs_m_scale = offs_m // QUANT_BLOCK_SIZE
        else:
            offs_m_scale = offs_m
        if USE_2DBLOCK_B:
            offs_k_scale = offs_k // QUANT_BLOCK_SIZE
        else:
            offs_k_scale = offs_k

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Initialize accumulator for the gradient
        grad_input = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

        # Compute gradient with respect to inputs in tiles along N dimension
        for ni in range(num_n_tiles):
            # offsets and mask for N dimension
            offs_n = tl.arange(0, BLOCK_SIZE_N) + ni * BLOCK_SIZE_N
            mask_n = offs_n < N
            offs_n_pack = tl.arange(0, PACKED_BLOCK_SIZE_N) + ni * PACKED_BLOCK_SIZE_N
            mask_n_pack = offs_n_pack < N_PACKED

            offs_n_scale = ni * n_rep_n + tl.arange(0, n_rep_n)
            mask_n_scale = offs_n_scale < Ns

            # Masks for grad_output and weights
            mask_go = mask_m[:, None] & mask_n_pack[None, :]

            # Load grad_output with bounds checking
            go_ptrs = (grad_output_ptr + offs_m[:, None] * stride_gom + offs_n_pack[None, :] * stride_gon)
            go = tl.load(go_ptrs, mask=mask_go, other=0)

            # Load expert weights for the expert assigned to this block
            # For backward pass, we need W, not W^T, so dimensions are [N, K]
            # N is the actual reduction dim, so K_PACK_B means packed along N dim
            if K_PACK_B:
                w_ptrs = (b_ptr + expert_idx * stride_be + offs_n_pack[:, None] * stride_bn +
                          offs_k[None, :] * stride_bk)
                mask_w = mask_n_pack[:, None] & mask_k[None, :]
            else:
                w_ptrs = (b_ptr + expert_idx * stride_be + offs_n[:, None] * stride_bn +
                          offs_k_pack[None, :] * stride_bk)
                mask_w = mask_n[:, None] & mask_k_pack[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0)

            go_s_ptrs = (go_s_ptr + offs_m_scale[:, None] * stride_gosm + offs_n_scale[None, :] * stride_gosn)
            # B scales are K x N even though B operand is N x K.
            b_s_ptrs = (b_s_ptr + expert_idx * stride_bse + offs_k_scale[:, None] * stride_bsk +
                        offs_n_scale[None, :] * stride_bsn)
            go_s = tl.load(go_s_ptrs, mask=mask_m[:, None] & mask_n_scale[None, :], other=1)
            b_s = tl.load(b_s_ptrs, mask=mask_k[:, None] & mask_n_scale[None, :], other=1)
            grad_input += tl.dot_scaled(go,
                                        go_s,
                                        "e2m1",
                                        w,
                                        b_s,
                                        "e2m1",
                                        lhs_k_pack=True,
                                        rhs_k_pack=K_PACK_B,
                                        out_dtype=tl.float32)

        # Store results with bounds checking
        grad_input_ptrs = (grad_input_ptr + offs_m[:, None] * stride_gim + offs_k[None, :] * stride_gik)
        mask_gi = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_input_ptrs, grad_input, mask=mask_gi)


# ============ Triton kernel for contiguous grouped GEMM backward weights ============


# =============== Functions for backward pass =================
# ==== simpler approach =============
@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=[
        "N",
        "K",
        "NUM_EXPERTS",
        "GROUP_SIZE_M",
        "K_PACK_GO",
        "K_PACK_A",
        "USE_2DBLOCK_GO",
        "USE_2DBLOCK_A",
        "QUANT_BLOCK_SIZE",
    ],
)
@triton.jit
def _kernel_mxfp4_grouped_gemm_backward_dw(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    indices_ptr,  # [M_total]
    go_s_ptr,
    a_s_ptr,
    stride_gom,
    stride_gon,
    stride_am,
    stride_ak,
    stride_gbe,
    stride_gbn,
    stride_gbk,
    stride_gosm,
    stride_gosn,
    stride_asm,
    stride_ask,
    # Matrix dimensions
    M_TOTAL,  # Total M dimension
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Group parameters
    GROUP_SIZE_M: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    K_PACK_GO: tl.constexpr,
    K_PACK_A: tl.constexpr,
    USE_2DBLOCK_GO: tl.constexpr,
    USE_2DBLOCK_A: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """
    Significantly simplified kernel for weight gradient computation.
    This kernel processes one N-K tile across all groups that use the same expert.
    """
    if USE_2DBLOCK_GO:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    PACKED_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_M // 2
    PACKED_M: tl.constexpr = M_TOTAL // 2
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    PACKED_N: tl.constexpr = N // 2
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 2
    PACKED_K: tl.constexpr = K // 2
    # number of MXFP blocks inside a tile
    n_rep_m: tl.constexpr = BLOCK_SIZE_M // QUANT_BLOCK_SIZE
    # total number of scales
    Ms: tl.constexpr = (M_TOTAL // QUANT_BLOCK_SIZE)

    pid = tl.program_id(0)

    # Determine expert and position within the matrix
    expert_id = pid // ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))
    position_id = pid % ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))

    # Only process if expert is valid
    if expert_id < NUM_EXPERTS:
        # Calculate positions in N and K dimensions
        n_tiles = K // BLOCK_SIZE_K
        tile_n = position_id // n_tiles
        tile_k = position_id % n_tiles

        n_start = tile_n * BLOCK_SIZE_N
        n_start_pack = tile_n * PACKED_BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K
        k_start_pack = tile_k * PACKED_BLOCK_SIZE_K

        # Only process if in bounds
        if n_start < N and k_start < K:
            # Create offset arrays
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
            offs_n_pack = tl.arange(0, PACKED_BLOCK_SIZE_N) + n_start_pack
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start
            offs_k_pack = tl.arange(0, PACKED_BLOCK_SIZE_K) + k_start_pack

            # Create masks for bounds checking
            mask_n = offs_n < N
            mask_n_pack = offs_n_pack < PACKED_N
            mask_k = offs_k < K
            mask_k_pack = offs_k_pack < PACKED_K

            if USE_2DBLOCK_A:
                offs_k_scale = offs_k // QUANT_BLOCK_SIZE
            else:
                offs_k_scale = offs_k
            if USE_2DBLOCK_GO:
                offs_n_scale = offs_n // QUANT_BLOCK_SIZE
            else:
                offs_n_scale = offs_n

            # Initialize accumulator for the gradient
            grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

            # Go through all groups to find those using this expert
            for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
                group_start = group_idx * GROUP_SIZE_M

                # Get expert ID for this group
                group_expert = tl.load(indices_ptr + group_start)

                # Only process if this group uses the current expert
                if group_expert == expert_id:
                    # Process the group in blocks
                    for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                        # Global offsets for group's data
                        m_start = group_start + m_offset
                        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
                        offs_m_pack = tl.arange(0, PACKED_BLOCK_SIZE_M) + m_start // 2

                        group_length = group_start + GROUP_SIZE_M

                        # Create mask for M dimension
                        mask_m = offs_m < tl.minimum(group_length, M_TOTAL)
                        mask_m_pack = offs_m_pack < tl.minimum(group_length // 2, PACKED_M)

                        offs_m_scale = m_start // QUANT_BLOCK_SIZE + tl.arange(0, n_rep_m)
                        mask_m_scale = offs_m_scale < tl.minimum(group_length // QUANT_BLOCK_SIZE, Ms)

                        # Load grad_output [M, N].T
                        if K_PACK_GO:
                            go_ptrs = (grad_output_ptr + offs_m_pack[None, :] * stride_gom +
                                       offs_n[:, None] * stride_gon)
                            mask_go = mask_m_pack[None, :] & mask_n[:, None]
                        else:
                            go_ptrs = (grad_output_ptr + offs_m[None, :] * stride_gom +
                                       offs_n_pack[:, None] * stride_gon)
                            mask_go = mask_m[None, :] & mask_n_pack[:, None]
                        go = tl.load(go_ptrs, mask=mask_go, other=0)

                        # Load inputs [M, K]
                        if K_PACK_A:
                            in_ptrs = (inputs_ptr + offs_m_pack[:, None] * stride_am + offs_k[None, :] * stride_ak)
                            mask_in = mask_m_pack[:, None] & mask_k[None, :]
                        else:
                            in_ptrs = (inputs_ptr + offs_m[:, None] * stride_am + offs_k_pack[None, :] * stride_ak)
                            mask_in = mask_m[:, None] & mask_k_pack[None, :]
                        inp = tl.load(in_ptrs, mask=mask_in, other=0)

                        go_s_ptrs = go_s_ptr + offs_n_scale[:, None] * stride_gosn + offs_m_scale[None, :] * stride_gosm
                        # input scales are K x M even though the input operand is M x K.
                        inp_s_ptrs = a_s_ptr + offs_k_scale[:, None] * stride_ask + offs_m_scale[None, :] * stride_asm

                        go_s = tl.load(go_s_ptrs, mask=mask_n[:, None] & mask_m_scale[None, :], other=1)
                        inp_s = tl.load(inp_s_ptrs, mask=mask_k[:, None] & mask_m_scale[None, :], other=1)

                        grad_weights += tl.dot_scaled(go,
                                                      go_s,
                                                      "e2m1",
                                                      inp,
                                                      inp_s,
                                                      "e2m1",
                                                      lhs_k_pack=K_PACK_GO,
                                                      rhs_k_pack=K_PACK_A,
                                                      out_dtype=tl.float32)

            # Store results to the appropriate part of the expert's weight gradients
            grad_w_ptrs = (grad_weights_ptr + expert_id * stride_gbe + offs_n[:, None] * stride_gbn +
                           offs_k[None, :] * stride_gbk)
            mask_gw = mask_n[:, None] & mask_k[None, :]
            tl.store(grad_w_ptrs, grad_weights, mask=mask_gw)


@triton_op("torchtitan::mxfp4_grouped_gemm_backward_weights", mutates_args={})
def mxfp4_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    inputs: torch.Tensor,  # [M_bufferlen, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    go_scales: torch.Tensor,
    input_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_go: bool = False,
    use_2dblock_x: bool = True,
    k_pack_go: bool = True,
    k_pack_x: bool = True,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Simple version of backward pass for weights using a single kernel launch.

    Args:
        grad_output: Gradient from output, shape [M_bufferlen, N]
        inputs: Input tensor, shape [M_bufferlen, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_bufferlen_go, N = grad_output.shape
    if k_pack_go:
        M_bufferlen_go *= 2
    else:
        N *= 2
    M_bufferlen, K = inputs.shape
    if k_pack_x:
        M_bufferlen *= 2
    else:
        K *= 2
    assert M_bufferlen_go == M_bufferlen, f"Output gradient and inputs have different size in M dim: {M_bufferlen_go} vs. {M_bufferlen}"
    M_total = expert_indices.shape[0]
    torch._check(M_total > 0)
    torch._check(M_total % ALIGN_SIZE_M == 0)

    # Check if dimensions match
    assert (M_total % ALIGN_SIZE_M == 0), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"

    # Ensure expert_indices has the right dtype
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Create output tensor for gradients
    if trans_weights:
        grad_weights = torch.zeros((num_experts, N, K), device=grad_output.device, dtype=output_dtype)
        stride_gbe, stride_gbn, stride_gbk = grad_weights.stride()
    else:
        grad_weights = torch.zeros((num_experts, K, N), device=grad_output.device, dtype=output_dtype)
        stride_gbe, stride_gbk, stride_gbn = grad_weights.stride()
    stride_gom, stride_gon = grad_output.stride()
    stride_gosm, stride_gosn = go_scales.stride()
    stride_am, stride_ak = inputs.stride()
    stride_asm, stride_ask = input_scales.stride()

    # Calculate grid size for the kernel
    # Each thread block handles one expert's N-K tile
    grid = lambda meta: (num_experts * triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
    # Launch kernel
    wrap_triton(_kernel_mxfp4_grouped_gemm_backward_dw)[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        go_scales,
        input_scales,
        stride_gom,
        stride_gon,
        stride_am,
        stride_ak,
        stride_gbe,
        stride_gbn,
        stride_gbk,
        stride_gosm,
        stride_gosn,
        stride_asm,
        stride_ask,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        K_PACK_GO=k_pack_go,
        K_PACK_A=k_pack_x,
        USE_2DBLOCK_GO=use_2dblock_go,
        USE_2DBLOCK_A=use_2dblock_x,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
    )

    return grad_weights


@triton_op("torchtitan::mxfp4_grouped_gemm_backward_inputs", mutates_args={})
def mxfp4_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,  # [M_bufferlen, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K] if trans_weights else [num_experts, K, N]
    expert_indices: torch.Tensor,  # [M_total]
    go_scales: torch.Tensor,
    expert_weight_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    k_pack_w: bool = True,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to inputs.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        expert_weights: Expert weight tensor, shape [num_experts, N, K] if trans_weights else [num_experts, K, N]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]

    Returns:
        grad_inputs: Gradient with respect to inputs, shape [M_total, K]
    """
    # Validate inputs
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_bufferlen, N = grad_output.shape
    # assume packed along N dim
    N *= 2

    if trans_weights:
        num_experts, Nw, K = expert_weights.shape
        stride_be, stride_bn, stride_bk = expert_weights.stride()
        stride_bse, stride_bsn, stride_bsk = expert_weight_scales.stride()
    else:
        num_experts, K, Nw = expert_weights.shape
        stride_be, stride_bk, stride_bn = expert_weights.stride()
        stride_bse, stride_bsk, stride_bsn = expert_weight_scales.stride()
    # N is the reduction dim so k_pack_w actually means packed along the N dim
    if k_pack_w:
        Nw *= 2
    else:
        K *= 2
    assert N == Nw, f"Output gradient and expert weights have different size in N dim: {N} vs. {Nw}"

    M_total = expert_indices.shape[0]
    torch._check(M_total > 0)
    torch._check(M_total % ALIGN_SIZE_M == 0)

    # Check if dimensions match
    assert (M_total % ALIGN_SIZE_M == 0), f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"

    # Create output tensor for gradients
    grad_inputs = torch.zeros((M_bufferlen, K), device=grad_output.device, dtype=output_dtype)
    stride_gom, stride_gon = grad_output.stride()
    stride_gosm, stride_gosn = go_scales.stride()
    stride_gim, stride_gik = grad_inputs.stride()

    # Calculate grid size for the kernel
    grid = lambda meta: (triton.cdiv(M_total, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)

    # Launch kernel
    wrap_triton(_kernel_mxfp4_grouped_gemm_backward_dx)[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        go_scales,
        expert_weight_scales,
        stride_gom,
        stride_gon,
        stride_be,
        stride_bn,
        stride_bk,
        stride_gim,
        stride_gik,
        stride_gosm,
        stride_gosn,
        stride_bse,
        stride_bsn,
        stride_bsk,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        K_PACK_B=k_pack_w,
        USE_2DBLOCK_GO=use_2dblock_x,
        USE_2DBLOCK_B=use_2dblock_w,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
    )
    return grad_inputs

