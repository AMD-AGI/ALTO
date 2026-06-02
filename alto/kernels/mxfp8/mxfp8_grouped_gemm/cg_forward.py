# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""MXFP8 contiguous grouped GEMM — forward pass."""

from typing import Optional

import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

from alto.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    _dequantize_fp8,
    is_cdna4,
)
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
    """Forward grouped GEMM kernel: Y = X @ W^T per expert.

    A = X [M_TOTAL, K] (e4m3), B = W [num_experts, N, K] (e4m3); reduction over K.
    IMPORTANT: assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M (or vice versa)
    and inputs are pre-aligned to block boundaries.
    """
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)

    c_type = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    # number of MXFP scale groups inside a K tile
    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    # total number of scales along K
    Ks: tl.constexpr = K // QUANT_BLOCK_SIZE

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):

        tile_m_idx, tile_n_idx = _compute_pid(tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M)

        m_start = tile_m_idx * BLOCK_SIZE_M
        n_start = tile_n_idx * BLOCK_SIZE_N

        if m_start < M_TOTAL:

            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

            if USE_2DBLOCK_A:
                offs_m_scale = offs_m // QUANT_BLOCK_SIZE
            else:
                offs_m_scale = offs_m
            if USE_2DBLOCK_B:
                offs_n_scale = offs_n // QUANT_BLOCK_SIZE
            else:
                offs_n_scale = offs_n

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):

                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                offs_k_scale = ki * n_rep_k + tl.arange(0, n_rep_k)
                mask_k_scale = offs_k_scale < Ks

                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k = offs_k < K

                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_k[:, None] & mask_n[None, :]

                # Determine the expert group index and load expert ID
                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                # Load inputs A [BLOCK_M, BLOCK_K]
                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                # Load expert weights B [BLOCK_K, BLOCK_N] for this block's expert
                b_ptrs = (b_ptr + expert_idx * stride_be + offs_n[None, :] * stride_bn +
                          offs_k[:, None] * stride_bk)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)

                a_s_ptrs = a_s_ptr + offs_m_scale[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
                # B scales are N x K even though B operand is K x N.
                b_s_ptrs = (b_s_ptr + expert_idx * stride_bse + offs_n_scale[:, None] * stride_bsn +
                            offs_k_scale[None, :] * stride_bsk)

                a_s = tl.load(a_s_ptrs, mask=mask_m[:, None] & mask_k_scale[None, :], other=1)
                b_s = tl.load(b_s_ptrs, mask=mask_n[:, None] & mask_k_scale[None, :], other=1)

                if USE_DOT_SCALED:
                    # CDNA4 path: native scaled dot. dtype strings chosen by format id.
                    if LHS_FORMAT_ID == 0:
                        if RHS_FORMAT_ID == 0:
                            accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3",
                                                        acc=accumulator, out_dtype=tl.float32)
                        else:
                            accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e5m2",
                                                        acc=accumulator, out_dtype=tl.float32)
                    else:
                        if RHS_FORMAT_ID == 0:
                            accumulator = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e4m3",
                                                        acc=accumulator, out_dtype=tl.float32)
                        else:
                            accumulator = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2",
                                                        acc=accumulator, out_dtype=tl.float32)
                else:
                    # CDNA3 fallback: dequantize to fp32 then plain dot.
                    # Scale offsets above already expand to per-row scales of shape
                    # [BLOCK, n_rep_k], so dequant treats them as 1D (IS_2D_BLOCK=False)
                    # regardless of how the scales were originally produced.
                    a_dq = _dequantize_fp8(
                        a, a_s,
                        output_dtype=tl.float32,
                        BLOCK_M=BLOCK_SIZE_M,
                        BLOCK_N=BLOCK_SIZE_K,
                        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                        FP8_FORMAT=LHS_FORMAT_ID,
                        IS_2D_BLOCK=False,
                        USE_ASM=False,
                    )
                    b_dq = tl.trans(_dequantize_fp8(
                        tl.trans(b), b_s,
                        output_dtype=tl.float32,
                        BLOCK_M=BLOCK_SIZE_N,
                        BLOCK_N=BLOCK_SIZE_K,
                        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                        FP8_FORMAT=RHS_FORMAT_ID,
                        IS_2D_BLOCK=False,
                        USE_ASM=False,
                    ))
                    accumulator = tl.dot(a_dq, b_dq, acc=accumulator, out_dtype=tl.float32)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M)

            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N
            mask_c = mask_m[:, None] & mask_n[None, :]

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, accumulator.to(c_type), mask=mask_c)


@triton_op("alto::mxfp8_grouped_gemm_forward", mutates_args={})
def mxfp8_grouped_gemm_forward(
    inputs: torch.Tensor,            # [M_total, K], fp8
    expert_weights: torch.Tensor,    # [num_experts, N, K] (trans) or [num_experts, K, N], fp8
    expert_indices: torch.Tensor,    # [M_total], int32
    input_scales: torch.Tensor,
    weight_scales: torch.Tensor,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    lhs_format_id: int = 0,          # 0=e4m3, 1=e5m2
    rhs_format_id: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    use_dot_scaled: Optional[bool] = None,
) -> torch.Tensor:
    """Launch the mxfp8 forward grouped GEMM kernel on pre-quantized operands.

    Y[m, :] = X[m] @ W[expert(m)]^T, computed per contiguous group of tokens.
    """
    assert inputs.dim() == 2, f"inputs must be 2D, got {inputs.dim()}D"
    assert expert_weights.dim() == 3, f"expert_weights must be 3D, got {expert_weights.dim()}D"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"
    if use_dot_scaled is None:
        use_dot_scaled = is_cdna4()

    M_total, K = inputs.shape
    torch._check(M_total > 0)
    assert M_total % ALIGN_SIZE_M == 0, \
        f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"
    assert expert_indices.numel() == M_total, \
        f"expert_indices length ({expert_indices.numel()}) must match M_total ({M_total})"
    if K % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"K ({K}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT})")

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    if trans_weights:
        num_experts, N, K_weights = expert_weights.shape
        stride_be, stride_bn, stride_bk = expert_weights.stride()
        stride_bse, stride_bsn, stride_bsk = weight_scales.stride()
    else:
        num_experts, K_weights, N = expert_weights.shape
        stride_be, stride_bk, stride_bn = expert_weights.stride()
        stride_bse, stride_bsk, stride_bsn = weight_scales.stride()

    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    if use_2dblock_w and N % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"N ({N}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT}) for 2D weight scales")

    expected_input_scales = (
        (M_total // BLOCK_SIZE_DEFAULT, K // BLOCK_SIZE_DEFAULT)
        if use_2dblock_x else
        (M_total, K // BLOCK_SIZE_DEFAULT)
    )
    if trans_weights:
        expected_weight_scales = (
            (num_experts, N // BLOCK_SIZE_DEFAULT, K // BLOCK_SIZE_DEFAULT)
            if use_2dblock_w else
            (num_experts, N, K // BLOCK_SIZE_DEFAULT)
        )
    else:
        expected_weight_scales = (
            (num_experts, K // BLOCK_SIZE_DEFAULT, N // BLOCK_SIZE_DEFAULT)
            if use_2dblock_w else
            (num_experts, K // BLOCK_SIZE_DEFAULT, N)
        )
    assert input_scales.shape == torch.Size(expected_input_scales), \
        f"input_scales shape {input_scales.shape} must be {expected_input_scales}"
    assert weight_scales.shape == torch.Size(expected_weight_scales), \
        f"weight_scales shape {weight_scales.shape} must be {expected_weight_scales}"

    output = torch.zeros((M_total, N), device=inputs.device, dtype=output_dtype)

    stride_am, stride_ak = inputs.stride()
    stride_asm, stride_ask = input_scales.stride()
    stride_cm, stride_cn = output.stride()

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = (NUM_SMS, 1, 1)

    wrap_triton(_kernel_mxfp8_grouped_gemm_forward)[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        input_scales,
        weight_scales,
        stride_am, stride_ak,
        stride_be, stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_asm, stride_ask,
        stride_bse, stride_bsn, stride_bsk,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        NUM_SMS=NUM_SMS,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        SUPER_GROUP_M=32,
        USE_2DBLOCK_A=use_2dblock_x,
        USE_2DBLOCK_B=use_2dblock_w,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        LHS_FORMAT_ID=lhs_format_id,
        RHS_FORMAT_ID=rhs_format_id,
        USE_DOT_SCALED=use_dot_scaled,
    )

    return output
