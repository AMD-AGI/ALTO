# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""MXFP8 contiguous grouped GEMM — backward (dgrad + wgrad) + autograd Function."""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT, _dequantize_fp8, is_cdna4
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import (
    DGRAD_CONFIGS,
    WGRAD_CONFIGS,
    ALIGN_SIZE_M,
)
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_forward import mxfp8_grouped_gemm_forward


def _format_to_id(fmt: str) -> int:
    if fmt == "e4m3":
        return 0
    if fmt == "e5m2":
        return 1
    raise ValueError(f"Unsupported MXFP8 format: {fmt}")


# ============ dgrad: grad_input = grad_output @ expert_weights ============

@triton.autotune(
    configs=DGRAD_CONFIGS,
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
    """dgrad kernel: dX = GO @ W (N is reduction dim)."""
    if USE_2DBLOCK_GO:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)

    c_type = grad_input_ptr.dtype.element_ty
    n_rep_n: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE
    Ns: tl.constexpr = N // QUANT_BLOCK_SIZE

    pid = tl.program_id(0)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K

    if m_start < M_TOTAL:
        offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K

        if USE_2DBLOCK_GO:
            offs_m_scale = offs_m // QUANT_BLOCK_SIZE
        else:
            offs_m_scale = offs_m
        if USE_2DBLOCK_B:
            offs_k_scale = offs_k // QUANT_BLOCK_SIZE
        else:
            offs_k_scale = offs_k

        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        grad_input = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        for ni in range(num_n_tiles):
            offs_n = ni * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_n_scale = ni * n_rep_n + tl.arange(0, n_rep_n)
            mask_n = offs_n < N
            mask_n_scale = offs_n_scale < Ns

            go_ptrs = grad_output_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon
            go = tl.load(go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

            w_ptrs = b_ptr + expert_idx * stride_be + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
            w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            go_s_ptrs = go_s_ptr + offs_m_scale[:, None] * stride_gosm + offs_n_scale[None, :] * stride_gosn
            # W scales are K x N-scale even though W operand is N x K.
            b_s_ptrs = b_s_ptr + expert_idx * stride_bse + offs_k_scale[:, None] * stride_bsk + offs_n_scale[None, :] * stride_bsn
            go_s = tl.load(go_s_ptrs, mask=mask_m[:, None] & mask_n_scale[None, :], other=1)
            b_s = tl.load(b_s_ptrs, mask=mask_k[:, None] & mask_n_scale[None, :], other=1)

            if USE_DOT_SCALED:
                if LHS_FORMAT_ID == 0:
                    if RHS_FORMAT_ID == 0:
                        grad_input = tl.dot_scaled(go, go_s, "e4m3", w, b_s, "e4m3",
                                                   acc=grad_input, out_dtype=tl.float32)
                    else:
                        grad_input = tl.dot_scaled(go, go_s, "e4m3", w, b_s, "e5m2",
                                                   acc=grad_input, out_dtype=tl.float32)
                else:
                    if RHS_FORMAT_ID == 0:
                        grad_input = tl.dot_scaled(go, go_s, "e5m2", w, b_s, "e4m3",
                                                   acc=grad_input, out_dtype=tl.float32)
                    else:
                        grad_input = tl.dot_scaled(go, go_s, "e5m2", w, b_s, "e5m2",
                                                   acc=grad_input, out_dtype=tl.float32)
            else:
                go_dq = _dequantize_fp8(
                    go, go_s,
                    output_dtype=tl.float32,
                    BLOCK_M=BLOCK_SIZE_M,
                    BLOCK_N=BLOCK_SIZE_N,
                    QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                    FP8_FORMAT=LHS_FORMAT_ID,
                    IS_2D_BLOCK=False,
                    USE_ASM=False,
                )
                w_dq = tl.trans(_dequantize_fp8(
                    tl.trans(w), b_s,
                    output_dtype=tl.float32,
                    BLOCK_M=BLOCK_SIZE_K,
                    BLOCK_N=BLOCK_SIZE_N,
                    QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                    FP8_FORMAT=RHS_FORMAT_ID,
                    IS_2D_BLOCK=False,
                    USE_ASM=False,
                ))
                grad_input = tl.dot(go_dq, w_dq, acc=grad_input, out_dtype=tl.float32)

        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * stride_gim + offs_k[None, :] * stride_gik
        tl.store(grad_input_ptrs, grad_input.to(c_type), mask=mask_m[:, None] & mask_k[None, :])


# ============ wgrad: grad_weights = grad_output^T @ inputs (per expert) ============

@triton.autotune(
    configs=WGRAD_CONFIGS,
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
    """wgrad kernel: dW = GO^T @ X (M is reduction dim)."""
    if USE_2DBLOCK_GO:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)

    c_type = grad_weights_ptr.dtype.element_ty
    n_rep_m: tl.constexpr = BLOCK_SIZE_M // QUANT_BLOCK_SIZE
    Ms: tl.constexpr = M_TOTAL // QUANT_BLOCK_SIZE

    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    tiles_per_expert = n_tiles * k_tiles
    expert_id = pid // tiles_per_expert
    position_id = pid % tiles_per_expert

    if expert_id < NUM_EXPERTS:
        tile_n = position_id // k_tiles
        tile_k = position_id % k_tiles
        n_start = tile_n * BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K

        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_n = offs_n < N
        mask_k = offs_k < K

        if USE_2DBLOCK_GO:
            offs_n_scale = offs_n // QUANT_BLOCK_SIZE
        else:
            offs_n_scale = offs_n
        if USE_2DBLOCK_A:
            offs_k_scale = offs_k // QUANT_BLOCK_SIZE
        else:
            offs_k_scale = offs_k

        grad_weights = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
        for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
            group_start = group_idx * GROUP_SIZE_M
            group_expert = tl.load(indices_ptr + group_start)

            if group_expert == expert_id:
                for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                    m_start = group_start + m_offset
                    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
                    offs_m_scale = m_start // QUANT_BLOCK_SIZE + tl.arange(0, n_rep_m)
                    mask_m = offs_m < tl.minimum(group_start + GROUP_SIZE_M, M_TOTAL)
                    mask_m_scale = offs_m_scale < tl.minimum((group_start + GROUP_SIZE_M) // QUANT_BLOCK_SIZE, Ms)

                    go_ptrs = grad_output_ptr + offs_n[:, None] * stride_gon + offs_m[None, :] * stride_gom
                    go = tl.load(go_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)

                    in_ptrs = inputs_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                    inp = tl.load(in_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                    go_s_ptrs = go_s_ptr + offs_n_scale[:, None] * stride_gosn + offs_m_scale[None, :] * stride_gosm
                    # Input scales are K x M-scale even though input operand is M x K.
                    inp_s_ptrs = a_s_ptr + offs_k_scale[:, None] * stride_ask + offs_m_scale[None, :] * stride_asm
                    go_s = tl.load(go_s_ptrs, mask=mask_n[:, None] & mask_m_scale[None, :], other=1)
                    inp_s = tl.load(inp_s_ptrs, mask=mask_k[:, None] & mask_m_scale[None, :], other=1)

                    if USE_DOT_SCALED:
                        if LHS_FORMAT_ID == 0:
                            if RHS_FORMAT_ID == 0:
                                grad_weights = tl.dot_scaled(go, go_s, "e4m3", inp, inp_s, "e4m3",
                                                             acc=grad_weights, out_dtype=tl.float32)
                            else:
                                grad_weights = tl.dot_scaled(go, go_s, "e4m3", inp, inp_s, "e5m2",
                                                             acc=grad_weights, out_dtype=tl.float32)
                        else:
                            if RHS_FORMAT_ID == 0:
                                grad_weights = tl.dot_scaled(go, go_s, "e5m2", inp, inp_s, "e4m3",
                                                             acc=grad_weights, out_dtype=tl.float32)
                            else:
                                grad_weights = tl.dot_scaled(go, go_s, "e5m2", inp, inp_s, "e5m2",
                                                             acc=grad_weights, out_dtype=tl.float32)
                    else:
                        go_dq = _dequantize_fp8(
                            go, go_s,
                            output_dtype=tl.float32,
                            BLOCK_M=BLOCK_SIZE_N,
                            BLOCK_N=BLOCK_SIZE_M,
                            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                            FP8_FORMAT=LHS_FORMAT_ID,
                            IS_2D_BLOCK=False,
                            USE_ASM=False,
                        )
                        inp_dq = tl.trans(_dequantize_fp8(
                            tl.trans(inp), inp_s,
                            output_dtype=tl.float32,
                            BLOCK_M=BLOCK_SIZE_K,
                            BLOCK_N=BLOCK_SIZE_M,
                            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                            FP8_FORMAT=RHS_FORMAT_ID,
                            IS_2D_BLOCK=False,
                            USE_ASM=False,
                        ))
                        grad_weights = tl.dot(go_dq, inp_dq, acc=grad_weights, out_dtype=tl.float32)

        grad_w_ptrs = grad_weights_ptr + expert_id * stride_gbe + offs_n[:, None] * stride_gbn + offs_k[None, :] * stride_gbk
        tl.store(grad_w_ptrs, grad_weights.to(c_type), mask=mask_n[:, None] & mask_k[None, :])


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
    use_dot_scaled: Optional[bool] = None,
) -> torch.Tensor:
    """dgrad wrapper: grad_input = grad_output @ W."""
    assert grad_output.dim() == 2, f"grad_output must be 2D, got {grad_output.dim()}D"
    assert expert_weights.dim() == 3, f"expert_weights must be 3D, got {expert_weights.dim()}D"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"
    if use_dot_scaled is None:
        use_dot_scaled = is_cdna4()

    M_bufferlen, N = grad_output.shape
    M_total = expert_indices.numel()
    torch._check(M_total > 0)
    assert M_bufferlen >= M_total, \
        f"M_bufferlen ({M_bufferlen}) must be >= M_total ({M_total})"
    assert M_bufferlen % BLOCK_SIZE_DEFAULT == 0, \
        f"M_bufferlen ({M_bufferlen}) must be a multiple of block_size ({BLOCK_SIZE_DEFAULT})"
    assert M_total % ALIGN_SIZE_M == 0, \
        f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"
    if N % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"N ({N}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT})")

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    if trans_weights:
        num_experts, N_weights, K = expert_weights.shape
        stride_be, stride_bn, stride_bk = expert_weights.stride()
        stride_bse, stride_bsn, stride_bsk = expert_weight_scales.stride()
        expected_weight_scales = (
            (num_experts, N // BLOCK_SIZE_DEFAULT, K // BLOCK_SIZE_DEFAULT)
            if use_2dblock_w else
            (num_experts, N // BLOCK_SIZE_DEFAULT, K)
        )
    else:
        num_experts, K, N_weights = expert_weights.shape
        stride_be, stride_bk, stride_bn = expert_weights.stride()
        stride_bse, stride_bsk, stride_bsn = expert_weight_scales.stride()
        expected_weight_scales = (
            (num_experts, K // BLOCK_SIZE_DEFAULT, N // BLOCK_SIZE_DEFAULT)
            if use_2dblock_w else
            (num_experts, K, N // BLOCK_SIZE_DEFAULT)
        )

    assert N == N_weights, f"grad_output N ({N}) must match weight N ({N_weights})"
    if K % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"K ({K}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT})")

    expected_go_scales = (
        (M_bufferlen // BLOCK_SIZE_DEFAULT, N // BLOCK_SIZE_DEFAULT)
        if use_2dblock_x else
        (M_bufferlen, N // BLOCK_SIZE_DEFAULT)
    )
    assert go_scales.shape == torch.Size(expected_go_scales), \
        f"go_scales shape {go_scales.shape} must be {expected_go_scales}"
    assert expert_weight_scales.shape == torch.Size(expected_weight_scales), \
        f"expert_weight_scales shape {expert_weight_scales.shape} must be {expected_weight_scales}"

    grad_inputs = torch.zeros((M_bufferlen, K), device=grad_output.device, dtype=output_dtype)
    stride_gom, stride_gon = grad_output.stride()
    stride_gosm, stride_gosn = go_scales.stride()
    stride_gim, stride_gik = grad_inputs.stride()

    grid = lambda meta: (triton.cdiv(M_total, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
    wrap_triton(_kernel_mxfp8_grouped_gemm_backward_dx)[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        go_scales,
        expert_weight_scales,
        stride_gom, stride_gon,
        stride_be, stride_bn, stride_bk,
        stride_gim, stride_gik,
        stride_gosm, stride_gosn,
        stride_bse, stride_bsn, stride_bsk,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        USE_2DBLOCK_GO=use_2dblock_x,
        USE_2DBLOCK_B=use_2dblock_w,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        LHS_FORMAT_ID=lhs_format_id,
        RHS_FORMAT_ID=rhs_format_id,
        USE_DOT_SCALED=use_dot_scaled,
    )
    return grad_inputs


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
    use_dot_scaled: Optional[bool] = None,
) -> torch.Tensor:
    """wgrad wrapper: grad_weight = grad_output^T @ inputs per expert."""
    assert grad_output.dim() == 2, f"grad_output must be 2D, got {grad_output.dim()}D"
    assert inputs.dim() == 2, f"inputs must be 2D, got {inputs.dim()}D"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"
    if use_dot_scaled is None:
        use_dot_scaled = is_cdna4()

    M_bufferlen, N = grad_output.shape
    M_inputs, K = inputs.shape
    assert M_inputs == M_bufferlen, f"inputs M ({M_inputs}) must match grad_output M ({M_bufferlen})"
    M_total = expert_indices.numel()
    torch._check(M_total > 0)
    assert M_bufferlen >= M_total, \
        f"M_bufferlen ({M_bufferlen}) must be >= M_total ({M_total})"
    assert M_bufferlen % BLOCK_SIZE_DEFAULT == 0, \
        f"M_bufferlen ({M_bufferlen}) must be a multiple of block_size ({BLOCK_SIZE_DEFAULT})"
    assert M_total % ALIGN_SIZE_M == 0, \
        f"M_total ({M_total}) must be a multiple of group_size_m ({ALIGN_SIZE_M})"
    if N % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"N ({N}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT})")
    if K % BLOCK_SIZE_DEFAULT != 0:
        raise ValueError(f"K ({K}) must be divisible by block_size ({BLOCK_SIZE_DEFAULT})")

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    expected_go_scales = (
        (M_bufferlen // BLOCK_SIZE_DEFAULT, N // BLOCK_SIZE_DEFAULT)
        if use_2dblock_go else
        (M_bufferlen // BLOCK_SIZE_DEFAULT, N)
    )
    expected_input_scales = (
        (M_bufferlen // BLOCK_SIZE_DEFAULT, K // BLOCK_SIZE_DEFAULT)
        if use_2dblock_x else
        (M_bufferlen // BLOCK_SIZE_DEFAULT, K)
    )
    assert go_scales.shape == torch.Size(expected_go_scales), \
        f"go_scales shape {go_scales.shape} must be {expected_go_scales}"
    assert input_scales.shape == torch.Size(expected_input_scales), \
        f"input_scales shape {input_scales.shape} must be {expected_input_scales}"

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

    grid = lambda meta: (num_experts * triton.cdiv(N, meta["BLOCK_SIZE_N"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"]),)
    wrap_triton(_kernel_mxfp8_grouped_gemm_backward_dw)[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        go_scales,
        input_scales,
        stride_gom, stride_gon,
        stride_am, stride_ak,
        stride_gbe, stride_gbn, stride_gbk,
        stride_gosm, stride_gosn,
        stride_asm, stride_ask,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=ALIGN_SIZE_M,
        USE_2DBLOCK_GO=use_2dblock_go,
        USE_2DBLOCK_A=use_2dblock_x,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        LHS_FORMAT_ID=lhs_format_id,
        RHS_FORMAT_ID=rhs_format_id,
        USE_DOT_SCALED=use_dot_scaled,
    )
    return grad_weights


# =============== Autograd Function ===============

@torch.compiler.allow_in_graph
class MXFP8GroupedGEMM(torch.autograd.Function):
    """Autograd Function for the V1 MXFP8 grouped GEMM path."""

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
        original_dtype = inputs.dtype
        fwd_format_id = _format_to_id(fwd_format)
        bwd_grad_format_id = _format_to_id(bwd_grad_format)
        quant_axis_w = -1 if trans_weights else -2
        requant_axis_w = -2 if trans_weights else -1

        inputs_mxfp8, input_scales = torch.ops.alto.convert_to_mxfp8(
            inputs,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=fwd_format,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )
        expert_weights_mxfp8, expert_weight_scales = torch.ops.alto.convert_to_mxfp8(
            expert_weights,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=fwd_format,
            axis=quant_axis_w,
            is_2d_block=use_2dblock_w,
        )

        output = mxfp8_grouped_gemm_forward(
            inputs_mxfp8,
            expert_weights_mxfp8,
            expert_indices,
            input_scales,
            expert_weight_scales,
            trans_weights=trans_weights,
            use_2dblock_x=use_2dblock_x,
            use_2dblock_w=use_2dblock_w,
            lhs_format_id=fwd_format_id,
            rhs_format_id=fwd_format_id,
            output_dtype=original_dtype,
        )

        if use_2dblock_w:
            expert_weights_dgrad = expert_weights_mxfp8
            expert_weight_dgrad_scales = expert_weight_scales
        else:
            expert_weights_dgrad, expert_weight_dgrad_scales = torch.ops.alto.convert_to_mxfp8(
                expert_weights,
                block_size=BLOCK_SIZE_DEFAULT,
                mxfp_format=fwd_format,
                axis=requant_axis_w,
                is_2d_block=False,
            )

        if use_2dblock_x:
            inputs_wgrad = inputs_mxfp8
            input_wgrad_scales = input_scales
        else:
            inputs_wgrad, input_wgrad_scales = torch.ops.alto.convert_to_mxfp8(
                inputs,
                block_size=BLOCK_SIZE_DEFAULT,
                mxfp_format=fwd_format,
                axis=0,
                is_2d_block=False,
            )

        ctx.save_for_backward(
            inputs_wgrad,
            input_wgrad_scales,
            expert_weights_dgrad,
            expert_weight_dgrad_scales,
            expert_indices,
        )
        ctx.trans_weights = trans_weights
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.use_sr_grad = use_sr_grad
        ctx.fwd_format_id = fwd_format_id
        ctx.bwd_grad_format_id = bwd_grad_format_id
        ctx.bwd_grad_format = bwd_grad_format
        ctx.original_dtype = original_dtype
        ctx.num_experts = expert_weights.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs_wgrad, input_wgrad_scales, expert_weights_dgrad, expert_weight_dgrad_scales, expert_indices = ctx.saved_tensors

        if ctx.use_2dblock_x:
            grad_output_dgrad, grad_output_dgrad_scales = torch.ops.alto.convert_to_mxfp8(
                grad_output,
                block_size=BLOCK_SIZE_DEFAULT,
                mxfp_format=ctx.bwd_grad_format,
                axis=-1,
                is_2d_block=True,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_wgrad = grad_output_dgrad
            grad_output_wgrad_scales = grad_output_dgrad_scales
        else:
            grad_output_dgrad, grad_output_dgrad_scales = torch.ops.alto.convert_to_mxfp8(
                grad_output,
                block_size=BLOCK_SIZE_DEFAULT,
                mxfp_format=ctx.bwd_grad_format,
                axis=-1,
                is_2d_block=False,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_wgrad, grad_output_wgrad_scales = torch.ops.alto.convert_to_mxfp8(
                grad_output,
                block_size=BLOCK_SIZE_DEFAULT,
                mxfp_format=ctx.bwd_grad_format,
                axis=0,
                is_2d_block=False,
                use_sr=ctx.use_sr_grad,
            )

        grad_inputs = torch.ops.alto.mxfp8_grouped_gemm_backward_inputs(
            grad_output=grad_output_dgrad,
            expert_weights=expert_weights_dgrad,
            expert_indices=expert_indices,
            go_scales=grad_output_dgrad_scales,
            expert_weight_scales=expert_weight_dgrad_scales,
            trans_weights=ctx.trans_weights,
            use_2dblock_x=ctx.use_2dblock_x,
            use_2dblock_w=ctx.use_2dblock_w,
            lhs_format_id=ctx.bwd_grad_format_id,
            rhs_format_id=ctx.fwd_format_id,
            output_dtype=ctx.original_dtype,
        )
        grad_weights = torch.ops.alto.mxfp8_grouped_gemm_backward_weights(
            grad_output=grad_output_wgrad,
            inputs=inputs_wgrad,
            expert_indices=expert_indices,
            num_experts=ctx.num_experts,
            go_scales=grad_output_wgrad_scales,
            input_scales=input_wgrad_scales,
            trans_weights=ctx.trans_weights,
            use_2dblock_go=ctx.use_2dblock_x,
            use_2dblock_x=ctx.use_2dblock_x,
            lhs_format_id=ctx.bwd_grad_format_id,
            rhs_format_id=ctx.fwd_format_id,
            output_dtype=ctx.original_dtype,
        )
        return grad_inputs, grad_weights, None, None, None, None, None, None, None
