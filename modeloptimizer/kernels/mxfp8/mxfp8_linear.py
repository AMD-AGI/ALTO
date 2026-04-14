# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

from .mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_FORMATS,
    convert_from_mxfp8,
    convert_to_mxfp8,
    is_cdna4,
)


@triton.jit
def blockwise_mxfp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,  # 0=e4m3, 1=e5m2
    USE_ACCUMULATOR_ADD: tl.constexpr,
):
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)

    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    Ks: tl.constexpr = K // QUANT_BLOCK_SIZE

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(num_blocks_k):
        offs_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]

        offs_k_scale = i * n_rep_k + tl.arange(0, n_rep_k)
        mask_k_scale = offs_k_scale < Ks
        a_s_ptrs = a_s_ptr + offs_m[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
        # B scales are [N, K//block_size] even though B operand is [K, N]
        b_s_ptrs = b_s_ptr + offs_n[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        a_s = tl.load(a_s_ptrs, mask=mask_m[:, None] & mask_k_scale[None, :], other=1)
        b_s = tl.load(b_s_ptrs, mask=mask_n[:, None] & mask_k_scale[None, :], other=1)

        if FP8_FORMAT == 0:
            if USE_ACCUMULATOR_ADD:
                accumulator += tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", out_dtype=tl.float32)
            else:
                accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=accumulator, out_dtype=tl.float32)
        else:
            if USE_ACCUMULATOR_ADD:
                accumulator += tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2", out_dtype=tl.float32)
            else:
                accumulator = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2", acc=accumulator, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

@triton_op("modeloptimizer::blockwise_mxfp8_gemm", mutates_args={})
def blockwise_mxfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    block_size: int = BLOCK_SIZE_DEFAULT,
    output_dtype: torch.dtype = torch.float32,
    use_accumulator_add: bool = False,
) -> torch.Tensor:
    """
    Blockwise MXFP8 GEMM: C = A @ B with per-block E8M0 scales.

    A scales layout: [M, K//block_size] (trans_a=False) or [K//block_size, M] (trans_a=True).
    B scales layout: [N, K//block_size] (trans_b=True) or [K//block_size, N] (trans_b=False).

    Args:
        a: FP8 tensor, shape [M, K] or [K, M] if trans_a.
        a_s: uint8 E8M0 scales for A.
        b: FP8 tensor, shape [K, N] or [N, K] if trans_b.
        b_s: uint8 E8M0 scales for B.
        trans_a: Whether A is transposed.
        trans_b: Whether B is transposed.
        block_size: Quantization block size along K.
        output_dtype: Output dtype (float32 or bfloat16).
        use_accumulator_add: Whether to use ``accumulator += tl.dot_scaled(...)`` instead
            of ``tl.dot_scaled(..., acc=accumulator)`` for performance comparisons.
    """
    assert a.dim() == 2 and b.dim() == 2
    if a.dtype != b.dtype:
        raise ValueError(f"A/B FP8 dtypes must match, got a.dtype={a.dtype}, b.dtype={b.dtype}")

    if a.dtype == torch.float8_e4m3fn:
        fp8_format_id = 0
    elif a.dtype == torch.float8_e5m2:
        fp8_format_id = 1
    else:
        raise ValueError(f"Unsupported FP8 dtype: {a.dtype}")

    if trans_a:
        K, M = a.shape
        if K % block_size != 0:
            raise ValueError(f"K={K} must be divisible by block_size={block_size}")
        assert a_s.shape == torch.Size((K // block_size, M)), \
            f"A scale has shape {a_s.shape}, expected {(K // block_size, M)}"
        stride_ak, stride_am = a.stride()
        stride_ask, stride_asm = a_s.stride()
    else:
        M, K = a.shape
        if K % block_size != 0:
            raise ValueError(f"K={K} must be divisible by block_size={block_size}")
        assert a_s.shape == torch.Size((M, K // block_size)), \
            f"A scale has shape {a_s.shape}, expected {(M, K // block_size)}"
        stride_am, stride_ak = a.stride()
        stride_asm, stride_ask = a_s.stride()

    if trans_b:
        N, KB = b.shape
        assert KB == K, f"B reduction dim ({KB}) does not match A ({K})"
        assert b_s.shape == torch.Size((N, K // block_size)), \
            f"B scale has shape {b_s.shape}, expected {(N, K // block_size)}"
        stride_bn, stride_bk = b.stride()
        stride_bsn, stride_bsk = b_s.stride()
    else:
        KB, N = b.shape
        assert KB == K, f"B reduction dim ({KB}) does not match A ({K})"
        # b_s stored as [K//block_size, N]; kernel accesses as [N, K//block_size] via swapped strides
        assert b_s.shape == torch.Size((K // block_size, N)), \
            f"B scale has shape {b_s.shape}, expected {(K // block_size, N)}"
        stride_bk, stride_bn = b.stride()
        stride_bsk, stride_bsn = b_s.stride()

    c = a.new_empty((M, N), dtype=output_dtype)
    stride_cm, stride_cn = c.stride()

    M, N, K = int(M), int(N), int(K)
    BLOCK_SIZE_M = 64 if M >= 64 else M
    BLOCK_SIZE_N = 64 if N >= 64 else N
    BLOCK_SIZE_K = 64 if K >= 64 else K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    wrap_triton(blockwise_mxfp8_gemm_kernel)[grid](
        a, b, c, a_s, b_s,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_asm, stride_ask,
        stride_bsn, stride_bsk,
        M=M, N=N, K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=block_size,
        FP8_FORMAT=fp8_format_id,
        USE_ACCUMULATOR_ADD=use_accumulator_add,
    )
    return c


@torch.compiler.allow_in_graph
class MXFP8LinearFunction(torch.autograd.Function):
    """
    Autograd function for MXFP8 linear.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        mxfp_format: str = "e4m3",
        use_sr_grad: bool = False,
        use_accumulator_add: bool = False,
    ) -> torch.Tensor:
        if mxfp_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {mxfp_format}. Supported: {SUPPORTED_FORMATS}")
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"x dtype must be float32/bfloat16, got {x.dtype}")
        if weight.dtype != x.dtype:
            raise ValueError(f"weight dtype ({weight.dtype}) must match x dtype ({x.dtype})")

        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        output_dtype = x.dtype

        x_lp, x_scales = convert_to_mxfp8(
            x_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=-1,
            is_2d_block=False,
        )
        w_lp, w_scales = convert_to_mxfp8(
            weight,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=-1,
            is_2d_block=False,
        )

        # Save dequantized tensors for stable, portable backward path.
        x_dq = convert_from_mxfp8(
            x_lp,
            x_scales,
            output_dtype=output_dtype,
            block_size=BLOCK_SIZE_DEFAULT,
            axis=-1,
            is_2d_block=False,
        )
        w_dq = convert_from_mxfp8(
            w_lp,
            w_scales,
            output_dtype=output_dtype,
            block_size=BLOCK_SIZE_DEFAULT,
            axis=-1,
            is_2d_block=False,
        )

        if is_cdna4():
            y = blockwise_mxfp8_gemm(
                x_lp,
                x_scales,
                w_lp,
                w_scales,
                trans_b=True,
                block_size=BLOCK_SIZE_DEFAULT,
                output_dtype=output_dtype,
                use_accumulator_add=use_accumulator_add,
            )
        else:
            y = x_dq @ w_dq.T

        ctx.save_for_backward(x_dq, w_dq)
        ctx.use_sr_grad = use_sr_grad
        ctx.mxfp_format = mxfp_format
        ctx.input_shape = original_shape
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output_2d = grad_output.reshape(-1, original_shape[-1])

        grad_lp, grad_scales = convert_to_mxfp8(
            grad_output_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=ctx.mxfp_format,
            axis=-1,
            is_2d_block=False,
            use_sr=ctx.use_sr_grad,
        )
        grad_dq = convert_from_mxfp8(
            grad_lp,
            grad_scales,
            output_dtype=grad_output.dtype,
            block_size=BLOCK_SIZE_DEFAULT,
            axis=-1,
            is_2d_block=False,
        )

        grad_input = grad_dq @ w_dq
        grad_weight = grad_dq.T @ x_dq
        return grad_input.view(*ctx.input_shape), grad_weight, None, None, None


def _to_mxfp8_then_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    mxfp_format: str = "e4m3",
    use_sr_grad: bool = False,
    use_accumulator_add: bool = False,
) -> torch.Tensor:
    return MXFP8LinearFunction.apply(
        a,
        b,
        mxfp_format,
        use_sr_grad,
        use_accumulator_add,
    )