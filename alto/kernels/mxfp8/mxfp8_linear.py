# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper

from .mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_FORMATS,
    _dequantize_fp8,
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
    USE_DOT_SCALED: tl.constexpr,
    DEBUG_PRINT: tl.constexpr = False,
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

    if DEBUG_PRINT:
        tl.device_print("=== GEMM DEBUG ===")
        tl.device_print("pid_m", pid_m)
        tl.device_print("pid_n", pid_n)
        tl.device_print("M, N, K", M, N, K)
        tl.device_print("stride_am, stride_ak", stride_am, stride_ak)
        tl.device_print("stride_bn, stride_bk", stride_bn, stride_bk)
        tl.device_print("stride_asm, stride_ask", stride_asm, stride_ask)
        tl.device_print("stride_bsn, stride_bsk", stride_bsn, stride_bsk)

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
        # B scales are loaded as [N, K//block_size] even though the logical B operand is [K, N].
        b_s_ptrs = b_s_ptr + offs_n[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        a_s = tl.load(a_s_ptrs, mask=mask_m[:, None] & mask_k_scale[None, :], other=1)
        b_s = tl.load(b_s_ptrs, mask=mask_n[:, None] & mask_k_scale[None, :], other=1)

        if USE_DOT_SCALED:
            if FP8_FORMAT == 0:
                accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=accumulator, out_dtype=tl.float32)
            else:
                accumulator = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2", acc=accumulator, out_dtype=tl.float32)
        else:
            a_dq = _dequantize_fp8(
                a,
                a_s,
                output_dtype=tl.float32,
                BLOCK_M=BLOCK_SIZE_M,
                BLOCK_N=BLOCK_SIZE_K,
                QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                FP8_FORMAT=FP8_FORMAT,
                IS_2D_BLOCK=False,
                USE_ASM=False,
            )
            b_dq = tl.trans(_dequantize_fp8(
                tl.trans(b),
                b_s,
                output_dtype=tl.float32,
                BLOCK_M=BLOCK_SIZE_N,
                BLOCK_N=BLOCK_SIZE_K,
                QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
                FP8_FORMAT=FP8_FORMAT,
                IS_2D_BLOCK=False,
                USE_ASM=False,
            ))

            if DEBUG_PRINT:
                tl.device_print("--- K iteration", i)
                tl.device_print("a_dq", a_dq)
                tl.device_print("b_dq", b_dq)

            accumulator = tl.dot(a_dq, b_dq, acc=accumulator, out_dtype=tl.float32)

            if DEBUG_PRINT:
                tl.device_print("accumulator after dot", accumulator)

    c = accumulator.to(c_ptr.dtype.element_ty)
    if DEBUG_PRINT:
        tl.device_print("final c", c)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

@triton_op("alto::blockwise_mxfp8_gemm", mutates_args={})
def blockwise_mxfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    block_size: int = BLOCK_SIZE_DEFAULT,
    output_dtype: torch.dtype = torch.float32,
    use_dot_scaled: Optional[bool] = None,
    debug_print: bool = False,
) -> torch.Tensor:
    """
    Blockwise MXFP8 GEMM: C = A @ B with per-block E8M0 scales.

    A scales layout: [M, K//block_size] (trans_a=False) or [K//block_size, M] (trans_a=True).
    B scales layout: [N, K//block_size] (trans_b=True) or [K//block_size, N] (trans_b=False).
    """
    if use_dot_scaled is None:
        use_dot_scaled = is_cdna4()

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

    BLOCK_SIZE_M = 64 if M >= 64 else M
    BLOCK_SIZE_N = 64 if N >= 64 else N
    if use_dot_scaled:
        # Keep one quant block per dot_scaled call. When a single dot_scaled spans
        # multiple 32-wide scale groups, outlier-heavy inputs diverge sharply from
        # the dequantize-then-matmul reference.
        BLOCK_SIZE_K = block_size
    else:
        BLOCK_SIZE_K = 64 if K >= 64 else K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    if debug_print:
        print(f"\n{'='*60}")
        print(f"DEQUANT DEBUG GEMM: M={M}, N={N}, K={K}")
        print(f"trans_a={trans_a}, trans_b={trans_b}")
        print(f"a.shape={a.shape}, a.stride()={a.stride()}")
        print(f"b.shape={b.shape}, b.stride()={b.stride()}")
        print(f"a_s.shape={a_s.shape}, a_s.stride()={a_s.stride()}")
        print(f"b_s.shape={b_s.shape}, b_s.stride()={b_s.stride()}")
        print(f"stride_am={stride_am}, stride_ak={stride_ak}")
        print(f"stride_bn={stride_bn}, stride_bk={stride_bk}")
        print(f"stride_asm={stride_asm}, stride_ask={stride_ask}")
        print(f"stride_bsn={stride_bsn}, stride_bsk={stride_bsk}")
        print(f"{'='*60}\n")

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
        USE_DOT_SCALED=use_dot_scaled,
        DEBUG_PRINT=debug_print,
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
    ) -> torch.Tensor:
        weight = unwrap_weight_wrapper(weight)

        if mxfp_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {mxfp_format}. Supported: {SUPPORTED_FORMATS}")
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"x dtype must be float32/bfloat16, got {x.dtype}")
        if weight.dtype != x.dtype:
            raise ValueError(f"weight dtype ({weight.dtype}) must match x dtype ({x.dtype})")

        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        output_dtype = x.dtype

        x_lp, x_scales = torch.ops.alto.convert_to_mxfp8(
            x_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=-1,
            is_2d_block=False,
        )
        w_lp, w_scales = torch.ops.alto.convert_to_mxfp8(
            weight,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=-1,
            is_2d_block=False,
        )

        y = torch.ops.alto.blockwise_mxfp8_gemm(
            x_lp,
            x_scales,
            w_lp,
            w_scales,
            trans_b=True,
            block_size=BLOCK_SIZE_DEFAULT,
            output_dtype=output_dtype,
        )

        x_m_lp, x_m_scales = torch.ops.alto.convert_to_mxfp8(
            x_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=0,
            is_2d_block=False,
        )
        w_m_lp, w_m_scales = torch.ops.alto.convert_to_mxfp8(
            weight,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=mxfp_format,
            axis=0,
            is_2d_block=False,
        )

        ctx.save_for_backward(x_m_lp, x_m_scales, w_m_lp, w_m_scales)
        ctx.use_sr_grad = use_sr_grad
        ctx.mxfp_format = mxfp_format
        ctx.output_dtype = output_dtype
        ctx.input_shape = original_shape
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_m_lp, x_m_scales, w_m_lp, w_m_scales = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output_2d = grad_output.reshape(-1, original_shape[-1])

        grad_lp, grad_scales = torch.ops.alto.convert_to_mxfp8(
            grad_output_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=ctx.mxfp_format,
            axis=-1,
            is_2d_block=False,
            use_sr=ctx.use_sr_grad,
        )
        grad_m_lp, grad_m_scales = torch.ops.alto.convert_to_mxfp8(
            grad_output_2d,
            block_size=BLOCK_SIZE_DEFAULT,
            mxfp_format=ctx.mxfp_format,
            axis=0,
            is_2d_block=False,
            use_sr=ctx.use_sr_grad,
        )

        grad_input = torch.ops.alto.blockwise_mxfp8_gemm(
            grad_lp,
            grad_scales,
            w_m_lp,
            w_m_scales,
            block_size=BLOCK_SIZE_DEFAULT,
            output_dtype=ctx.output_dtype,
        )
        grad_weight = torch.ops.alto.blockwise_mxfp8_gemm(
            grad_m_lp,
            grad_m_scales,
            x_m_lp,
            x_m_scales,
            trans_a=True,
            block_size=BLOCK_SIZE_DEFAULT,
            output_dtype=ctx.output_dtype,
        )
        return grad_input.view(*ctx.input_shape), grad_weight, None, None


def _to_mxfp8_then_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    mxfp_format: str = "e4m3",
    use_sr_grad: bool = False,
) -> torch.Tensor:
    # Backward-compatible public wrapper around MXFP8LinearFunction.apply.
    return MXFP8LinearFunction.apply(
        a,
        b,
        mxfp_format,
        use_sr_grad,
    )