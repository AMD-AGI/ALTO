# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
from torch.library import triton_op, wrap_triton
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import PlacementList
from torch.distributed.tensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)

import triton
import triton.language as tl

from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.fp4.fp4_common.grad_clip_registry import apply_clip, get as get_grad_clip_cfg
from alto.kernels.hadamard_transform import (HadamardTransform, HadamardFactory)
from alto.kernels.dge import dge_bwd
from .mxfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    is_cdna4,
)
from .macro_block_scaling import macro_block_scaling, macro_block_descaling


@triton.jit
def blockwise_mxfp4_gemm_kernel(
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
    USE_2DBLOCK_A: tl.constexpr,
    USE_2DBLOCK_B: tl.constexpr,
    K_PACK_A: tl.constexpr,
    K_PACK_B: tl.constexpr,
):
    if USE_2DBLOCK_A:
        tl.assume(BLOCK_SIZE_M % QUANT_BLOCK_SIZE == 0)
    if USE_2DBLOCK_B:
        tl.assume(BLOCK_SIZE_N % QUANT_BLOCK_SIZE == 0)
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)

    PACKED_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_M // 2
    PACKED_M: tl.constexpr = M // 2
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N // 2
    PACKED_N: tl.constexpr = N // 2
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // 2
    PACKED_K: tl.constexpr = K // 2
    # number of MXFP blocks inside a tile
    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    # total number of scales
    Ks: tl.constexpr = (K // QUANT_BLOCK_SIZE)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_m_pack = pid_m * PACKED_BLOCK_SIZE_M + tl.arange(0, PACKED_BLOCK_SIZE_M)
    mask_m_pack = offs_m_pack < PACKED_M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    offs_n_pack = pid_n * PACKED_BLOCK_SIZE_N + tl.arange(0, PACKED_BLOCK_SIZE_N)
    mask_n_pack = offs_n_pack < PACKED_N

    if USE_2DBLOCK_A:
        offs_m_scale = offs_m // QUANT_BLOCK_SIZE
    else:
        offs_m_scale = offs_m
    if USE_2DBLOCK_B:
        offs_n_scale = offs_n // QUANT_BLOCK_SIZE
    else:
        offs_n_scale = offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(num_blocks_k):
        offs_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        offs_k_pack = i * PACKED_BLOCK_SIZE_K + tl.arange(0, PACKED_BLOCK_SIZE_K)
        mask_k_pack = offs_k_pack < PACKED_K
        if K_PACK_A:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_pack[None, :] * stride_ak
            mask_a = mask_m[:, None] & mask_k_pack[None, :]
        else:
            a_ptrs = a_ptr + offs_m_pack[:, None] * stride_am + offs_k[None, :] * stride_ak
            mask_a = mask_m_pack[:, None] & mask_k[None, :]
        if K_PACK_B:
            b_ptrs = b_ptr + offs_k_pack[:, None] * stride_bk + offs_n[None, :] * stride_bn
            mask_b = mask_k_pack[:, None] & mask_n[None, :]
        else:
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n_pack[None, :] * stride_bn
            mask_b = mask_k[:, None] & mask_n_pack[None, :]
        offs_k_scale = i * n_rep_k + tl.arange(0, n_rep_k)
        mask_k_scale = offs_k_scale < Ks
        a_s_ptrs = a_s_ptr + offs_m_scale[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
        # B scales are N x K even though B operand is K x N.
        b_s_ptrs = b_s_ptr + offs_n_scale[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk

        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        a_s = tl.load(a_s_ptrs, mask=mask_m[:, None] & mask_k_scale[None, :], other=1)
        b_s = tl.load(b_s_ptrs, mask=mask_n[:, None] & mask_k_scale[None, :], other=1)
        accumulator += tl.dot_scaled(a,
                                     a_s,
                                     "e2m1",
                                     b,
                                     b_s,
                                     "e2m1",
                                     lhs_k_pack=K_PACK_A,
                                     rhs_k_pack=K_PACK_B,
                                     out_dtype=tl.float32)

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


@triton_op("torchtitan::blockwise_mxfp4_gemm", mutates_args={})
def blockwise_mxfp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    use_2dblock_a: bool = False,
    use_2dblock_b: bool = False,
    k_pack_a: bool = True,
    k_pack_b: bool = True,
    trans_a: bool = False,
    trans_b: bool = False,
    block_size: int = BLOCK_SIZE_DEFAULT,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # M, N, K here refers to the shape of unpacked fp4
    assert a.dim() == 2 and b.dim() == 2
    if trans_a:
        K, M = a.shape
        if k_pack_a:
            K *= 2
        else:
            M *= 2
        if use_2dblock_a:
            assert a_s.shape == torch.Size(
                (K // block_size, M //
                 block_size)), f"A scale has shape {a_s.shape}, but {(K // block_size, M // block_size)} is expected."
        else:
            assert a_s.shape == torch.Size(
                (K // block_size, M)), f"A scale has shape {a_s.shape}, but {(K // block_size, M)} is expected."
        stride_ask, stride_asm = a_s.stride()
        stride_ak, stride_am = a.stride()
    else:
        M, K = a.shape
        if k_pack_a:
            K *= 2
        else:
            M *= 2
        if use_2dblock_a:
            assert a_s.shape == torch.Size(
                (M // block_size, K //
                 block_size)), f"A scale has shape {a_s.shape}, but {(M // block_size, K // block_size)} is expected."
        else:
            assert a_s.shape == torch.Size(
                (M, K // block_size)), f"A scale has shape {a_s.shape}, but {(M, K // block_size)} is expected."
        stride_asm, stride_ask = a_s.stride()
        stride_am, stride_ak = a.stride()

    if trans_b:
        N, KB = b.shape
        if k_pack_b:
            KB *= 2
        else:
            N *= 2
        assert KB == K, f"Input B has unpacked reduction shape ({KB}) but {(K)} is expected."
        stride_bn, stride_bk = b.stride()
        if use_2dblock_b:
            assert b_s.shape == torch.Size((N // block_size, K // block_size))
        else:
            assert b_s.shape == torch.Size((N, K // block_size))
        stride_bsn, stride_bsk = b_s.stride()
    else:
        KB, N = b.shape
        if k_pack_b:
            KB *= 2
        else:
            N *= 2
        assert KB == K, f"Input B has unpacked reduction shape ({KB}) but {(K)} is expected."
        stride_bk, stride_bn = b.stride()
        if use_2dblock_b:
            assert b_s.shape == torch.Size((K // block_size, N // block_size))
        else:
            assert b_s.shape == torch.Size((K // block_size, N))
        stride_bsk, stride_bsn = b_s.stride()

    c = a.new_empty((M, N), dtype=output_dtype)
    stride_cm, stride_cn = c.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_M = 64 if M >= 64 else M
    BLOCK_SIZE_N = 64 if N >= 64 else N
    BLOCK_SIZE_K = 64 if K >= 64 else K
    wrap_triton(blockwise_mxfp4_gemm_kernel)[grid](
        a,
        b,
        c,
        a_s,
        b_s,
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
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=block_size,
        USE_2DBLOCK_A=use_2dblock_a,
        USE_2DBLOCK_B=use_2dblock_b,
        K_PACK_A=k_pack_a,
        K_PACK_B=k_pack_b,
    )
    return c


@torch.compiler.allow_in_graph
class MXFP4LinearFunction(torch.autograd.Function):
    """
    Custom autograd function for MXFP linear operations.
    This function handles the forward and backward passes for the linear layer.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_dge,
        clip_mode,
        use_macro_block_scaling,
        hadamard_transform: Optional[HadamardTransform] = None,
        module_id: Optional[int] = None,
    ):
        """
        Forward pass for the blockwise FP8 linear operation.

        Args:
            ctx: Context object to save information for backward pass.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor, optional): Bias tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        # Normalize any training-time weight wrapper to its underlying local
        # tensor before saving / quantizing it.  As with NVFP4, DTensor inputs
        # are normally already unwrapped to local tensors before this function
        # is entered.
        weight = unwrap_weight_wrapper(weight)

        original_shape = x.shape
        original_dtype = x.dtype
        x = x.reshape(-1, original_shape[-1])  # Ensure x is 2D

        if use_macro_block_scaling:
            x_scaled, x_mbs = macro_block_scaling(x, axis=-1, use_2d_block=use_2dblock_x)
            w_scaled, w_mbs = macro_block_scaling(weight, axis=-1, use_2d_block=use_2dblock_w)
        else:
            x_scaled = x
            x_mbs = x.new_empty([])
            w_scaled = weight
            w_mbs = weight.new_empty([])

        x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(
            x_scaled,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )

        w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(
            w_scaled,
            axis=-1,
            is_2d_block=use_2dblock_w,
        )

        if is_cdna4():
            assert not use_macro_block_scaling, "Macro block scaling is not supported in real MXFP4 kernels"
            y = torch.ops.torchtitan.blockwise_mxfp4_gemm(
                x_mxfp4,
                x_scale,
                w_mxfp4,
                w_scale,
                use_2dblock_a=use_2dblock_x,
                use_2dblock_b=use_2dblock_w,
                trans_b=True,
                output_dtype=original_dtype,
            )
        else:
            x_dq = torch.ops.torchtitan.convert_from_mxfp4(
                x_mxfp4,
                x_scale,
                original_dtype,
                axis=-1,
                is_2d_block=use_2dblock_x,
            )
            w_dq = torch.ops.torchtitan.convert_from_mxfp4(
                w_mxfp4,
                w_scale,
                original_dtype,
                axis=-1,
                is_2d_block=use_2dblock_w,
            )

            if use_macro_block_scaling:
                x_dq = macro_block_descaling(x_dq, x_mbs, axis=-1, use_2d_block=use_2dblock_x)
                w_dq = macro_block_descaling(w_dq, w_mbs, axis=-1, use_2d_block=use_2dblock_w)

            y = x_dq @ w_dq.T

        if not use_2dblock_w:
            if use_macro_block_scaling:
                w_scaled, w_mbs = macro_block_scaling(weight, axis=0, use_2d_block=False)
            else:
                w_scaled = weight
                w_mbs = weight.new_empty([])

            w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(
                w_scaled,
                axis=0,
                is_2d_block=False,
            )
            if not is_cdna4():
                w_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    w_mxfp4,
                    w_scale,
                    original_dtype,
                    axis=0,
                    is_2d_block=False,
                )
                if use_macro_block_scaling:
                    w_dq = macro_block_descaling(w_dq, w_mbs, axis=0, use_2d_block=False)

        if not use_2dblock_x:
            if hadamard_transform is not None:
                x = hadamard_transform(x, left_mul=True)
            if use_macro_block_scaling:
                x_scaled, x_mbs = macro_block_scaling(x, axis=0, use_2d_block=False)
            else:
                x_scaled = x
                x_mbs = x.new_empty([])
            x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(
                x_scaled,
                axis=0,
                is_2d_block=False,
                clip_mode=clip_mode,
            )
            if not is_cdna4():
                x_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    x_mxfp4,
                    x_scale,
                    original_dtype,
                    axis=0,
                    is_2d_block=False,
                    clip_mode=clip_mode,
                )
                if use_macro_block_scaling:
                    x_dq = macro_block_descaling(x_dq, x_mbs, axis=0, use_2d_block=False)
        if is_cdna4():
            ctx.save_for_backward(x_mxfp4, x_scale, w_mxfp4, w_scale, x_mbs, w_mbs)
        else:
            ctx.save_for_backward(x_dq, w_dq, w_mxfp4)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.original_dtype = original_dtype
        ctx.use_sr_grad = use_sr_grad
        ctx.hadamard_transform = hadamard_transform
        ctx.use_dge = use_dge
        ctx.clip_mode = clip_mode
        ctx.use_macro_block_scaling = use_macro_block_scaling
        ctx.module_id = module_id

        return y.view(*original_shape[:-1], -1)  # Reshape back to original

    @staticmethod
    def backward(ctx, grad_output):
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])  # Ensure grad_output is 2D
        # PyTorch disables autocast inside autograd.Function.backward, so without FSDP's
        # MixedPrecisionPolicy grad_output can arrive in a dtype that differs from the saved
        # x_dq / w_dq (which were cast to ctx.original_dtype in forward). Use the dtype
        # committed by forward to keep the non-CDNA4 matmul below well-typed.
        original_dtype = ctx.original_dtype

        # Site ①: clip grad_output before it enters the quantizer.
        _clip_cfg = get_grad_clip_cfg(ctx.module_id)
        if _clip_cfg is not None and _clip_cfg.clip_grad_output:
            grad_output = apply_clip(grad_output, _clip_cfg.grad_output_max_norm,
                                     _clip_cfg.grad_output_clip_value)

        if is_cdna4():
            inputs_mxfp4, input_scales, weight_mxfp4, weight_scales, x_mbs, w_mbs = ctx.saved_tensors
        else:
            x_dq, w_dq, weight_mxfp4 = ctx.saved_tensors

        # dequant-quant as deepseek-v3 paper
        if ctx.use_2dblock_x:
            if ctx.use_macro_block_scaling:
                grad_output_scaled, grad_output_mbs = macro_block_scaling(grad_output,
                                                                          axis=-1,
                                                                          use_2d_block=ctx.use_2dblock_x)
            else:
                grad_output_scaled = grad_output
                grad_output_mbs = grad_output.new_empty([])
            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output_scaled,
                axis=-1,
                is_2d_block=True,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_mxfp4_m = grad_output_mxfp4
            grad_output_scales_m = grad_output_scales
            grad_output_mbs_m = grad_output_mbs

            if not is_cdna4():
                grad_output_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4,
                    grad_output_scales,
                    original_dtype,
                    axis=-1,
                    is_2d_block=True,
                )
                if ctx.use_macro_block_scaling:
                    grad_output_dq = macro_block_descaling(grad_output_dq, grad_output_mbs, axis=-1, use_2d_block=True)
                grad_output_m_dq = grad_output_dq
        else:
            if ctx.use_macro_block_scaling:
                grad_output_scaled, grad_output_mbs = macro_block_scaling(grad_output, axis=-1, use_2d_block=False)
            else:
                grad_output_scaled = grad_output
                grad_output_mbs = grad_output.new_empty([])
            grad_output_mxfp4, grad_output_scales = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output_scaled,
                axis=-1,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
            )

            if ctx.hadamard_transform is not None:
                grad_output = ctx.hadamard_transform(grad_output, left_mul=True)
            if ctx.use_macro_block_scaling:
                grad_output_scaled_m, grad_output_mbs_m = macro_block_scaling(grad_output, axis=0, use_2d_block=False)
            else:
                grad_output_scaled_m = grad_output
                grad_output_mbs_m = grad_output.new_empty([])
            grad_output_mxfp4_m, grad_output_scales_m = torch.ops.torchtitan.convert_to_mxfp4(
                grad_output_scaled_m,
                axis=0,
                use_sr=ctx.use_sr_grad,
                is_2d_block=False,
                clip_mode=ctx.clip_mode,
            )

            if not is_cdna4():
                grad_output_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4,
                    grad_output_scales,
                    original_dtype,
                    axis=-1,
                    is_2d_block=False,
                )
                grad_output_m_dq = torch.ops.torchtitan.convert_from_mxfp4(
                    grad_output_mxfp4_m,
                    grad_output_scales_m,
                    original_dtype,
                    axis=0,
                    is_2d_block=False,
                    clip_mode=ctx.clip_mode,
                )
                if ctx.use_macro_block_scaling:
                    grad_output_dq = macro_block_descaling(grad_output_dq, grad_output_mbs, axis=-1, use_2d_block=False)
                    grad_output_m_dq = macro_block_descaling(grad_output_m_dq,
                                                             grad_output_mbs_m,
                                                             axis=0,
                                                             use_2d_block=False)

        # Compute gradients
        if is_cdna4():
            grad_inputs = torch.ops.torchtitan.blockwise_mxfp4_gemm(
                grad_output_mxfp4,
                grad_output_scales,
                weight_mxfp4,
                weight_scales,
                use_2dblock_a=ctx.use_2dblock_x,
                use_2dblock_b=ctx.use_2dblock_w,
                k_pack_b=not ctx.use_2dblock_w,
                output_dtype=ctx.original_dtype,
            )
            grad_weights = torch.ops.torchtitan.blockwise_mxfp4_gemm(
                grad_output_mxfp4_m,
                grad_output_scales_m,
                inputs_mxfp4,
                input_scales,
                use_2dblock_a=ctx.use_2dblock_x,
                use_2dblock_b=ctx.use_2dblock_x,
                trans_a=True,
                k_pack_a=not ctx.use_2dblock_x,
                k_pack_b=not ctx.use_2dblock_x,
                output_dtype=ctx.original_dtype,
            )
            if ctx.clip_mode == "static":
                grad_weights *= (16.0 / 9.0)
        else:
            grad_inputs = grad_output_dq @ w_dq
            grad_weights = grad_output_m_dq.T @ x_dq

        # Site ②: clip grad_weights after the wgrad GEMM, before optimizer sees it.
        if _clip_cfg is not None and _clip_cfg.clip_grad_weight:
            grad_weights = apply_clip(grad_weights, _clip_cfg.grad_weight_max_norm,
                                      _clip_cfg.grad_weight_clip_value)

        if ctx.use_dge:
            if ctx.use_2dblock_w:
                scale_shape = [
                    weight_mxfp4.shape[0] // BLOCK_SIZE_DEFAULT,
                    (weight_mxfp4.shape[1] * 2) // BLOCK_SIZE_DEFAULT,
                ]
            else:
                scale_shape = [
                    (weight_mxfp4.shape[0] * 2) // BLOCK_SIZE_DEFAULT,
                    weight_mxfp4.shape[1],
                ]
            scale_ones = weight_mxfp4.new_ones(scale_shape, dtype=torch.uint8) * 127
            w_fp4_values = torch.ops.torchtitan.convert_from_mxfp4(
                weight_mxfp4,
                scale_ones,
                ctx.original_dtype,
                axis=0 if not ctx.use_2dblock_w else -1,
                is_2d_block=ctx.use_2dblock_w,
            )
            grad_weights *= dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)

        return grad_inputs.view(*original_shape[:-1], -1), grad_weights, None, None, None, None, None, None, None, None


def _to_mxfp4_then_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: int,
    use_dge: bool,
    clip_mode: str,
    use_hadamard: bool,
    use_macro_block_scaling: bool = False,
    module_id: Optional[int] = None,
) -> torch.Tensor:
    if use_hadamard:
        with torch.no_grad():
            hadamard_transform = HadamardFactory.create_transform(device=b.device)
    else:
        hadamard_transform = None
    y = MXFP4LinearFunction.apply(
        a,
        b,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_dge,
        clip_mode,
        use_macro_block_scaling,
        hadamard_transform,
        module_id,
    )
    return y
