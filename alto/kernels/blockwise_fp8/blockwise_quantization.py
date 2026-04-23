# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch/torchtitan licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

from typing import Tuple

import torch
from torch.library import triton_op, wrap_triton
# from torch.library import custom_op as triton_op
# wrap_triton = lambda x: x
import triton
import triton.language as tl
from triton import Config

from .utils import (
    get_f8_fwd_dtype,
    F8_FWD_MAX,
)

# Original implementation at https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
torch_dtype = torch.bfloat16
f8_fwd_max_const = x = triton.language.constexpr(F8_FWD_MAX)


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_M": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def blockwise_fp8_gemm_kernel(
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
    stride_bsm,
    stride_bsk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        start_k = i * BLOCK_SIZE_K
        offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        a_s_ptrs = a_s_ptr + offs_m * stride_asm + i * stride_ask
        b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * stride_bsm + i * stride_bsk
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        a_s = tl.load(a_s_ptrs, mask=mask_m, other=1.0)
        b_s = tl.load(b_s_ptrs, mask=mask_n, other=1.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (mask_m[:, None] & mask_n[None, :])
    tl.store(c_ptrs, c, mask=mask)


@triton_op("torchtitan::blockwise_fp8_gemm", mutates_args={})
def blockwise_fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "a and b must be 2D"
    assert a_s.dim() == 2 and b_s.dim() == 2, "a_s and b_s must be 2D"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_zeros(*a.size()[:-1], N, dtype=torch_dtype)
    stride_am, stride_ak = a.stride()
    stride_bn, stride_bk = b.stride()
    stride_cm, stride_cn = c.stride()
    stride_asm, stride_ask = a_s.stride()
    stride_bsm, stride_bsk = b_s.stride()
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, block_size),
    )
    wrap_triton(blockwise_fp8_gemm_kernel)[grid](
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
        stride_bsm,
        stride_bsk,
        M,
        N,
        K,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_K=block_size,
    )
    return c


@blockwise_fp8_gemm.register_fake
def blockwise_fp8_gemm_fake(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fake implementation of the block-wise FP8 GEMM for testing purposes.
    Returns a zero tensor of the appropriate shape.
    """
    assert a.dim() == 2 and b.dim() == 2, "a and b must be 2D"
    assert a_s.dim() == 2 and b_s.dim() == 2, "a_s and b_s must be 2D"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_zeros(*a.size()[:-1], N, dtype=torch_dtype)
    return c


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def blockwise_fp8_gemm_backward_weight_kernel(
    grad_weight_ptr,
    go_ptr,
    go_s_ptr,
    a_ptr,
    a_s_ptr,
    stride_gwn,
    stride_gwk,
    stride_gom,
    stride_gon,
    stride_gsm,
    stride_gsn,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_n = offs_n < N
    mask_k = offs_k < K
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for mi in range(tl.cdiv(M, BLOCK_SIZE_M)):
        start_m = mi * BLOCK_SIZE_M
        offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offs_m < M
        go_ptrs = go_ptr + offs_m[None, :] * stride_gom + offs_n[:, None] * stride_gon
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

        go_s_ptrs = go_s_ptr + mi * stride_gsm + offs_n * stride_gsn
        a_s_ptrs = a_s_ptr + mi * stride_asm + offs_k * stride_ask
        go = tl.load(go_ptrs, mask=mask_m[None, :] & mask_n[:, None], other=0.0)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        go_s = tl.load(go_s_ptrs, mask=mask_n, other=1.0)
        a_s = tl.load(a_s_ptrs, mask=mask_k, other=1.0)
        accumulator += tl.dot(go, a) * go_s[:, None] * a_s[None, :]

    c = accumulator.to(grad_weight_ptr.dtype.element_ty)
    c_ptrs = grad_weight_ptr + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
    tl.store(c_ptrs, c, mask=mask_n[:, None] & mask_k[None, :])


@triton_op("torchtitan::blockwise_fp8_gemm_backward_weight", mutates_args={})
def blockwise_fp8_gemm_backward_weight(
    grad_output: torch.Tensor,  # [M, N]
    go_s: torch.Tensor,  # [M // block_size, N]
    a: torch.Tensor,  # [M, K]
    a_s: torch.Tensor,  # [M // block_size, K]
    block_size: int = 128,
) -> torch.Tensor:  # [N, K]
    assert grad_output.dim() == 2 and a.dim() == 2, "Input tensors must have 2 dimensions"
    assert grad_output.size(0) % block_size == 0, (
        f"grad_output's first dimension must be divisible by block_size (block_size={block_size})")
    assert a.size(0) % block_size == 0, (
        f"a's first dimension must be divisible by block_size (block_size={block_size})")
    M, N = grad_output.size()
    _, K = a.size()
    grad_weight = grad_output.new_zeros(N, K, dtype=torch_dtype)
    stride_gwn, stride_gwk = grad_weight.stride()
    stride_gom, stride_gon = grad_output.stride()
    stride_gsm, stride_gsn = go_s.stride()
    stride_am, stride_ak = a.stride()
    stride_asm, stride_ask = a_s.stride()
    grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )
    wrap_triton(blockwise_fp8_gemm_backward_weight_kernel)[grid](
        grad_weight,
        grad_output,
        go_s,
        a,
        a_s,
        stride_gwn,
        stride_gwk,
        stride_gom,
        stride_gon,
        stride_gsm,
        stride_gsn,
        stride_am,
        stride_ak,
        stride_asm,
        stride_ask,
        M,
        N,
        K,
        BLOCK_SIZE_M=block_size,
    )
    return grad_weight


@blockwise_fp8_gemm_backward_weight.register_fake
def blockwise_fp8_gemm_backward_weight_fake(
    grad_output: torch.Tensor,  # [M, N]
    go_s: torch.Tensor,  # [M // block_size, N]
    a: torch.Tensor,  # [M, K]
    a_s: torch.Tensor,  # [M // block_size, K]
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fake implementation of the block-wise FP8 GEMM backward weight for testing purposes.
    Returns a zero tensor of the appropriate shape.
    """
    assert grad_output.dim() == 2 and a.dim() == 2, "Input tensors must have 2 dimensions"
    assert grad_output.size(0) % block_size == 0, (
        f"grad_output's first dimension must be divisible by block_size (block_size={block_size})")
    assert a.size(0) % block_size == 0, (
        f"a's first dimension must be divisible by block_size (block_size={block_size})")

    M, N = grad_output.size()
    _, K = a.size()
    grad_weight = grad_output.new_zeros(N, K, dtype=torch_dtype)
    return grad_weight


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_SIZE_M": 128,
            },
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def blockwise_fp8_gemm_backward_act_kernel(
    grad_act_ptr,
    go_ptr,
    go_s_ptr,
    b_ptr,
    b_s_ptr,
    stride_gam,
    stride_gak,
    stride_gom,
    stride_gon,
    stride_gsm,
    stride_gsn,
    stride_bn,
    stride_bk,
    stride_bsm,
    stride_bsk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for ni in range(tl.cdiv(N, BLOCK_SIZE_N)):
        start_n = ni * BLOCK_SIZE_N
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        go_ptrs = go_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        go_s_ptrs = go_s_ptr + offs_m * stride_gsm + ni * stride_gsn
        b_s_ptrs = b_s_ptr + ni * stride_bsm + (offs_k // BLOCK_SIZE_N) * stride_bsk
        go = tl.load(go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        go_s = tl.load(go_s_ptrs, mask=mask_m, other=1.0)
        b_s = tl.load(b_s_ptrs, mask=mask_k, other=1.0)
        accumulator += tl.dot(go, b) * go_s[:, None] * b_s[None, :]
    c = accumulator.to(grad_act_ptr.dtype.element_ty)
    c_ptrs = grad_act_ptr + offs_m[:, None] * stride_gam + offs_k[None, :] * stride_gak
    mask = (mask_m[:, None] & mask_k[None, :])
    tl.store(c_ptrs, c, mask=mask)


@triton_op("torchtitan::blockwise_fp8_gemm_backward_act", mutates_args={})
def blockwise_fp8_gemm_backward_act(
    grad_output: torch.Tensor,  # [M, N]
    go_s: torch.Tensor,  # [M, N // block_size]
    b: torch.Tensor,  # [N, K]
    b_s: torch.Tensor,  # [N // block_size, K // block_size]
    block_size: int = 128,
) -> torch.Tensor:  # [M, K]
    assert grad_output.dim() == 2 and b.dim() == 2, "Input tensors must have 3 and 2 dimensions respectively"
    assert b.size(0) // block_size == b_s.size(0), (
        f"b's first dimension divided by block_size must match b_s's first dimension (block_size={block_size})")
    assert b.size(1) // block_size == b_s.size(1), (
        f"b's second dimension divided by block_size must match b_s's second dimension (block_size={block_size})")
    assert grad_output.size(1) // block_size == go_s.size(1), (
        f"grad_output's second dimension divided by block_size must match go_s's second dimension (block_size={block_size})"
    )

    M, N = grad_output.size()
    _, K = b.size()
    grad_act = grad_output.new_zeros(M, K, dtype=torch_dtype)
    stride_gam, stride_gak = grad_act.stride()
    stride_gom, stride_gon = grad_output.stride()
    stride_gsm, stride_gsn = go_s.stride()
    stride_bn, stride_bk = b.stride()
    stride_bsm, stride_bsk = b_s.stride()
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(K, block_size),
    )
    wrap_triton(blockwise_fp8_gemm_backward_act_kernel)[grid](
        grad_act,
        grad_output,
        go_s,
        b,
        b_s,
        stride_gam,
        stride_gak,
        stride_gom,
        stride_gon,
        stride_gsm,
        stride_gsn,
        stride_bn,
        stride_bk,
        stride_bsm,
        stride_bsk,
        M,
        N,
        K,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_K=block_size,
    )
    return grad_act


@blockwise_fp8_gemm_backward_act.register_fake
def blockwise_fp8_gemm_backward_act_fake(
    grad_output: torch.Tensor,  # [M, N]
    go_s: torch.Tensor,  # [M, N // block_size]
    b: torch.Tensor,  # [N, K]
    b_s: torch.Tensor,  # [N // block_size, K // block_size]
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fake implementation of the block-wise FP8 GEMM backward activation for testing purposes.
    Returns a zero tensor of the appropriate shape.
    """
    assert grad_output.dim() == 2 and b.dim() == 2, "Input tensors must have 2 dimensions"
    assert b.size(0) // block_size == b_s.size(0), (
        f"b's first dimension divided by block_size must match b_s's first dimension (block_size={block_size})")
    assert b.size(1) // block_size == b_s.size(1), (
        f"b's second dimension divided by block_size must match b_s's second dimension (block_size={block_size})")
    assert grad_output.size(1) // block_size == go_s.size(1), (
        f"grad_output's second dimension divided by block_size must match go_s's second dimension (block_size={block_size})"
    )

    M, N = grad_output.size()
    _, K = b.size()
    grad_act = grad_output.new_zeros(M, K, dtype=torch_dtype)
    return grad_act


@triton.jit
def fp8_blockwise_act_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_n = pid_n * BLOCK_SIZE

    offs_x = pid_m * stride_xm + (start_n + tl.arange(0, BLOCK_SIZE)) * stride_xn

    x = tl.load(x_ptr + offs_x)
    amax = tl.clamp(tl.max(tl.abs(x)), min=1e-12, max=float("inf")).to(tl.float64)
    s = (amax / f8_fwd_max_const).to(tl.float32)
    y = tl.clamp(x / s, -f8_fwd_max_const, f8_fwd_max_const).to(y_ptr.dtype.element_ty)
    offs_y = pid_m * stride_ym + (start_n + tl.arange(0, BLOCK_SIZE)) * stride_yn
    offs_s = pid_m * stride_sm + pid_n * stride_sn
    tl.store(y_ptr + offs_y, y)
    tl.store(s_ptr + offs_s, s.to(s_ptr.dtype.element_ty))


@triton_op("torchtitan::fp8_blockwise_act_quant", mutates_args={})
def fp8_blockwise_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with block size being BLOCK_SIZEx1.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        axis (int, optional): The dimension along which to apply the quantization. Default is -1 (the last dimension).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with FP8 dtype.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.size(axis) % block_size == 0, (f"Dimension size must be divisible by block_size (block_size={block_size})")
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    dtype = get_f8_fwd_dtype()

    y = torch.zeros_like(x, dtype=dtype, memory_format=torch.contiguous_format)
    scale_shape = list(x.shape)
    scale_shape[axis] = scale_shape[axis] // block_size
    s = torch.ones(scale_shape, device=x.device, dtype=torch.float32)
    x = x.transpose(axis, -1)
    y = y.transpose(axis, -1)
    s = s.transpose(axis, -1)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()
    stride_sm, stride_sn = s.stride()
    M, N = x.shape
    num_blocks_n = N // block_size
    grid = (
        M,
        num_blocks_n,
    )
    wrap_triton(fp8_blockwise_act_quant_kernel)[grid](
        x,
        y,
        s,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )

    return y.transpose(-1, axis), s.transpose(-1, axis)


@fp8_blockwise_act_quant.register_fake
def fp8_blockwise_act_quant_fake(
    x: torch.Tensor,
    block_size: int = 128,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation of the block-wise quantization for testing purposes.
    Returns the input tensor and a dummy scaling factor tensor.
    """
    assert x.size(axis) % block_size == 0, (f"Dimension size must be divisible by block_size (block_size={block_size})")
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    dtype = get_f8_fwd_dtype()
    y = torch.zeros_like(x, dtype=dtype)
    scale_shape = list(x.shape)
    scale_shape[axis] = scale_shape[axis] // block_size
    s = torch.ones(scale_shape, device=x.device, dtype=torch.float32)
    return y, s


@triton.jit
def fp8_blockwise_act_dequant_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantizes the input tensor `x_ptr` using the scaling factors from `s_ptr` and stores the result in `y_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the quantized input tensor.
        s_ptr (triton.Pointer): Pointer to the scaling factors tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where dequantized values will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    start_n = pid_n * BLOCK_SIZE

    offs_x = pid_m * stride_xm + (start_n + tl.arange(0, BLOCK_SIZE)) * stride_xn
    x = tl.load(x_ptr + offs_x).to(tl.float32)

    s = tl.load(s_ptr + pid_m * stride_sm + pid_n * stride_sn)
    y = x * s
    y = y.to(y_ptr.dtype.element_ty)

    offs_y = pid_m * stride_ym + (start_n + tl.arange(0, BLOCK_SIZE)) * stride_yn
    tl.store(y_ptr + offs_y, y)


@triton_op("torchtitan::fp8_blockwise_act_dequant", mutates_args={})
def fp8_blockwise_act_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    axis: int = -1,
) -> torch.Tensor:
    """
    Dequantizes the input tensor `x` using the provided scaling factors `s`.

    Args:
        x (torch.Tensor): The quantized tensor to be dequantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        s (torch.Tensor): The scaling factors tensor. Must have the same shape as `x` but with the last dimension size divided by `block_size`.
        block_size (int, optional): The size of the blocks used for quantization. Default is 128.
        axis (int, optional): The dimension along which to apply the dequantization. Default is -1 (the last dimension).

    Returns:
        torch.Tensor: The dequantized tensor of the same shape as `x`.
    """
    assert x.size(axis) % block_size == 0, (f"Dimension size must be divisible by block_size (block_size={block_size})")
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    y = torch.zeros_like(x, dtype=torch_dtype, memory_format=torch.contiguous_format)

    x = x.transpose(axis, -1)
    s = s.transpose(axis, -1)
    y = y.transpose(axis, -1)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()
    stride_sm, stride_sn = s.stride()
    M, N = x.shape
    num_blocks_n = N // block_size
    grid = (
        M,
        num_blocks_n,
    )
    wrap_triton(fp8_blockwise_act_dequant_kernel)[grid](
        x,
        s,
        y,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE=block_size,
    )
    return y.transpose(-1, axis)


@fp8_blockwise_act_dequant.register_fake
def fp8_blockwise_act_dequant_fake(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    axis: int = -1,
) -> torch.Tensor:
    """
    Fake implementation of the block-wise dequantization for testing purposes.
    Returns the input tensor multiplied by the scaling factors tensor.
    """
    assert x.size(axis) % block_size == 0, (f"Dimension size must be divisible by block_size (block_size={block_size})")
    assert x.dim() == 2, "Input tensor must have 2 dimensions"

    y = torch.zeros_like(x, dtype=torch_dtype)
    return y


@triton.jit
def fp8_blockwise_weight_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_yb,
    stride_ym,
    stride_yn,
    stride_sb,
    stride_sm,
    stride_sn,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factors in `s_ptr`.

    Args:
        x_ptr (tl.pointer): Pointer to the input tensor.
        y_ptr (tl.pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (tl.pointer): Pointer to the output tensor where scaling factors will be stored.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    m = tl.cdiv(M, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_x = pid_b * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    offs_y = pid_b * stride_yb + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_x, mask=mask).to(tl.float32)
    amax = tl.clamp(tl.max(tl.abs(x)), min=1e-12, max=float("inf")).to(tl.float64)
    s = (amax / f8_fwd_max_const).to(tl.float32)
    y = tl.clamp(x / s, -f8_fwd_max_const, f8_fwd_max_const).to(y_ptr.dtype.element_ty)
    s = s.to(s_ptr.dtype.element_ty)
    tl.store(y_ptr + offs_y, y, mask=mask)
    tl.store(s_ptr + pid_b * stride_sb + pid_m * stride_sm + pid_n * stride_sn, s)


@triton_op("torchtitan::fp8_blockwise_weight_quant", mutates_args={})
def fp8_blockwise_weight_quant(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given weight tensor using block-wise quantization with block size being BLOCK_SIZExBLOCK_SIZE.

    Args:
        x (torch.Tensor): The weight tensor to be quantized.
        block_size (int, optional): The block size to use for quantization. Defaults to 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized weight tensor with dtype `dtype`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.dim() in [2, 3], "Input tensor must have 2 or 3 dimensions"
    assert x.size(-2) % block_size == 0 and x.size(-1) % block_size == 0, (
        f"Both dimensions of x ({x.shape}) must be divisible by block_size (block_size={block_size})")
    dtype = get_f8_fwd_dtype()
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    else:
        B, M, N = x.size()
    y = torch.zeros_like(x, dtype=dtype)
    s = x.new_ones(B, M // block_size, N // block_size, dtype=torch.float32)
    if x.dim() == 2:
        stride_xb = 0
        stride_xm, stride_xn = x.stride()
        stride_yb = 0
        stride_ym, stride_yn = y.stride()
    else:
        stride_xb, stride_xm, stride_xn = x.stride()
        stride_yb, stride_ym, stride_yn = y.stride()
    stride_sb, stride_sm, stride_sn = s.stride()
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    wrap_triton(fp8_blockwise_weight_quant_kernel)[grid](
        x,
        y,
        s,
        stride_xb,
        stride_xm,
        stride_xn,
        stride_yb,
        stride_ym,
        stride_yn,
        stride_sb,
        stride_sm,
        stride_sn,
        M,
        N,
        BLOCK_SIZE=block_size,
    )
    if x.dim() == 2:
        s = s.squeeze(0)
    return y, s


@fp8_blockwise_weight_quant.register_fake
def fp8_blockwise_weight_quant_fake(
    x: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fake implementation of the block-wise weight quantization for testing purposes.
    Returns the input tensor and a dummy scaling factor tensor.
    """
    assert x.dim() in [2, 3], "Input tensor must have 2 or 3 dimensions"
    assert x.size(-2) % block_size == 0 and x.size(-1) % block_size == 0, (
        f"Both dimensions of x ({x.shape}) must be divisible by block_size (block_size={block_size})")
    dtype = get_f8_fwd_dtype()
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    else:
        B, M, N = x.size()
    y = torch.zeros_like(x, dtype=dtype)
    s = x.new_ones(B, M // block_size, N // block_size, dtype=torch.float32)
    if x.dim() == 2:
        s = s.squeeze(0)
    return y, s


@triton.jit
def fp8_blockwise_weight_dequant_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    stride_xb,
    stride_xm,
    stride_xn,
    stride_sb,
    stride_sm,
    stride_sn,
    stride_yb,
    stride_ym,
    stride_yn,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_x = pid_b * stride_xb + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    offs_y = pid_b * stride_yb + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_x, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_b * stride_sb + pid_m * stride_sm + pid_n * stride_sn)
    y = x * s
    tl.store(y_ptr + offs_y, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton_op("torchtitan::fp8_blockwise_weight_dequant", mutates_args={})
def fp8_blockwise_weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert (x.dim() == 2 and s.dim() == 2) or (x.dim() == 3 and
                                               s.dim() == 3), "Input tensors must have 2 or 3 dimensions"
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    else:
        B, M, N = x.size()
    y = torch.zeros_like(x, dtype=torch_dtype)
    if x.dim() == 2:
        stride_xb = 0
        stride_xm, stride_xn = x.stride()
        stride_yb = 0
        stride_ym, stride_yn = y.stride()
        stride_sb = 0
        stride_sm, stride_sn = s.stride()
    else:
        stride_xb, stride_xm, stride_xn = x.stride()
        stride_yb, stride_ym, stride_yn = y.stride()
        stride_sb, stride_sm, stride_sn = s.stride()
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    wrap_triton(fp8_blockwise_weight_dequant_kernel)[grid](
        x,
        s,
        y,
        stride_xb,
        stride_xm,
        stride_xn,
        stride_sb,
        stride_sm,
        stride_sn,
        stride_yb,
        stride_ym,
        stride_yn,
        M,
        N,
        BLOCK_SIZE=block_size,
    )
    return y


@fp8_blockwise_weight_dequant.register_fake
def fp8_blockwise_weight_dequant_fake(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Fake implementation of the block-wise weight dequantization for testing purposes.
    Returns the input tensor multiplied by the scaling factor tensor.
    """
    assert (x.dim() == 2 and s.dim() == 2) or (x.dim() == 3 and
                                               s.dim() == 3), "Input tensors must have 2 or 3 dimensions"
    y = torch.zeros_like(x, dtype=torch_dtype)
    return y
