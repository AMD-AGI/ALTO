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
            accumulator = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=accumulator, out_dtype=tl.float32)
        else:
            accumulator = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2", acc=accumulator, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])

