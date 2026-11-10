# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

"""
from typing import Tuple, Optional
import os
import math
import torch
import triton
import triton.language as tl
from torch._library import triton_op, wrap_triton

from alto.kernels.hadamard_transform import HadamardFactory

from .mxfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    is_cdna4,
    _calculate_scales,
    _pack_fp4,
    _unpack_fp4,
)

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.float32
# Seed the RNG so we get reproducible results for testing.
philox_seed: tl.constexpr = 0x1BF52
philox_offset: tl.constexpr = 0x1D4B42

AUTOTUNE = os.environ.get('FLASH_ATTENTION_TRITON_AMD_AUTOTUNE', '0').lower() in ('1', 'true', 'yes')
DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_AMD_DEBUG', '0').lower() in ('1', 'true', 'yes')
PERF = os.environ.get('FLASH_ATTENTION_TRITON_AMD_PERF', '0').lower() in ('1', 'true', 'yes')


def get_shape_from_layout(q, k, v, layout, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):
    if layout == 'bhsd':
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
        batch_v, nheads_v, max_seqlen_v, head_size_v = v.shape
    elif layout == 'bshd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
        batch_v, max_seqlen_v, nheads_v, head_size_v = v.shape
    elif layout == 'thd':
        batch_q, max_seqlen_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        batch_k, max_seqlen_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2]
        batch_v, max_seqlen_v, nheads_v, head_size_v = len(cu_seqlens_k) - 1, max_seqlen_k, v.shape[1], v.shape[2]
    else:
        assert False, "Got unsupported layout."

    # Assume FP4 is packed along the head_dim
    head_size_q *= 2
    head_size_k *= 2
    head_size_v *= 2

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, head_size_v, max_seqlen_q, max_seqlen_k


def get_strides_from_layout(q, layout):
    if layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    elif layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    elif layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second, other=0):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=other)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=other)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=other)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _qk_dot(
    q,
    qs,
    k,
    ks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """QK product. USE_ASM selects the native scaled dot; otherwise dequantize and use tl.dot."""
    if USE_ASM:
        return tl.dot_scaled(q, qs, "e2m1", k, ks, "e2m1", lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32)
    q_deq = _unpack_fp4(
        q,
        qs,
        tl.float32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_D,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_ASM=USE_ASM,
    )
    # k is [D/2, N] with scales [N, D/32]. Unpack along the head dimension.
    k_deq = _unpack_fp4(
        tl.trans(k),
        ks,
        tl.float32,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_D,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_ASM=USE_ASM,
    )
    return tl.dot(q_deq, tl.trans(k_deq), out_dtype=tl.float32)


@triton.jit
def _pv_dot(
    p_deq,
    p_fp4,
    ps,
    v,
    vs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """PV product. p_deq is the same tile added into the row normalizer.

    USE_ASM selects the native scaled dot. The software path multiplies the
    dequantized probability by the dequantized V tile.
    """
    if USE_ASM:
        return tl.dot_scaled(
            p_fp4, ps, "e2m1", v, vs, "e2m1", lhs_k_pack=True, rhs_k_pack=False, out_dtype=tl.float32
        )
    v_deq = _dequant_v(
        v,
        vs,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        USE_ASM=USE_ASM,
    )
    return tl.dot(p_deq, v_deq, out_dtype=tl.float32)


@triton.jit
def _dequant_v(
    v,
    vs,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Dequantize V.

    ``vs`` is ``[D, N/32]`` with each group of 32 head rows repeated. The
    unpacked tile is ``[N, D]`` and consumes scales ``[N, D/32]``.
    """
    scale_groups: tl.constexpr = BLOCK_D // QUANT_BLOCK_SIZE
    k_groups: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    # Repeated rows hold the same E8M0 byte. Sum and divide instead of slicing,
    # which Triton cannot do with a constexpr index.
    compact = tl.sum(vs.to(tl.int32).reshape(scale_groups, QUANT_BLOCK_SIZE, k_groups), 1) // QUANT_BLOCK_SIZE
    compact = compact.to(tl.uint8)
    scales = tl.trans(compact).expand_dims(axis=1).broadcast_to(k_groups, QUANT_BLOCK_SIZE, scale_groups)
    scales = scales.reshape(BLOCK_N, scale_groups)
    return _unpack_fp4(
        v,
        scales,
        tl.float32,
        BLOCK_M=BLOCK_N,
        BLOCK_N=BLOCK_D,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False,
        USE_ASM=USE_ASM,
    )


@triton.jit
def _repeat_transposed_scale(
    scales,
    SRC_ROWS: tl.constexpr,
    DST_ROWS: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Turn ``[SRC_ROWS, DST_ROWS/32]`` into ``[DST_ROWS, SRC_ROWS/32]``.

    Each group of 32 source rows repeats one E8M0 byte. Compact that byte, then
    repeat it along the destination rows. This is the scale layout ``tl.dot_scaled``
    expects when the reduction axis is not the packed axis.
    """
    src_groups: tl.constexpr = SRC_ROWS // QUANT_BLOCK_SIZE
    dst_groups: tl.constexpr = DST_ROWS // QUANT_BLOCK_SIZE
    compact = tl.sum(scales.to(tl.int32).reshape(src_groups, QUANT_BLOCK_SIZE, dst_groups), 1) // QUANT_BLOCK_SIZE
    compact = compact.to(tl.uint8)
    out = tl.trans(compact).expand_dims(axis=1).broadcast_to(dst_groups, QUANT_BLOCK_SIZE, src_groups)
    return out.reshape(DST_ROWS, src_groups)


@triton.jit
def _qdq_tile(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Quantize a tile along its last axis and dequantize it back to fp32."""
    scales = _calculate_scales(
        x, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False)
    packed = _pack_fp4(
        x, scales, None, None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False,
        USE_SR=False, USE_ASM=USE_ASM)
    return _unpack_fp4(
        packed, scales, tl.float32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False, USE_ASM=USE_ASM)


@triton.jit
def _dp_dot(
    do,
    do_deq,
    v,
    vs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """dP = dO @ V^T. ``do_deq`` is the MXFP4 round-trip of ``do`` and is what delta uses."""
    if USE_ASM:
        do_s = _calculate_scales(do, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False)
        do_fp4 = _pack_fp4(
            do, do_s, None, None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False,
            USE_SR=False, USE_ASM=USE_ASM)
        v_ks = _repeat_transposed_scale(vs, BLOCK_D, BLOCK_N, QUANT_BLOCK_SIZE)
        return tl.dot_scaled(
            do_fp4, do_s, "e2m1", tl.trans(v), v_ks, "e2m1", lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32)
    v_deq = _dequant_v(v, vs, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, USE_ASM=USE_ASM)
    return tl.dot(do_deq, tl.trans(v_deq), out_dtype=tl.float32)


@triton.jit
def _dq_dot(
    ds,
    k,
    ks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """dQ = dS @ K. USE_ASM quantizes dS along N and consumes packed K."""
    if USE_ASM:
        ds_s = _calculate_scales(ds, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False)
        ds_fp4 = _pack_fp4(
            ds, ds_s, None, None, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False,
            USE_SR=False, USE_ASM=USE_ASM)
        ks_vs = _repeat_transposed_scale(ks, BLOCK_N, BLOCK_D, QUANT_BLOCK_SIZE)
        return tl.dot_scaled(
            ds_fp4, ds_s, "e2m1", tl.trans(k), ks_vs, "e2m1", lhs_k_pack=True, rhs_k_pack=False, out_dtype=tl.float32)
    k_deq = _unpack_fp4(
        tl.trans(k), ks, tl.float32, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_D, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=False, USE_ASM=USE_ASM)
    return tl.dot(ds, k_deq, out_dtype=tl.float32)


@triton.jit
def _dk_dot(
    ds,
    q,
    qs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """dK = dS^T @ Q. USE_ASM quantizes dS along M and consumes packed Q."""
    if USE_ASM:
        dst = tl.trans(ds)
        dst_s = _calculate_scales(
            dst, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False)
        dst_fp4 = _pack_fp4(
            dst, dst_s, None, None, BLOCK_M=BLOCK_N, BLOCK_N=BLOCK_M, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            IS_2D_BLOCK=False, USE_SR=False, USE_ASM=USE_ASM)
        qs_vs = _repeat_transposed_scale(qs, BLOCK_M, BLOCK_D, QUANT_BLOCK_SIZE)
        return tl.dot_scaled(
            dst_fp4, dst_s, "e2m1", q, qs_vs, "e2m1", lhs_k_pack=True, rhs_k_pack=False, out_dtype=tl.float32)
    q_deq = _unpack_fp4(
        q, qs, tl.float32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D, QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE, IS_2D_BLOCK=False,
        USE_ASM=USE_ASM)
    return tl.dot(tl.trans(ds), q_deq, out_dtype=tl.float32)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    qs,
    k_ptrs,
    v_ptrs,
    ks_ptrs,
    vs_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    stride_kscale_n,
    stride_vscale_k,
    start_m,
    actual_seqlen_k,
    actual_seqlen_k_scale,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    exp_scores_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    score_ptrs,
    scores_scaled_shifted_ptrs,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HALF_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_BLOCK_DMODEL_QK: tl.constexpr,
    HALF_BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_SCALE: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    HALF_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    HALF_ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
            vs_offs_n = start_n // QUANT_BLOCK_SIZE + tl.arange(0, BLOCK_N_SCALE)
        else:
            k_offs_n = None
            vs_offs_n = None
        if PADDED_HEAD_QK:
            k_offs_k = tl.arange(0, HALF_BLOCK_DMODEL_QK)
            # k_offs_k = tl.arange(0, HALF_BLOCK_DMODEL_QK * 2)
            ks_offs_k = tl.arange(0, SCALE_BLOCK_DMODEL_QK)
        else:
            k_offs_k = None
            ks_offs_k = None
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, HALF_ACTUAL_BLOCK_DMODEL_QK, actual_seqlen_k)
        ks = load_fn(ks_ptrs, k_offs_n, ks_offs_k, actual_seqlen_k, SCALE_ACTUAL_BLOCK_DMODEL_QK, other=1)

        if PADDED_HEAD_V:
            v_offs_k = tl.arange(0, HALF_BLOCK_DMODEL_V)
            vs_offs_k = tl.arange(0, BLOCK_DMODEL_V)
        else:
            v_offs_k = None
            vs_offs_k = None
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, HALF_ACTUAL_BLOCK_DMODEL_V)
            vs = load_fn(vs_ptrs, vs_offs_k, vs_offs_n, ACTUAL_BLOCK_DMODEL_V, actual_seqlen_k_scale, other=1)
        # -- compute qk ----
        qk = _qk_dot(
            q,
            qs,
            k,
            ks,
            BLOCK_M,
            BLOCK_N,
            HALF_BLOCK_DMODEL_QK * 2,
            QUANT_BLOCK_SIZE,
            USE_ASM,
        )

        qk_scaled = qk * SM_SCALE
        if MASK_STEPS:
            # Mask after the dot. Adding the product onto a pre-filled -inf tile
            # does not survive the dot, so out-of-range columns would stay finite.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                qk_scaled = tl.where(size_n < boundary_m[:, None], qk_scaled, float("-inf"))

        if RETURN_SCORES:
            score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(score_ptrs, qk_scaled, mask=score_mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            qk_scaled += bias

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
        # Online softmax. The row sum uses the dequantized numerator, and that
        # same tile multiplies V, so the weights induced by MXFP4 still sum to one.
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))
        q_shifted = qk_scaled - m_ij[:, None]
        if RETURN_SCORES:
            scores_scaled_shifted_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(scores_scaled_shifted_ptrs, q_shifted, mask=scores_scaled_shifted_mask)

        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)
        p = tl.where(qk_scaled == float("-inf"), 0.0, p)

        ps = _calculate_scales(
            p,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            IS_2D_BLOCK=False,
        )
        p_fp4 = _pack_fp4(
            p,
            ps,
            None,
            None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            IS_2D_BLOCK=False,
            USE_SR=False,
            USE_ASM=USE_ASM,
        )
        p_deq = _unpack_fp4(
            p_fp4,
            ps,
            tl.float32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            IS_2D_BLOCK=False,
            USE_ASM=USE_ASM,
        )
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_SCORES:
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                    (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
                tl.store(exp_scores_ptrs, tl.where(keep, p_deq, -p_deq), mask=exp_score_mask)
            p_deq = tl.where(keep, p_deq, 0.0)
        elif RETURN_SCORES:
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(exp_scores_ptrs, p_deq, mask=exp_score_mask)

        l_ij = tl.sum(p_deq, 1)
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, HALF_ACTUAL_BLOCK_DMODEL_V)
            vs = load_fn(vs_ptrs, vs_offs_k, vs_offs_n, ACTUAL_BLOCK_DMODEL_V, actual_seqlen_k_scale, other=1)
        acc += _pv_dot(
            p_deq,
            p_fp4,
            ps,
            v,
            vs,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_V,
            QUANT_BLOCK_SIZE,
            USE_ASM,
        )

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        ks_ptrs += BLOCK_N_SCALE * stride_kscale_n
        vs_ptrs += BLOCK_N_SCALE * stride_vscale_k
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += BLOCK_N
            scores_scaled_shifted_ptrs += BLOCK_N
            exp_scores_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_autotune_fwd_configs():
    return [
        triton.Config(
            {
                "PRE_LOAD_V": False,
            },
            num_stages=1,
            num_warps=4,
        ),
    ], [
        "IS_CAUSAL", "dropout_p", "MAX_SEQLENS_Q", "MAX_SEQLENS_K", "ACTUAL_BLOCK_DMODEL_QK", "ACTUAL_BLOCK_DMODEL_V",
        "VARLEN", "HQ", "HK"
    ]


autotune_fwd_configs, autotune_fwd_keys = get_autotune_fwd_configs()


@triton.jit
def _attn_fwd_blocks(
    acc,
    l_i,
    m_i,
    q,
    qs,
    k_ptrs,
    v_ptrs,
    ks_ptrs,
    vs_ptrs,
    bias_ptrs,
    score_ptrs,
    scores_scaled_shifted_ptrs,
    exp_scores_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    stride_kscale_n,
    stride_vscale_k,
    start_m,
    seqlen_k,
    seqlen_k_scale,
    seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    alibi_slope,
    offs_m,
    offs_n,
    n_full_blocks,
    n_blocks,
    n_extra_tokens,
    masked_blocks,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HALF_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_BLOCK_DMODEL_QK: tl.constexpr,
    HALF_BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_SCALE: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    HALF_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    HALF_ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_BIAS: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    block_min = 0
    block_max = n_blocks * BLOCK_N
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            qs,
            k_ptrs,
            v_ptrs,
            ks_ptrs,
            vs_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_kscale_n,
            stride_vscale_k,
            start_m,
            seqlen_k,
            seqlen_k_scale,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            block_min,
            block_max,
            0,
            0,
            0,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            False,
            BLOCK_M,
            HALF_BLOCK_DMODEL_QK,
            SCALE_BLOCK_DMODEL_QK,
            HALF_BLOCK_DMODEL_V,
            BLOCK_DMODEL_V,
            BLOCK_N,
            BLOCK_N_SCALE,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            HALF_ACTUAL_BLOCK_DMODEL_QK,
            SCALE_ACTUAL_BLOCK_DMODEL_QK,
            HALF_ACTUAL_BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            USE_ASM=USE_ASM,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        ks_ptrs += n_full_blocks * BLOCK_N_SCALE * stride_kscale_n
        vs_ptrs += n_full_blocks * BLOCK_N_SCALE * stride_vscale_k
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += n_full_blocks * BLOCK_N
            scores_scaled_shifted_ptrs += n_full_blocks * BLOCK_N
            exp_scores_ptrs += n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            qs,
            k_ptrs,
            v_ptrs,
            ks_ptrs,
            vs_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_kscale_n,
            stride_vscale_k,
            start_m,
            seqlen_k,
            seqlen_k_scale,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            IS_CAUSAL,
            BLOCK_M,
            HALF_BLOCK_DMODEL_QK,
            SCALE_BLOCK_DMODEL_QK,
            HALF_BLOCK_DMODEL_V,
            BLOCK_DMODEL_V,
            BLOCK_N,
            BLOCK_N_SCALE,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            HALF_ACTUAL_BLOCK_DMODEL_QK,
            SCALE_ACTUAL_BLOCK_DMODEL_QK,
            HALF_ACTUAL_BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            USE_ASM=USE_ASM,
        )
    return acc, l_i, m_i


@triton.autotune(
    configs=autotune_fwd_configs,
    key=autotune_fwd_keys,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    SM_SCALE: tl.constexpr,
    LSE,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    stride_qscale_z,
    stride_qscale_h,
    stride_qscale_m,
    stride_qscale_k,
    stride_kscale_z,
    stride_kscale_h,
    stride_kscale_n,
    stride_kscale_k,
    stride_vscale_z,
    stride_vscale_h,
    stride_vscale_k,
    stride_vscale_n,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    scores,
    scores_scaled_shifted,
    exp_scores,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    HALF_BLOCK_DMODEL_QK: tl.constexpr = BLOCK_DMODEL_QK // 2
    HALF_BLOCK_DMODEL_V: tl.constexpr = BLOCK_DMODEL_V // 2
    SCALE_BLOCK_DMODEL_QK: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    HALF_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK // 2
    HALF_ACTUAL_BLOCK_DMODEL_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V // 2
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    BLOCK_N_SCALE: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_d_qk_pack = tl.arange(0, HALF_BLOCK_DMODEL_QK)
    offs_d_qk_scale = tl.arange(0, SCALE_BLOCK_DMODEL_QK)
    offs_d_v_pack = tl.arange(0, HALF_BLOCK_DMODEL_V)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_n_scale = tl.arange(0, BLOCK_N_SCALE)

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    PADDED_HEAD_QK: tl.constexpr = (ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK)
    PADDED_HEAD_V: tl.constexpr = (ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        # print("cu_seqlens_q_start:", cu_seqlens_q_start)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    cu_seqlens_q_start_scale = cu_seqlens_q_start // QUANT_BLOCK_SIZE
    cu_seqlens_k_start_scale = cu_seqlens_k_start // QUANT_BLOCK_SIZE
    seqlen_k_scale = seqlen_k // QUANT_BLOCK_SIZE

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            if PADDED_HEAD_V:
                o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
            l_ptrs = l_offset + offs_m * stride_lse_m

            l = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)

            # mask_m_offsets = start_m + tl.arange(0, BLOCK_M)
            # lse_mask = mask_m_offsets < causal_start_idx
            # softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk_pack[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d_qk_pack[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v_pack[None, :] * stride_vn
    qs_offset = (q_scale_ptr + off_z * stride_qscale_z + off_h_q * stride_qscale_h +
                 cu_seqlens_q_start_scale * stride_qscale_m)
    qs_ptrs = (qs_offset + (offs_m[:, None] // QUANT_BLOCK_SIZE) * stride_qscale_m +
               offs_d_qk_scale[None, :] * stride_qscale_k)
    ks_offset = (k_scale_ptr + off_z * stride_kscale_z + off_h_k * stride_kscale_h +
                 cu_seqlens_k_start_scale * stride_kscale_n)
    # k scale is N*K even though k is K*N, this is required by tl.dot_scaled
    ks_ptrs = (ks_offset + (offs_n[:, None] // QUANT_BLOCK_SIZE) * stride_kscale_n +
               offs_d_qk_scale[None, :] * stride_kscale_k)
    vs_offset = (v_scale_ptr + off_z * stride_vscale_z + off_h_k * stride_vscale_h +
                 cu_seqlens_k_start_scale * stride_vscale_k)
    vs_ptrs = (vs_offset + (offs_d_v[:, None] // QUANT_BLOCK_SIZE) * stride_vscale_n +
               offs_n_scale[None, :] * stride_vscale_k)
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_z * stride_bz + off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        scores_offset = scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        score_ptrs = scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn

        scores_scaled_shifted_offset = scores_scaled_shifted + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        scores_scaled_shifted_ptrs = scores_scaled_shifted_offset + offs_m[:, None] * stride_sm + offs_n[
            None, :] * stride_sn

        exp_scores_offset = exp_scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        exp_scores_ptrs = exp_scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        score_ptrs = None
        scores_scaled_shifted_ptrs = None
        exp_scores_ptrs = None

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    qs_ptrs_mask = q_ptrs_mask
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk_pack[None, :] < HALF_ACTUAL_BLOCK_DMODEL_QK)
        qs_ptrs_mask = qs_ptrs_mask & (offs_d_qk_scale[None, :] < SCALE_ACTUAL_BLOCK_DMODEL_QK)

    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0)
    qs = tl.load(qs_ptrs, mask=qs_ptrs_mask, other=1)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    acc, l_i, m_i = _attn_fwd_blocks(
        acc,
        l_i,
        m_i,
        q,
        qs,
        k_ptrs,
        v_ptrs,
        ks_ptrs,
        vs_ptrs,
        bias_ptrs,
        score_ptrs,
        scores_scaled_shifted_ptrs,
        exp_scores_ptrs,
        stride_kn,
        stride_vk,
        stride_bn,
        stride_kscale_n,
        stride_vscale_k,
        start_m,
        seqlen_k,
        seqlen_k_scale,
        seqlen_q,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        alibi_slope,
        offs_m,
        offs_n,
        n_full_blocks,
        n_blocks,
        n_extra_tokens,
        masked_blocks,
        IS_CAUSAL,
        BLOCK_M,
        HALF_BLOCK_DMODEL_QK,
        SCALE_BLOCK_DMODEL_QK,
        HALF_BLOCK_DMODEL_V,
        BLOCK_DMODEL_V,
        BLOCK_N,
        BLOCK_N_SCALE,
        PRE_LOAD_V,
        ENABLE_DROPOUT,
        PADDED_HEAD_QK,
        PADDED_HEAD_V,
        HALF_ACTUAL_BLOCK_DMODEL_QK,
        SCALE_ACTUAL_BLOCK_DMODEL_QK,
        HALF_ACTUAL_BLOCK_DMODEL_V,
        ACTUAL_BLOCK_DMODEL_V,
        SM_SCALE,
        USE_EXP2,
        RETURN_SCORES,
        USE_BIAS,
        QUANT_BLOCK_SIZE,
        USE_ASM,
    )
    # epilogue
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k

    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL_V,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.dtype))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    offs_l_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = l_offset + offs_l_m * stride_lse_m
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL:
        # zero out nans caused by -infs when doing causal
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, softmax_lse)

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_ptrs_mask)


@triton.jit
def _load_q_dequant(
    q_base,
    qs_base,
    offs_m,
    stride_qm,
    stride_qk,
    stride_qscale_m,
    stride_qscale_k,
    seqlen_q,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    ACTUAL_D: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    offs_pack = tl.arange(0, BLOCK_D // 2)
    offs_scale = tl.arange(0, BLOCK_D // QUANT_BLOCK_SIZE)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_pack[None, :] * stride_qk,
        mask=(offs_m[:, None] < seqlen_q) & (offs_pack[None, :] < ACTUAL_D // 2),
        other=0,
    )
    qs = tl.load(
        qs_base + (offs_m[:, None] // QUANT_BLOCK_SIZE) * stride_qscale_m + offs_scale[None, :] * stride_qscale_k,
        mask=(offs_m[:, None] < seqlen_q) & (offs_scale[None, :] < ACTUAL_D // QUANT_BLOCK_SIZE),
        other=1,
    )
    return q, qs


@triton.jit
def _load_kv_dequant(
    k_base,
    v_base,
    ks_base,
    vs_base,
    stride_kn,
    stride_kk,
    stride_vk,
    stride_vn,
    stride_kscale_n,
    stride_kscale_k,
    stride_vscale_k,
    stride_vscale_n,
    start_n,
    seqlen_k,
    BLOCK_N: tl.constexpr,
    BLOCK_D_QK: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    ACTUAL_D_QK: tl.constexpr,
    ACTUAL_D_V: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_pack = tl.arange(0, BLOCK_D_QK // 2)
    offs_scale = tl.arange(0, BLOCK_D_QK // QUANT_BLOCK_SIZE)
    k = tl.load(
        k_base + offs_pack[:, None] * stride_kk + offs_n[None, :] * stride_kn,
        mask=(offs_pack[:, None] < ACTUAL_D_QK // 2) & (offs_n[None, :] < seqlen_k),
        other=0,
    )
    ks = tl.load(
        ks_base + (offs_n[:, None] // QUANT_BLOCK_SIZE) * stride_kscale_n + offs_scale[None, :] * stride_kscale_k,
        mask=(offs_n[:, None] < seqlen_k) & (offs_scale[None, :] < ACTUAL_D_QK // QUANT_BLOCK_SIZE),
        other=1,
    )
    offs_v_pack = tl.arange(0, BLOCK_D_V // 2)
    offs_v = tl.arange(0, BLOCK_D_V)
    offs_ns = start_n // QUANT_BLOCK_SIZE + tl.arange(0, BLOCK_N // QUANT_BLOCK_SIZE)
    v = tl.load(
        v_base + offs_n[:, None] * stride_vk + offs_v_pack[None, :] * stride_vn,
        mask=(offs_n[:, None] < seqlen_k) & (offs_v_pack[None, :] < ACTUAL_D_V // 2),
        other=0,
    )
    vs = tl.load(
        vs_base + (offs_v[:, None] // QUANT_BLOCK_SIZE) * stride_vscale_n + offs_ns[None, :] * stride_vscale_k,
        mask=(offs_v[:, None] < ACTUAL_D_V) & (offs_ns[None, :] < seqlen_k // QUANT_BLOCK_SIZE),
        other=1,
    )
    return k, ks, v, vs


@triton.jit
def _masked_scores(
    q,
    qs,
    k,
    ks,
    offs_m,
    offs_n,
    seqlen_q,
    seqlen_k,
    sm_scale,
    bias,
    alibi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    qk = _qk_dot(q, qs, k, ks, BLOCK_M, BLOCK_N, BLOCK_D, QUANT_BLOCK_SIZE, USE_ASM)
    qk *= sm_scale
    valid = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
    if CAUSAL:
        valid = valid & (offs_m[:, None] >= (offs_n[None, :] + seqlen_q - seqlen_k))
    if USE_BIAS:
        qk += bias
    if USE_ALIBI:
        qk += alibi
    return tl.where(valid, qk, float("-inf"))


@triton.jit
def _softmax_prob(
    qk_scaled,
    lse,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    philox_seed,
    batch_philox_offset,
    dropout_p,
    block_m,
    start_n,
    seqlen_k,
):
    """P = exp(score - lse), the same reconstruction the FP8 backward uses."""
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        p = tl.math.exp2((qk_scaled - lse[:, None]) * RCP_LN2)
    else:
        p = tl.math.exp(qk_scaled - lse[:, None])
    p = tl.where(qk_scaled == float("-inf"), 0.0, p)
    if ENABLE_DROPOUT:
        philox_off = batch_philox_offset + block_m * BLOCK_M * seqlen_k + start_n - BLOCK_N
        keep = dropout_mask(philox_seed, philox_off, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
        p = tl.where(keep, p, 0.0)
    return p


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    q_scale,
    k_scale,
    v_scale,
    DO,
    Out,
    DQ,
    DK,
    DV,
    LSE,
    bias,
    alibi_slopes,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    stride_qscale_z,
    stride_qscale_h,
    stride_qscale_m,
    stride_qscale_k,
    stride_kscale_z,
    stride_kscale_h,
    stride_kscale_n,
    stride_kscale_k,
    stride_vscale_z,
    stride_vscale_h,
    stride_vscale_k,
    stride_vscale_n,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sm_scale,
    max_seqlen_q,
    max_seqlen_k,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_QK: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    ACTUAL_D_QK: tl.constexpr,
    ACTUAL_D_V: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    VARLEN: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    off_h_k = tl.program_id(1)
    off_z = tl.program_id(2)
    GROUP_SIZE: tl.constexpr = HQ // HK

    if VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        seqlen_q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        k_start = tl.load(cu_seqlens_k + off_z)
        seqlen_k = tl.load(cu_seqlens_k + off_z + 1) - k_start
    else:
        q_start = 0
        k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
    if start_n >= seqlen_k:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_dv = tl.arange(0, BLOCK_D_V)
    offs_dk = tl.arange(0, BLOCK_D_QK)
    offs_dq = tl.arange(0, BLOCK_D_QK)
    k_base = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    ks_base = (k_scale + off_z * stride_kscale_z + off_h_k * stride_kscale_h +
               (k_start // QUANT_BLOCK_SIZE) * stride_kscale_n)
    v_base = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vk
    vs_base = (v_scale + off_z * stride_vscale_z + off_h_k * stride_vscale_h +
               (k_start // QUANT_BLOCK_SIZE) * stride_vscale_k)
    dk = tl.zeros([BLOCK_N, BLOCK_D_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    # One K/V tile is resident. Each query block contributes one atomic update to dQ.
    for group in range(GROUP_SIZE):
        off_h_q = off_h_k * GROUP_SIZE + group
        q_base = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
        qs_base = (q_scale + off_z * stride_qscale_z + off_h_q * stride_qscale_h +
                   (q_start // QUANT_BLOCK_SIZE) * stride_qscale_m)
        do_base = DO + off_z * stride_doz + off_h_q * stride_doh + q_start * stride_dom
        o_base = Out + off_z * stride_oz + off_h_q * stride_oh + q_start * stride_om
        lse_base = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + q_start * stride_lse_m
        if USE_BIAS:
            bias_base = bias + off_z * stride_bz + off_h_q * stride_bh
        if USE_ALIBI:
            alibi_slope = tl.load(alibi_slopes + off_z * stride_az + off_h_q * stride_ah)
        else:
            alibi_slope = 0.0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + (off_z * HQ + off_h_q) * seqlen_q * seqlen_k
        else:
            batch_philox_offset = 0
        for start_m in range(0, seqlen_q, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            k, ks, v, vs = _load_kv_dequant(
                k_base, v_base, ks_base, vs_base, stride_kn, stride_kk, stride_vk, stride_vn, stride_kscale_n,
                stride_kscale_k, stride_vscale_k, stride_vscale_n, start_n, seqlen_k, BLOCK_N, BLOCK_D_QK, BLOCK_D_V,
                QUANT_BLOCK_SIZE, ACTUAL_D_QK, ACTUAL_D_V, USE_ASM)
            q, qs = _load_q_dequant(
                q_base, qs_base, offs_m, stride_qm, stride_qk, stride_qscale_m, stride_qscale_k, seqlen_q, BLOCK_M,
                BLOCK_D_QK, QUANT_BLOCK_SIZE, ACTUAL_D_QK, USE_ASM)
            tile_mask = (offs_m[:, None] < seqlen_q) & (offs_dv[None, :] < ACTUAL_D_V)
            do = tl.load(
                do_base + offs_m[:, None] * stride_dom + offs_dv[None, :] * stride_dok, mask=tile_mask, other=0.0).to(
                    tl.float32)
            o = tl.load(
                o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od, mask=tile_mask, other=0.0).to(
                    tl.float32)
            if ENABLE_DROPOUT:
                do = do / (1.0 - dropout_p)
            do_deq = _qdq_tile(do, BLOCK_M, BLOCK_D_V, QUANT_BLOCK_SIZE, USE_ASM)
            row_mask = offs_m < seqlen_q
            lse = tl.load(lse_base + offs_m * stride_lse_m, mask=row_mask, other=0.0)
            row_delta = tl.sum(o * do_deq, 1)
            if USE_BIAS:
                bias_tile = tl.load(
                    bias_base + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
                    mask=(offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                    other=0.0,
                ).to(tl.float32)
            else:
                bias_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            if USE_ALIBI:
                alibi_tile = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n)
            else:
                alibi_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk_scaled = _masked_scores(
                q, qs, k, ks, offs_m, offs_n, seqlen_q, seqlen_k, sm_scale, bias_tile, alibi_tile, BLOCK_M, BLOCK_N,
                BLOCK_D_QK, QUANT_BLOCK_SIZE, CAUSAL, USE_BIAS, USE_ALIBI, USE_ASM)
            prob = _softmax_prob(qk_scaled, lse, BLOCK_M, BLOCK_N, USE_EXP2, ENABLE_DROPOUT, philox_seed,
                                 batch_philox_offset, dropout_p, start_m // BLOCK_M, start_n, seqlen_k)
            dp = _dp_dot(do, do_deq, v, vs, BLOCK_M, BLOCK_N, BLOCK_D_V, QUANT_BLOCK_SIZE, USE_ASM)
            ds = prob * (dp - row_delta[:, None]) * sm_scale
            dq_tile = _dq_dot(ds, k, ks, BLOCK_M, BLOCK_N, BLOCK_D_QK, QUANT_BLOCK_SIZE, USE_ASM)
            dk = dk + _dk_dot(ds, q, qs, BLOCK_M, BLOCK_N, BLOCK_D_QK, QUANT_BLOCK_SIZE, USE_ASM)
            dv = dv + tl.dot(tl.trans(prob), do_deq, out_dtype=tl.float32)
            dq_ptrs = (DQ + off_z * stride_dqz + off_h_q * stride_dqh + q_start * stride_dqm +
                       offs_m[:, None] * stride_dqm + offs_dq[None, :] * stride_dqk)
            tl.atomic_add(dq_ptrs, dq_tile, mask=row_mask[:, None] & (offs_dq[None, :] < ACTUAL_D_QK))

    col_mask = offs_n < seqlen_k
    tl.store(
        DK + off_z * stride_dkz + off_h_k * stride_dkh + k_start * stride_dkn + offs_n[:, None] * stride_dkn +
        offs_dk[None, :] * stride_dkk,
        dk,
        mask=col_mask[:, None] & (offs_dk[None, :] < ACTUAL_D_QK),
    )
    tl.store(
        DV + off_z * stride_dvz + off_h_k * stride_dvh + k_start * stride_dvn + offs_n[:, None] * stride_dvn +
        offs_dv[None, :] * stride_dvd,
        dv,
        mask=col_mask[:, None] & (offs_dv[None, :] < ACTUAL_D_V),
    )
    return


def _unpacked_zeros(packed: torch.Tensor, unpacked_head: int) -> torch.Tensor:
    shape = list(packed.shape)
    shape[-1] = unpacked_head
    return torch.zeros(shape, device=packed.device, dtype=torch.float32)


def attention_mxfp4_backward(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_lse: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    bias: torch.Tensor | None,
    sm_scale: float,
    dropout_p: float,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool,
    use_exp2: bool,
    layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_varlen = layout == "thd"
    batch, nheads_q, nheads_k, head_size_qk, head_size_v, _, _ = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    dq = _unpacked_zeros(q, head_size_qk)
    dk = _unpacked_zeros(k, head_size_qk)
    dv = _unpacked_zeros(v, head_size_v)
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    do_strides = get_strides_from_layout(grad_output, layout)
    dq_strides = get_strides_from_layout(dq, layout)
    dk_strides = get_strides_from_layout(dk, layout)
    dv_strides = get_strides_from_layout(dv, layout)
    qs_strides = get_strides_from_layout(q_scale, layout)
    ks_strides = get_strides_from_layout(k_scale, layout)
    vs_strides = get_strides_from_layout(v_scale, layout)
    if is_varlen:
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()
    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)
    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    common = dict(
        stride_qz=q_strides[0],
        stride_qh=q_strides[1],
        stride_qm=q_strides[2],
        stride_qk=q_strides[3],
        stride_kz=k_strides[0],
        stride_kh=k_strides[1],
        stride_kn=k_strides[2],
        stride_kk=k_strides[3],
        stride_vz=v_strides[0],
        stride_vh=v_strides[1],
        stride_vk=v_strides[2],
        stride_vn=v_strides[3],
        stride_doz=do_strides[0],
        stride_doh=do_strides[1],
        stride_dom=do_strides[2],
        stride_dok=do_strides[3],
        stride_lse_z=stride_lse_z,
        stride_lse_h=stride_lse_h,
        stride_lse_m=stride_lse_m,
        stride_qscale_z=qs_strides[0],
        stride_qscale_h=qs_strides[1],
        stride_qscale_m=qs_strides[2],
        stride_qscale_k=qs_strides[3],
        stride_kscale_z=ks_strides[0],
        stride_kscale_h=ks_strides[1],
        stride_kscale_n=ks_strides[2],
        stride_kscale_k=ks_strides[3],
        stride_vscale_z=vs_strides[0],
        stride_vscale_h=vs_strides[1],
        stride_vscale_k=vs_strides[2],
        stride_vscale_n=vs_strides[3],
        stride_bz=bias_strides[0],
        stride_bh=bias_strides[1],
        stride_bm=bias_strides[2],
        stride_bn=bias_strides[3],
        stride_az=alibi_strides[0],
        stride_ah=alibi_strides[1],
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        sm_scale=sm_scale,
        max_seqlen_q=max_seqlens_q,
        max_seqlen_k=max_seqlens_k,
        HQ=nheads_q,
        HK=nheads_k,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D_QK=get_padded_head_dim(head_size_qk),
        BLOCK_D_V=get_padded_head_dim(head_size_v),
        ACTUAL_D_QK=head_size_qk,
        ACTUAL_D_V=head_size_v,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        CAUSAL=causal,
        VARLEN=is_varlen,
        USE_BIAS=bias is not None,
        USE_ALIBI=alibi_slopes is not None,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        USE_ASM=is_cdna4(),
        num_warps=4,
        num_stages=1,
    )
    o_strides = get_strides_from_layout(out, layout)
    wrap_triton(_attn_bwd)[(triton.cdiv(max_seqlens_k, 64), nheads_k, batch)](
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        grad_output,
        out,
        dq,
        dk,
        dv,
        softmax_lse,
        bias,
        alibi_slopes,
        stride_oz=o_strides[0],
        stride_oh=o_strides[1],
        stride_om=o_strides[2],
        stride_od=o_strides[3],
        stride_dqz=dq_strides[0],
        stride_dqh=dq_strides[1],
        stride_dqm=dq_strides[2],
        stride_dqk=dq_strides[3],
        stride_dkz=dk_strides[0],
        stride_dkh=dk_strides[1],
        stride_dkn=dk_strides[2],
        stride_dkk=dk_strides[3],
        stride_dvz=dv_strides[0],
        stride_dvh=dv_strides[1],
        stride_dvn=dv_strides[2],
        stride_dvd=dv_strides[3],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        **common,
    )
    return dq, dk, dv


def get_padded_head_dim(head_size: int):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


@triton_op("torchtitan::attention_mxfp4_forward_triton_impl", mutates_args=())
def attention_mxfp4_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    bias: Optional[torch.Tensor],
    dropout_p: float,
    layout: str,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlens_q: Optional[int],
    max_seqlens_k: Optional[int],
    return_scores: bool,
    use_exp2: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if DEBUG:
        print()
        print("attention_mxfp4_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()
    assert v_scale.is_contiguous()

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    o_shape = (*q.shape[:-1], head_size_v)
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype,
        requires_grad=True,
    )

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    grid = lambda META: (triton.cdiv(max_seqlens_q, META["BLOCK_M"]), nheads_q, batch)

    if return_scores:
        scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
        scores_scaled_shifted = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k),
                                            device=q.device,
                                            dtype=torch.float32)
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q), device=q.device, dtype=torch.float32)
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    qs_strides = get_strides_from_layout(q_scale, layout)
    ks_strides = get_strides_from_layout(k_scale, layout)
    vs_strides = get_strides_from_layout(v_scale, layout)

    wrap_triton(attn_fwd)[grid](
        q,
        k,
        v,
        bias,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        *qs_strides,
        *ks_strides,
        *vs_strides,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=False if alibi_slopes is None else True,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=64,
        BLOCK_N=64,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        USE_ASM=is_cdna4(),
    )
    return o, softmax_lse, exp_scores


@torch.compiler.allow_in_graph
class _triton_attention_mxfp4(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alibi_slopes: torch.Tensor | None,
        bias: torch.Tensor | None,
        sm_scale: float,
        dropout_p: float,
        cu_seqlens_q: int,
        cu_seqlens_k: int,
        max_seqlens_q: int,
        max_seqlens_k: int,
        causal: bool,
        return_scores: bool,
        use_exp2: bool,
        layout: str,
    ):
        input_dtype = q.dtype
        # H H^T = I, so the rotation cancels in QK^T and stays in the Q/K gradients.
        # The transform object is saved on ctx; a later forward builds a new one.
        with torch.no_grad():
            qk_hadamard = HadamardFactory.create_transform(q.device)
        q = qk_hadamard(q)
        k = qk_hadamard(k)
        q, q_scale = torch.ops.torchtitan.convert_to_mxfp4(q, axis=-1, is_2d_block=True)
        k, k_scale = torch.ops.torchtitan.convert_to_mxfp4(k, axis=-1, is_2d_block=True)
        v, v_scale = torch.ops.torchtitan.convert_to_mxfp4(v, axis=-1, is_2d_block=True)

        output, softmax_lse, exp_scores = torch.ops.torchtitan.attention_mxfp4_forward_triton_impl(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            sm_scale=sm_scale,
            alibi_slopes=alibi_slopes,
            causal=causal,
            bias=bias,
            dropout_p=dropout_p,
            layout=layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlens_q=max_seqlens_q,
            max_seqlens_k=max_seqlens_k,
            return_scores=return_scores,
            use_exp2=use_exp2,
        )

        ctx.save_for_backward(q, k, v, output, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale)
        ctx.qk_hadamard = qk_hadamard
        _triton_attention_mxfp4.last_qk_hadamard = qk_hadamard
        ctx.input_dtype = input_dtype
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.layout = layout
        ctx.use_exp2 = use_exp2
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlens_q = max_seqlens_q
        ctx.max_seqlens_k = max_seqlens_k

        return output, softmax_lse, exp_scores

    @staticmethod
    def backward(ctx, grad_output, grad_lse, grad_scores):
        del grad_lse, grad_scores
        if grad_output is None or not any(ctx.needs_input_grad[:3]):
            return (None,) * 15
        q, k, v, output, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale = ctx.saved_tensors
        dq, dk, dv = attention_mxfp4_backward(
            grad_output,
            output,
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            softmax_lse,
            alibi_slopes,
            bias,
            ctx.sm_scale,
            ctx.dropout_p,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.causal,
            ctx.use_exp2,
            ctx.layout,
        )
        dq = ctx.qk_hadamard(dq, inverse=True)
        dk = ctx.qk_hadamard(dk, inverse=True)
        grads = []
        for needed, grad in zip(ctx.needs_input_grad[:3], (dq, dk, dv)):
            grads.append(grad.to(ctx.input_dtype) if needed else None)
        grads.extend([None] * 12)
        return tuple(grads)


@attention_mxfp4_forward_triton_impl.register_fake
def fake_attention_mxfp4_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    bias: Optional[torch.Tensor],
    dropout_p: float,
    layout: str,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlens_q: Optional[int],
    max_seqlens_k: Optional[int],
    return_scores: bool,
    use_exp2: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # check if varlen
    is_varlen = layout == "thd"

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    o_shape = list(q.shape)
    o_shape[-1] = head_size_v
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype,
        requires_grad=True,
    )

    if return_scores:
        scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if return_scores:
        exp_scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q), device=q.device, dtype=torch.float32)
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
    return o, softmax_lse, exp_scores


def triton_attention_mxfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor | None,
    bias: torch.Tensor | None,
    sm_scale: float,
    dropout_p: float,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlens_q: int,
    max_seqlens_k: int,
    causal: bool,
    return_scores: bool,
    use_exp2: bool,
    layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _triton_attention_mxfp4.apply(
        q,
        k,
        v,
        alibi_slopes,
        bias,
        sm_scale,
        dropout_p,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        causal,
        return_scores,
        use_exp2,
        layout,
    )
