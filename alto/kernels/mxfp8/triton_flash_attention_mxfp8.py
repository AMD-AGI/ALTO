# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""
Fused Attention (MXFP8 / e4m3)
==============================

Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf) with MXFP8 (e4m3) block
scaled operands.

This is a mechanical port of ``triton_flash_attention_mxfp4.py``:
    * FP4 packs two elements per byte along head_dim; FP8 stores one element per
      byte, so all ``// 2`` head-dim halving is removed.
    * ``tl.dot_scaled`` dtype string ``"e2m1"`` -> ``"e4m3"``; the ``lhs_k_pack``
      / ``rhs_k_pack`` arguments are dropped (they only exist for packed FP4).
    * The on-the-fly softmax-probability quantization uses the MXFP8 helpers
      ``_calculate_scales`` / ``_quantize_fp8`` instead of the FP4 pack path.

The scale layout is identical to the FP4 kernel (uint8, ``[.., dim/32]``, 2D
block), so all scale pointer arithmetic carries over unchanged.

Forward is a mechanical port of the FP4 kernel; backward is implemented from
the blockwise-fp8 backward structure with per-reduction-axis MXFP8 scales
(stage-2, ``tl.dot_scaled``; see the plan §9.5 and the backward section below).
"""
from typing import Tuple, Optional
import os
import torch
import triton
import triton.language as tl
from torch._library import triton_op, wrap_triton

from .mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    is_cdna4,
    _calculate_scales,
    _quantize_fp8,
)

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.float32
# Seed the RNG so we get reproducible results for testing.
philox_seed: tl.constexpr = 0x1BF52
philox_offset: tl.constexpr = 0x1D4B42

# e4m3 quantization constants (see mxfp8_quantization.FORMAT_TO_*).
E4M3_TARGET_MAX_POW2 = tl.constexpr(8)
E4M3_MBITS = tl.constexpr(3)
E4M3_FORMAT_ID = tl.constexpr(0)

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

    # FP8 stores one element per byte, so head_dim is not packed.

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
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second, other=0.0):
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
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


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
    BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_BLOCK_DMODEL_QK: tl.constexpr,
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
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
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
            k_offs_k = tl.arange(0, BLOCK_DMODEL_QK)
            ks_offs_k = tl.arange(0, SCALE_BLOCK_DMODEL_QK)
        else:
            k_offs_k = None
            ks_offs_k = None
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL_QK, actual_seqlen_k)
        ks = load_fn(ks_ptrs, k_offs_n, ks_offs_k, actual_seqlen_k, SCALE_ACTUAL_BLOCK_DMODEL_QK, other=1)

        if PADDED_HEAD_V:
            v_offs_k = tl.arange(0, BLOCK_DMODEL_V)
            vs_offs_k = tl.arange(0, BLOCK_DMODEL_V)
        else:
            v_offs_k = None
            vs_offs_k = None
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V)
            vs = load_fn(vs_ptrs, vs_offs_k, vs_offs_n, ACTUAL_BLOCK_DMODEL_V, actual_seqlen_k_scale, other=1)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk += tl.dot_scaled(q, qs, "e4m3", k, ks, "e4m3", out_dtype=tl.float32)

        qk_scaled = qk * SM_SCALE

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
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        if RETURN_SCORES:
            scores_scaled_shifted_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(scores_scaled_shifted_ptrs, q_shifted, mask=scores_scaled_shifted_mask)

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_SCORES:
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                    (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
                tl.store(exp_scores_ptrs, tl.where(keep, p, -p), mask=exp_score_mask)
            p = tl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(exp_scores_ptrs, p, mask=exp_score_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V)
            vs = load_fn(vs_ptrs, vs_offs_k, vs_offs_n, ACTUAL_BLOCK_DMODEL_V, actual_seqlen_k_scale, other=1)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        ps = _calculate_scales(
            p,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            target_max_pow2=E4M3_TARGET_MAX_POW2,
            mbits=E4M3_MBITS,
            IS_2D_BLOCK=False,
        )
        p_fp8 = _quantize_fp8(
            p,
            ps,
            philox_seed,
            batch_philox_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            FP8_FORMAT=E4M3_FORMAT_ID,
            IS_2D_BLOCK=False,
            USE_ASM=USE_ASM,
            USE_SR=False,
        )

        acc += tl.dot_scaled(p_fp8, ps, "e4m3", v, vs, "e4m3", out_dtype=tl.float32)

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

    SCALE_BLOCK_DMODEL_QK: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    BLOCK_N_SCALE: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_qk_scale = tl.arange(0, SCALE_BLOCK_DMODEL_QK)
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
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
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
            l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
            l_ptrs = l_offset + offs_m * stride_lse_m

            l = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)

            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            return

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn
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
        bias_offset = off_h_q * stride_bh
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
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)
        qs_ptrs_mask = qs_ptrs_mask & (offs_d_qk_scale[None, :] < SCALE_ACTUAL_BLOCK_DMODEL_QK)

    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    qs = tl.load(qs_ptrs, mask=qs_ptrs_mask, other=1)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.

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
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min,
            block_max,
            0,
            0,
            0,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            # IS_CAUSAL, ....
            False,
            BLOCK_M,
            BLOCK_DMODEL_QK,
            SCALE_BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            BLOCK_N,
            BLOCK_N_SCALE,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            SCALE_ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            USE_ASM=USE_ASM,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
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
            BLOCK_DMODEL_QK,
            SCALE_BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            BLOCK_N,
            BLOCK_N_SCALE,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            SCALE_ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            USE_ASM=USE_ASM,
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
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
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
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)  # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse)  # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_ptrs_mask)


def get_padded_head_dim(head_size: int):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


@triton_op("alto::attention_mxfp8_forward_triton_impl", mutates_args=())
def attention_mxfp8_forward_triton_impl(
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
        print("attention_mxfp8_forward_triton_impl")
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
    assert layout == "bhsd", (
        f"MXFP8 attention requires layout='bhsd', got {layout!r}: the 2D-block scale groups the "
        "data tensor's last two dims, which are (seqlen, head_dim) only for bhsd. Any other "
        "layout groups across heads and the scale pointer math below reads the wrong scale.")

    # check if varlen
    is_varlen = layout == "thd"

    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (bias is not None):
        assert (bias.numel() < 2**31)

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)
    assert not causal or seqlen_q == seqlen_k, (
        f"causal forward requires seqlen_q == seqlen_k, got {seqlen_q} vs {seqlen_k}: the "
        "kernel uses bottom-right causal masking while PyTorch SDPA/reference tests use top-left "
        "masking for non-square shapes.")
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

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.
    if return_scores:
        exp_scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    # stores LSE the log of the normalization constant / sum of exponential score (unnormalized probabilities)
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


RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _mx_quant(
    x,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Quantize a [BM, BN] tile to e4m3 with MX block scales along the last axis.

    Wraps ``_calculate_scales`` + ``_quantize_fp8`` with the e4m3 constants, so
    each backward dot can quantize an operand along its reduction axis (which the
    caller places last, transposing the tile when needed). ``IS_2D_BLOCK=True``
    groups both axes (head_dim reduction, like fwd Q/K/V); ``False`` is a 1D block
    along the last axis (seqlen reduction, like fwd P). Non-SR path, so philox is
    unused.
    """
    scales = _calculate_scales(
        x,
        BLOCK_M=BM,
        BLOCK_N=BN,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        target_max_pow2=E4M3_TARGET_MAX_POW2,
        mbits=E4M3_MBITS,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )
    xq = _quantize_fp8(
        x,
        scales,
        0,
        0,
        BLOCK_M=BM,
        BLOCK_N=BN,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        FP8_FORMAT=E4M3_FORMAT_ID,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_ASM=USE_ASM,
        USE_SR=False,
    )
    return xq, scales


# Backward operands split into two families (A1, plan §9.5 / AB decision):
#
#   * Q/K/V come from the forward pass as saved e4m3 + a compact 2D-block scale
#     ``[.., seqlen/32, head_dim/32]``. A1 **reuses** that exact quantization —
#     no dequant, no re-quantization. Each dot rebuilds the scale tile the
#     ``tl.dot_scaled`` layout wants (``[outer, reduction/32]``) directly from the
#     saved compact scale via pointer index math (the same broadcast the forward
#     kernel uses for QK). Because a 32x32 block shares one scale, the same saved
#     e4m3 values + tile serve both the head_dim-reduction dots (a/b/e/f) and the
#     seqlen-reduction dots (d/g); only the broadcast axis differs.
#   * dO/P/dS are backward-only tensors the forward never saw, so they are
#     quantized fresh in-kernel with 1D per-row scales (``IS_2D_BLOCK=False``)
#     along their reduction axis — the canonical layout ``tl.dot_scaled`` eats
#     with no broadcast.
_MX_2D = tl.constexpr(False)


@triton.jit
def _load_scale_hd(
    scale_base,
    offs_outer,
    stride_sg,
    stride_dg,
    N_CTX,
    SCALE_D: tl.constexpr,
    SCALE_ACTUAL_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Rebuild a ``[len(offs_outer), SCALE_D]`` scale tile for a **head_dim**-
    reduction dot from the saved compact 2D-block scale.

    ``offs_outer`` indexes the outer (token/seqlen) rows; each maps to compact
    seqlen-group ``offs_outer // QUANT_BLOCK_SIZE``. ``SCALE_D = head_dim/32``
    columns index the reduction (head_dim) groups directly. This mirrors the
    forward kernel's ``qs_ptrs`` broadcast (a 32-row block shares one scale).
    Out-of-range rows (past ``N_CTX``) and padded head groups load a neutral
    scale; they only ever multiply masked-to-zero e4m3 values.
    """
    offs_dg = tl.arange(0, SCALE_D)
    mask = (offs_outer[:, None] < N_CTX) & (offs_dg[None, :] < SCALE_ACTUAL_D)
    ptrs = (scale_base + (offs_outer[:, None] // QUANT_BLOCK_SIZE) * stride_sg +
            offs_dg[None, :] * stride_dg)
    return tl.load(ptrs, mask=mask, other=127)


@triton.jit
def _load_scale_sq(
    scale_base,
    start_outer,
    offs_d,
    stride_sg,
    stride_dg,
    N_CTX,
    BLOCK_OUTER: tl.constexpr,
    SCALE_ACTUAL_D: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """Rebuild a ``[len(offs_d), BLOCK_OUTER/32]`` scale tile for a **seqlen**-
    reduction dot from the same saved compact 2D-block scale.

    This is the transpose-symmetric reuse (AB decision A1): the reduction axis is
    now the seqlen block ``[start_outer, start_outer+BLOCK_OUTER)``, grouped by 32
    into ``BLOCK_OUTER/32`` columns, while ``offs_d`` (head_dim rows) map to
    compact head_dim-group ``offs_d // QUANT_BLOCK_SIZE`` and are broadcast across
    the 32 rows of each group. Element ``[d, sg] = scale[start_outer/32 + sg,
    d/32]`` — the same 32x32 block scale the head_dim path reads, indexed for the
    other axis.
    """
    n_sg: tl.constexpr = BLOCK_OUTER // QUANT_BLOCK_SIZE
    offs_sg = tl.arange(0, n_sg)
    seq_group = start_outer // QUANT_BLOCK_SIZE + offs_sg
    mask = (offs_d[:, None] < SCALE_ACTUAL_D * QUANT_BLOCK_SIZE) & \
           ((seq_group[None, :] * QUANT_BLOCK_SIZE) < N_CTX)
    ptrs = (scale_base + seq_group[None, :] * stride_sg +
            (offs_d[:, None] // QUANT_BLOCK_SIZE) * stride_dg)
    return tl.load(ptrs, mask=mask, other=127)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    cu_seqlens_q,
    max_seqlen_q,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """delta = rowsum(o * do), one value per query row. Feeds ds = p * (dp - delta)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_z = pid_bh // HQ
    off_h = pid_bh % HQ

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        N_CTX_Q = q_end - q_start
    else:
        q_start = 0
        N_CTX_Q = max_seqlen_q

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d_v = tl.arange(0, BLOCK_DMODEL_V)
    mask_m = off_m < N_CTX_Q
    mask_o = mask_m[:, None] & (off_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_doz + off_h * stride_doh + q_start * stride_dom
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d_v[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d_v[None, :] * stride_dok

    o = tl.load(out_ptrs, mask=mask_o, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_o, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    delta_offset = Delta + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    tl.store(delta_offset + off_m * stride_deltam, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dkdv_inner(
    kt_fp8,
    ks_hd,
    vt_fp8,
    vs_hd,
    dk,
    dv,
    q_scale_base,
    stride_qsm,
    stride_qsk,
    offs_d_qk,
    offs_d_v,
    offs_n,
    mask_d_qk,
    mask_d_v,
    q_offset,
    do_offset,
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    l_offset,
    d_offset,
    stride_ldm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    SCALE_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    sm_scale: tl.constexpr,
    lo,
    num_block_m: tl.constexpr,
    USE_EXP2: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Accumulate dk, dv over query blocks for one key block (A1, dot_scaled).

    ``kt_fp8`` / ``vt_fp8`` are the saved-e4m3 key/value tiles transposed to
    ``[head_dim, BLOCK_N]`` (fixed across the m-loop); ``ks_hd`` / ``vs_hd`` are
    their ``[BLOCK_N, head_dim/32]`` scales rebuilt from the forward's compact 2D
    scale by the caller. ``q`` is likewise the **saved e4m3** and is reused with
    no re-quantization: dot a reads its head_dim-grouped scale, dot d reads the
    same 2D block scale re-indexed along seqlen. Only dO/P/dS are quantized fresh
    (plan §9.5 dots a/b/c/d).
    """
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dok

        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d_qk[None, :]
        do_mask = mask_m[:, None] & mask_d_v[None, :]

        # Saved e4m3 q — reused directly, no re-quantization (A1).
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        # dot a: qk = q @ kᵀ, reduction = head_dim. Reuse q's saved head_dim scale.
        qs_hd = _load_scale_hd(q_scale_base, offs_m, stride_qsm, stride_qsk, N_CTX_Q,
                               SCALE_BLOCK_DMODEL_QK, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
        qk = tl.dot_scaled(q, qs_hd, "e4m3", kt_fp8, ks_hd, "e4m3", out_dtype=tl.float32)

        if CAUSAL:
            # Bottom-right causal, matches the forward kernel (offs_n_causal = offs_n + N_CTX_Q - N_CTX_K).
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = l_offset + offs_m * stride_ldm
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        if USE_EXP2:
            qk *= sm_scale * RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] * RCP_LN2)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])

        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        # dot b: dp = do @ vᵀ, reduction = head_dim_v. dO is fresh -> quantize here.
        do_fp8_hd, dos_hd = _mx_quant(do, BLOCK_M, BLOCK_DMODEL_V, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)
        dp = tl.dot_scaled(do_fp8_hd, dos_hd, "e4m3", vt_fp8, vs_hd, "e4m3", out_dtype=tl.float32)

        d_ptrs = d_offset + offs_m * stride_ldm
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        ds = p * (dp - Di[:, None])

        # dot c: dv += pᵀ @ do, reduction = seqlen_q (BLOCK_M). P/dO fresh -> quantize
        # 1D-block along BLOCK_M (their last axis after transpose).
        pt_fp8, ps_c = _mx_quant(tl.trans(p), BLOCK_N, BLOCK_M, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)
        do_t_fp8, dos_c = _mx_quant(tl.trans(do), BLOCK_DMODEL_V, BLOCK_M, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)
        dv += tl.dot_scaled(pt_fp8, ps_c, "e4m3", tl.trans(do_t_fp8), dos_c, "e4m3", out_dtype=tl.float32)

        # dot d: dk += dsᵀ @ q, reduction = seqlen_q (BLOCK_M). dS fresh (1D along
        # BLOCK_M); q reuses the saved 2D block scale re-indexed along seqlen.
        dst_fp8, dss_d = _mx_quant(tl.trans(ds), BLOCK_N, BLOCK_M, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)
        qs_sq = _load_scale_sq(q_scale_base, start_m, offs_d_qk, stride_qsm, stride_qsk, N_CTX_Q,
                               BLOCK_M, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
        dk += tl.dot_scaled(dst_fp8, dss_d, "e4m3", q, qs_sq, "e4m3", out_dtype=tl.float32)

    return dk, dv


@triton.jit
def _bwd_kernel_dkdv(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    V_scale,
    sm_scale: tl.constexpr,
    DO,
    DK,
    DV,
    LSE,
    Delta,
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
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_qsk,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_ksk,
    stride_vsz,
    stride_vsh,
    stride_vsn,
    stride_vsk,
    stride_ldz,
    stride_ldh,
    stride_ldm,
    Z,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """One program per (batch*head_k, key block). Parallelizes dk/dv over keys.

    A1: q/k/v arrive as the forward's saved e4m3 with compact 2D-block scales;
    the kernel reuses them (no re-quant), rebuilding each dot's scale tile from
    the saved scale. Only dO/P/dS are quantized fresh (see ``_attn_bwd_dkdv_inner``).
    """
    SCALE_BLOCK_DMODEL_QK: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    SCALE_BLOCK_DMODEL_V: tl.constexpr = BLOCK_DMODEL_V // QUANT_BLOCK_SIZE
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    SCALE_ACTUAL_BLOCK_DMODEL_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V // QUANT_BLOCK_SIZE

    off_hz = tl.program_id(0)
    start_n = tl.program_id(1)
    off_z = off_hz // HK
    off_h_k = off_hz % HK

    GROUP_SIZE: tl.constexpr = HQ // HK
    off_h_q = off_h_k * GROUP_SIZE if GROUP_SIZE != 1 else off_h_k

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        k_start = tl.load(cu_seqlens_k + off_z)
        N_CTX_Q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        N_CTX_K = tl.load(cu_seqlens_k + off_z + 1) - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    q_scale_start = q_start // QUANT_BLOCK_SIZE
    k_scale_start = k_start // QUANT_BLOCK_SIZE

    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_doz + off_h_q * stride_doh + q_start * stride_dom
    q_scale_base = Q_scale + off_z * stride_qsz + off_h_q * stride_qsh + q_scale_start * stride_qsm
    k_scale_base = K_scale + off_z * stride_ksz + off_h_k * stride_ksh + k_scale_start * stride_ksn
    v_scale_base = V_scale + off_z * stride_vsz + off_h_k * stride_vsh + k_scale_start * stride_vsn
    adj_delta = off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm
    l_offset = LSE + adj_delta
    d_offset = Delta + adj_delta
    dk_offset = DK + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn

    if CAUSAL:
        causal_boundary = start_n * BLOCK_N - BLOCK_M
        lo = (causal_boundary + 1) // BLOCK_M * BLOCK_M
    else:
        lo = 0

    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offs_n < N_CTX_K
    mask_d_qk = offs_d_qk < ACTUAL_BLOCK_DMODEL_QK
    mask_d_v = offs_d_v < ACTUAL_BLOCK_DMODEL_V

    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk
    # Saved e4m3 k/v reused directly; transpose to [head_dim, BLOCK_N] for the
    # QK / dP dots (a/b). Scales are rebuilt from the saved compact 2D scale.
    k_fp8 = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d_qk[None, :], other=0.0)
    v_fp8 = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_v[None, :], other=0.0)
    ks_hd = _load_scale_hd(k_scale_base, offs_n, stride_ksn, stride_ksk, N_CTX_K,
                           SCALE_BLOCK_DMODEL_QK, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
    vs_hd = _load_scale_hd(v_scale_base, offs_n, stride_vsn, stride_vsk, N_CTX_K,
                           SCALE_BLOCK_DMODEL_V, SCALE_ACTUAL_BLOCK_DMODEL_V, QUANT_BLOCK_SIZE)
    kt_fp8 = tl.trans(k_fp8)
    vt_fp8 = tl.trans(v_fp8)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)

    for _ in range(GROUP_SIZE):
        dk, dv = _attn_bwd_dkdv_inner(
            kt_fp8,
            ks_hd,
            vt_fp8,
            vs_hd,
            dk,
            dv,
            q_scale_base,
            stride_qsm,
            stride_qsk,
            offs_d_qk,
            offs_d_v,
            offs_n,
            mask_d_qk,
            mask_d_v,
            q_offset,
            do_offset,
            stride_qm,
            stride_qk,
            stride_dom,
            stride_dok,
            l_offset,
            d_offset,
            stride_ldm,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            SCALE_BLOCK_DMODEL_QK,
            SCALE_ACTUAL_BLOCK_DMODEL_QK,
            sm_scale,
            lo,
            num_block_m,
            USE_EXP2,
            N_CTX_Q,
            N_CTX_K,
            CAUSAL,
            QUANT_BLOCK_SIZE,
            USE_ASM,
        )
        q_offset += stride_qh
        do_offset += stride_doh
        q_scale_base += stride_qsh
        l_offset += stride_ldh
        d_offset += stride_ldh

    dk *= sm_scale

    tl.store(dk_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk, dk,
             mask=mask_n[:, None] & mask_d_qk[None, :])
    tl.store(dv_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk, dv,
             mask=mask_n[:, None] & mask_d_v[None, :])


@triton.jit
def _attn_bwd_dq_inner(
    dq,
    q_fp8_hd,
    qs_hd,
    do_fp8_hd,
    dos_hd,
    k_scale_base,
    v_scale_base,
    stride_ksn,
    stride_ksk,
    stride_vsn,
    stride_vsk,
    offs_d_qk,
    offs_d_v,
    offs_m,
    l_i,
    Di,
    mask_d_qk,
    mask_d_v,
    k_offset,
    v_offset,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    SCALE_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_BLOCK_DMODEL_V: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    SCALE_ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    sm_scale: tl.constexpr,
    hi,
    USE_EXP2: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Accumulate dq over key blocks for one query block (A1, dot_scaled).

    ``q_fp8_hd`` / ``qs_hd`` are the saved-e4m3 q + its head_dim scale (fixed
    across the n-loop, built once by the caller); ``do_fp8_hd`` / ``dos_hd`` are
    the fresh-quantized dO. k/v are the saved e4m3 reused with no re-quant: dots
    e/f read their head_dim scale, dot g reads k's 2D block scale re-indexed along
    seqlen. Only dS is quantized fresh (plan §9.5 dots e/f/g).
    """
    if USE_EXP2:
        l_i *= RCP_LN2

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX_K
        mask_k = mask_n[:, None] & mask_d_qk[None, :]
        mask_v = mask_n[:, None] & mask_d_v[None, :]

        k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
        v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk
        # Saved e4m3 k/v, reused directly.
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        # dot e: qk = q @ kᵀ, reduction = head_dim. Reuse k's saved head_dim scale.
        ks_e = _load_scale_hd(k_scale_base, offs_n, stride_ksn, stride_ksk, N_CTX_K,
                              SCALE_BLOCK_DMODEL_QK, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
        qk = tl.dot_scaled(q_fp8_hd, qs_hd, "e4m3", tl.trans(k), ks_e, "e4m3", out_dtype=tl.float32)

        if CAUSAL:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        if USE_EXP2:
            qk *= sm_scale * RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])

        # dot f: dp = do @ vᵀ, reduction = head_dim_v. Reuse v's saved head_dim scale.
        vs_f = _load_scale_hd(v_scale_base, offs_n, stride_vsn, stride_vsk, N_CTX_K,
                              SCALE_BLOCK_DMODEL_V, SCALE_ACTUAL_BLOCK_DMODEL_V, QUANT_BLOCK_SIZE)
        dp = tl.dot_scaled(do_fp8_hd, dos_hd, "e4m3", tl.trans(v), vs_f, "e4m3", out_dtype=tl.float32)
        ds = p * (dp - Di[:, None])

        # dot g: dq += ds @ k, reduction = seqlen_k (BLOCK_N). dS fresh (1D along
        # BLOCK_N); k reuses the saved 2D block scale re-indexed along seqlen.
        ds_fp8, dss_g = _mx_quant(ds, BLOCK_M, BLOCK_N, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)
        ks_sk = _load_scale_sq(k_scale_base, start_n, offs_d_qk, stride_ksn, stride_ksk, N_CTX_K,
                               BLOCK_N, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
        dq += tl.dot_scaled(ds_fp8, dss_g, "e4m3", k, ks_sk, "e4m3", out_dtype=tl.float32)

    return dq


@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    V_scale,
    sm_scale: tl.constexpr,
    DO,
    DQ,
    LSE,
    Delta,
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
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_qsk,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_ksk,
    stride_vsz,
    stride_vsh,
    stride_vsn,
    stride_vsk,
    stride_ldz,
    stride_ldh,
    stride_ldm,
    Z,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_block_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """One program per (batch*head_q, query block). Parallelizes dq over queries.

    A1: q/k/v are the forward's saved e4m3 + compact 2D-block scales, reused with
    no re-quant. Only dO (fresh input) and dS are quantized in-kernel.
    """
    SCALE_BLOCK_DMODEL_QK: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    SCALE_BLOCK_DMODEL_V: tl.constexpr = BLOCK_DMODEL_V // QUANT_BLOCK_SIZE
    SCALE_ACTUAL_BLOCK_DMODEL_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
    SCALE_ACTUAL_BLOCK_DMODEL_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V // QUANT_BLOCK_SIZE

    off_hz = tl.program_id(0)
    start_m = tl.program_id(1)
    off_z = off_hz // HQ
    off_h_q = off_hz % HQ

    GROUP_SIZE: tl.constexpr = HQ // HK
    off_h_k = off_h_q // GROUP_SIZE if GROUP_SIZE != 1 else off_h_q

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        k_start = tl.load(cu_seqlens_k + off_z)
        N_CTX_Q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        N_CTX_K = tl.load(cu_seqlens_k + off_z + 1) - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    q_scale_start = q_start // QUANT_BLOCK_SIZE
    k_scale_start = k_start // QUANT_BLOCK_SIZE

    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_doz + off_h_q * stride_doh + q_start * stride_dom
    q_scale_base = Q_scale + off_z * stride_qsz + off_h_q * stride_qsh + q_scale_start * stride_qsm
    k_scale_base = K_scale + off_z * stride_ksz + off_h_k * stride_ksh + k_scale_start * stride_ksn
    v_scale_base = V_scale + off_z * stride_vsz + off_h_k * stride_vsh + k_scale_start * stride_vsn
    adj_delta = off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm
    l_offset = LSE + adj_delta
    d_offset = Delta + adj_delta
    dq_offset = DQ + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm

    if CAUSAL:
        hi = tl.minimum(BLOCK_M // BLOCK_N * (start_m + 1), num_block_n) * BLOCK_N
    else:
        hi = num_block_n * BLOCK_N

    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_m = offs_m < N_CTX_Q
    mask_d_qk = offs_d_qk < ACTUAL_BLOCK_DMODEL_QK
    mask_d_v = offs_d_v < ACTUAL_BLOCK_DMODEL_V

    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    do_ptrs = do_offset + offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dok
    # Saved e4m3 q reused directly; dO is a fresh input and is quantized here.
    q_fp8_hd = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d_qk[None, :], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d_v[None, :], other=0.0)

    l_i = tl.load(l_offset + offs_m * stride_ldm, mask=mask_m, other=0.0)
    Di = tl.load(d_offset + offs_m * stride_ldm, mask=mask_m, other=0.0)

    # q's saved head_dim scale (dot e), fixed across the n-loop; dO quantized fresh.
    qs_hd = _load_scale_hd(q_scale_base, offs_m, stride_qsm, stride_qsk, N_CTX_Q,
                           SCALE_BLOCK_DMODEL_QK, SCALE_ACTUAL_BLOCK_DMODEL_QK, QUANT_BLOCK_SIZE)
    do_fp8_hd, dos_hd = _mx_quant(do, BLOCK_M, BLOCK_DMODEL_V, QUANT_BLOCK_SIZE, _MX_2D, USE_ASM)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    dq = _attn_bwd_dq_inner(
        dq,
        q_fp8_hd,
        qs_hd,
        do_fp8_hd,
        dos_hd,
        k_scale_base,
        v_scale_base,
        stride_ksn,
        stride_ksk,
        stride_vsn,
        stride_vsk,
        offs_d_qk,
        offs_d_v,
        offs_m,
        l_i,
        Di,
        mask_d_qk,
        mask_d_v,
        k_offset,
        v_offset,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL_QK,
        BLOCK_DMODEL_V,
        SCALE_BLOCK_DMODEL_QK,
        SCALE_BLOCK_DMODEL_V,
        SCALE_ACTUAL_BLOCK_DMODEL_QK,
        SCALE_ACTUAL_BLOCK_DMODEL_V,
        sm_scale,
        hi,
        USE_EXP2,
        N_CTX_Q,
        N_CTX_K,
        CAUSAL,
        QUANT_BLOCK_SIZE,
        USE_ASM,
    )

    dq *= sm_scale
    tl.store(dq_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk, dq,
             mask=mask_m[:, None] & mask_d_qk[None, :])


@triton_op("alto::attention_mxfp8_backward_triton_impl", mutates_args=())
def attention_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    causal: bool,
    layout: str,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    use_exp2: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 flash-attention backward (stage-2 / A1, ``tl.dot_scaled``).

    A1 operand source: the forward-saved e4m3 q/k/v and their compact 2D-block
    scales are passed straight in and reused in *every* dot (no entry dequant, no
    re-quantization of q/k/v). For head_dim-reduction dots the 2D scale already
    groups along the reduction axis; for seqlen-reduction dots the same compact
    2D scale is re-indexed (``_load_scale_sq``) to broadcast per reduction row.
    Only the backward-only tensors dO/P/dS are quantized in-kernel (1D per-row
    along each dot's reduction axis), then fed to ``tl.dot_scaled`` (plan §9.5).
    Matches ``mxfp8_attention_backward_reference_stage2``.

    Requires CDNA4 (native ``tl.dot_scaled``).
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_scale = q_scale.contiguous()
    k_scale = k_scale.contiguous()
    v_scale = v_scale.contiguous()

    if not do.is_contiguous():
        do = do.contiguous()

    assert layout == "bhsd", (
        f"MXFP8 attention requires layout='bhsd', got {layout!r}: the 2D-block scale groups the "
        "data tensor's last two dims, which are (seqlen, head_dim) only for bhsd. Any other "
        "layout groups across heads and the scale pointer math below reads the wrong scale.")

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    assert not causal or max_seqlen_q == max_seqlen_k, (
        f"causal backward requires seqlen_q == seqlen_k, got {max_seqlen_q} vs {max_seqlen_k}: the "
        "kernel masks bottom-right aligned while the PyTorch references and F.sdpa mask top-left, "
        "and the two conventions only agree on square shapes.")
    stride_qz, stride_qh, stride_qm, stride_qk = get_strides_from_layout(q, layout)
    stride_kz, stride_kh, stride_kn, stride_kk = get_strides_from_layout(k, layout)
    stride_vz, stride_vh, stride_vn, stride_vk = get_strides_from_layout(v, layout)
    stride_oz, stride_oh, stride_om, stride_ok = get_strides_from_layout(o, layout)
    stride_doz, stride_doh, stride_dom, stride_dok = get_strides_from_layout(do, layout)
    # Compact 2D-block scale strides ([.., seqlen/32, head_dim/32]); reused by the
    # kernels for both head_dim- and seqlen-reduction dots (A1).
    stride_qsz, stride_qsh, stride_qsm, stride_qsk = get_strides_from_layout(q_scale, layout)
    stride_ksz, stride_ksh, stride_ksn, stride_ksk = get_strides_from_layout(k_scale, layout)
    stride_vsz, stride_vsh, stride_vsn, stride_vsk = get_strides_from_layout(v_scale, layout)
    is_varlen = layout == "thd"

    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    dq = torch.zeros_like(q, dtype=bwd_torch_dtype)
    dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
    dv = torch.zeros_like(v, dtype=bwd_torch_dtype)

    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    num_block_m = triton.cdiv(max_seqlen_q, BLOCK_M)
    num_block_n = triton.cdiv(max_seqlen_k, BLOCK_N)

    wrap_triton(_bwd_preprocess)[(num_block_m, batch * nheads_q)](
        o,
        do,
        delta,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        cu_seqlens_q,
        max_seqlen_q,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
    )

    wrap_triton(_bwd_kernel_dq)[(batch * nheads_q, num_block_m)](
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        do,
        dq,
        softmax_lse,
        delta,
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
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_qsz,
        stride_qsh,
        stride_qsm,
        stride_qsk,
        stride_ksz,
        stride_ksh,
        stride_ksn,
        stride_ksk,
        stride_vsz,
        stride_vsh,
        stride_vsn,
        stride_vsk,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_n=num_block_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        USE_ASM=is_cdna4(),
        num_warps=4,
        num_stages=1,
    )

    wrap_triton(_bwd_kernel_dkdv)[(batch * nheads_k, num_block_n)](
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        do,
        dk,
        dv,
        softmax_lse,
        delta,
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
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_qsz,
        stride_qsh,
        stride_qsm,
        stride_qsk,
        stride_ksz,
        stride_ksh,
        stride_ksn,
        stride_ksk,
        stride_vsz,
        stride_vsh,
        stride_vsn,
        stride_vsk,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_DEFAULT,
        USE_ASM=is_cdna4(),
        num_warps=4,
        num_stages=1,
    )

    return dq, dk, dv


@attention_mxfp8_backward_triton_impl.register_fake
def fake_attention_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    causal: bool,
    layout: str,
    cu_seqlens_q: Optional[int],
    cu_seqlens_k: Optional[int],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    use_exp2: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q, dtype=bwd_torch_dtype)
    dk = torch.empty_like(k, dtype=bwd_torch_dtype)
    dv = torch.empty_like(v, dtype=bwd_torch_dtype)
    return dq, dk, dv


@torch.compiler.allow_in_graph
class _triton_attention_mxfp8(torch.autograd.Function):

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
        q, q_scale = torch.ops.alto.convert_to_mxfp8(q, mxfp_format="e4m3", axis=-1, is_2d_block=True)
        k, k_scale = torch.ops.alto.convert_to_mxfp8(k, mxfp_format="e4m3", axis=-1, is_2d_block=True)
        v, v_scale = torch.ops.alto.convert_to_mxfp8(v, mxfp_format="e4m3", axis=-1, is_2d_block=True)

        output, softmax_lse, exp_scores = torch.ops.alto.attention_mxfp8_forward_triton_impl(
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
    def backward(ctx, *grad_outputs):
        do = grad_outputs[0]
        q, k, v, o, softmax_lse, alibi_slopes, bias, q_scale, k_scale, v_scale = ctx.saved_tensors
        assert bias is None, "MXFP8 attention backward does not support bias yet."
        assert alibi_slopes is None, "MXFP8 attention backward does not support alibi yet."
        assert ctx.dropout_p == 0.0, "MXFP8 attention backward does not support dropout yet."

        dq, dk, dv = torch.ops.alto.attention_mxfp8_backward_triton_impl(
            do,
            q,
            k,
            v,
            o,
            softmax_lse,
            q_scale,
            k_scale,
            v_scale,
            sm_scale=ctx.sm_scale,
            causal=ctx.causal,
            layout=ctx.layout,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlens_q,
            max_seqlen_k=ctx.max_seqlens_k,
            use_exp2=ctx.use_exp2,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


@attention_mxfp8_forward_triton_impl.register_fake
def fake_attention_mxfp8_forward_triton_impl(
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
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]  # output shape should match v's head dim
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype,
        requires_grad=True,
    )

    # check if varlen
    is_varlen = layout == "thd"

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, seqlen_q, seqlen_k = get_shape_from_layout(
        q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k)

    if return_scores:
        scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)

    if return_scores:
        exp_scores = torch.zeros((batch, nheads_q, max_seqlens_q, max_seqlens_k), device=q.device, dtype=torch.float32)
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    if is_varlen:
        softmax_lse = torch.empty((q.shape[0], nheads_q), device=q.device, dtype=torch.float32)
    else:
        softmax_lse = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)
    return o, softmax_lse, exp_scores


def triton_attention_mxfp8(
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
    return _triton_attention_mxfp8.apply(
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
