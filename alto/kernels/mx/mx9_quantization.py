# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Packed MX9 quantization: each 16-element block becomes 18 uint8 bytes.

Per-block layout ``[max_exp(1B) | prime(1B) | q(16B)]`` = 9 bits/element:
  - ``max_exp`` : shared exponent, E8M0 (floor(log2(amax)) + 127)
  - ``prime``   : 1 bit per pair of 2 elements, set when the pair takes a
                  1-exponent demotion
  - ``q``       : quantized integer clamped +/-127, stored as the bit-identical
                  uint8 view of int8 (recover with ``.view(torch.int8)``)

Exponent / scale / prime math follows ``alto/modifiers/quantization/mx.py``; the
parts shared with MX6 live in ``_mx_common``.

q is clamped to +/-127 instead of widened to int16: demoted pairs use half-scale,
so a value can reach +/-128 (mx.py's quant_max=255) and overflow int8, but int16
would cost ~17 bits/element -- more than bf16, defeating the packing. The clamp
diverges from mx.py by 1 LSB on rare elements that are both demoted and reach
+/-128, so this module's reference is the clamp-127 variant of ``mx9_fake_quantize``.

block_size is fixed at 16 so 8 pairs pack into exactly one prime byte. Packed
integers cannot represent NaN, so q is undefined for blocks containing NaN.
"""

import torch
import triton
import triton.language as tl

from ._mx_common import (
    _calculate_mx_exp,
    _quantize_clamped,
    _pack_bits8,
    _unpack_bits8,
    _dequantize_decoded_q,
    _convert_to_mx_host,
    _convert_from_mx_host,
)

BLOCK_SIZE = 16
QUANT_BIT = 8
PRIME_GROUP = 2
BLOCKS_PER_PROG_DEFAULT = 64


@triton.jit
def _pack_mx9(
    x,
    shared_exp,
    pair,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):
    """Quantize x to int8 q and pack pair bits into prime bytes."""
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    Q_HI: tl.constexpr = (1 << (QUANT_BIT - 1)) - 1

    q = _quantize_clamped(x, shared_exp, QUANT_BIT, Q_HI)
    q_int = q.to(tl.int8)

    prime = _pack_bits8(pair, BLOCKS_PER_PROG, N_PRIME_BYTES)
    return q_int, prime


@triton.jit
def _unpack_mx9(
    q,
    max_exp,
    prime,
    out_dtype: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8

    pair = _unpack_bits8(prime, BLOCKS_PER_PROG, N_PRIME_BYTES)
    return _dequantize_decoded_q(
        q,
        pair,
        max_exp,
        out_dtype,
        BLOCKS_PER_PROG,
        BLOCK_SIZE,
        PRIME_GROUP,
        QUANT_BIT,
    )


# Grid kernels


@triton.jit
def _convert_to_mx9_kernel(
    x_ptr,
    packed_ptr,
    n_blocks,
    last,
    blocks_per_row,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
    N_PACKED_BYTES: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE % PRIME_GROUP == 0, "BLOCK_SIZE must be divisible by PRIME_GROUP")
    tl.static_assert((BLOCK_SIZE // PRIME_GROUP) % 8 == 0, "MX9 prime bitmap requires pair count divisible by 8")

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    PRIME_OFFSET: tl.constexpr = 1
    Q_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES

    # The row size the host allocated must match the segments written below,
    # otherwise a layout change on one side silently writes past the row.
    tl.static_assert(N_PACKED_BYTES == Q_OFFSET + BLOCK_SIZE, "N_PACKED_BYTES does not match the MX9 segment layout")

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    blk_mask = blk < n_blocks

    row = blk // blocks_per_row
    brow = blk % blocks_per_row
    col = tl.arange(0, BLOCK_SIZE)
    col_g = brow[:, None] * BLOCK_SIZE + col[None, :]

    in_range = col_g < last
    load_mask = blk_mask[:, None] & in_range
    in_offs = row[:, None] * stride_row + col_g * stride_col
    x = tl.load(x_ptr + in_offs, mask=load_mask, other=0.0)

    shared_exp, max_exp, pair = _calculate_mx_exp(x, BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP)
    q_int, prime = _pack_mx9(
        x,
        shared_exp,
        pair,
        BLOCKS_PER_PROG,
        BLOCK_SIZE,
        PRIME_GROUP,
        QUANT_BIT,
    )

    # E8M0 bias: store the true exponent + 127.
    e_store = (max_exp + 127).to(tl.uint8)
    tl.store(packed_ptr + blk * N_PACKED_BYTES, e_store, mask=blk_mask)

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * N_PACKED_BYTES + (PRIME_OFFSET + pcol[None, :])
    tl.store(packed_ptr + offs_p, prime, mask=blk_mask[:, None])

    # Preserve the signed int8 byte representation in uint8 storage.
    q_offs = blk[:, None] * N_PACKED_BYTES + (Q_OFFSET + col[None, :])
    tl.store(packed_ptr + q_offs, q_int.to(tl.uint8, bitcast=True), mask=blk_mask[:, None])


@triton.jit
def _convert_from_mx9_kernel(
    packed_ptr,
    y_ptr,
    n_blocks,
    stride_blk,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    N_PACKED_BYTES: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE % PRIME_GROUP == 0, "BLOCK_SIZE must be divisible by PRIME_GROUP")
    tl.static_assert((BLOCK_SIZE // PRIME_GROUP) % 8 == 0, "MX9 prime bitmap requires pair count divisible by 8")

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    PRIME_OFFSET: tl.constexpr = 1
    Q_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES
    tl.static_assert(N_PACKED_BYTES == Q_OFFSET + BLOCK_SIZE, "N_PACKED_BYTES does not match the MX9 segment layout")

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    col = tl.arange(0, BLOCK_SIZE)
    blk_mask = blk < n_blocks
    mask = blk_mask[:, None]

    # Undo the E8M0 bias to recover the true exponent.
    max_exp_u = tl.load(packed_ptr + blk * stride_blk, mask=blk_mask, other=0)
    max_exp = max_exp_u.to(tl.int32) - 127

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * stride_blk + (PRIME_OFFSET + pcol[None, :]) * stride_col
    prime = tl.load(packed_ptr + offs_p, mask=mask, other=0)

    q_offs = blk[:, None] * stride_blk + (Q_OFFSET + col[None, :]) * stride_col
    q_u8 = tl.load(packed_ptr + q_offs, mask=mask, other=0)
    q = q_u8.to(tl.int8, bitcast=True)

    y = _unpack_mx9(
        q,
        max_exp[:, None],
        prime,
        OUT_DTYPE,
        BLOCKS_PER_PROG,
        BLOCK_SIZE,
        PRIME_GROUP,
        QUANT_BIT,
    )
    out_offs = blk[:, None] * BLOCK_SIZE + col[None, :]
    tl.store(y_ptr + out_offs, y, mask=mask)


# Host wrappers


def _mx9_packed_bytes(block_size: int) -> int:
    """Return bytes per MX9 packed block."""
    n_prime_bytes = (block_size // PRIME_GROUP) // 8
    return 1 + n_prime_bytes + block_size


def convert_to_mx9(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
) -> torch.Tensor:
    """High-precision tensor -> packed MX9 ``[n_blocks, n_packed_bytes]`` uint8.

    Blocks are formed along ``axis``, which is transposed to the last dim
    internally. The original shape / axis are not stored, so the caller must pass
    them back to convert_from_mx9.
    """
    return _convert_to_mx_host(
        data_hp,
        kernel=_convert_to_mx9_kernel,
        n_packed_bytes=_mx9_packed_bytes(block_size),
        fmt="mx9_quantization",
        block_size=block_size,
        prime_group=PRIME_GROUP,
        quant_bit=QUANT_BIT,
        axis=axis,
        blocks_per_program=blocks_per_program,
    )


def convert_from_mx9(
    packed: torch.Tensor,
    out_dtype: torch.dtype,
    out_shape,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
) -> torch.Tensor:
    """Packed MX9 tensor -> reconstructed high-precision tensor.

    ``out_shape`` and ``axis`` must exactly match the original convert_to_mx9
    call. They are not stored in ``packed``; a mismatch may silently reorder
    values when it happens to produce the same block count.
    """
    return _convert_from_mx_host(
        packed,
        kernel=_convert_from_mx9_kernel,
        n_packed_bytes=_mx9_packed_bytes(block_size),
        fmt="mx9_quantization",
        out_dtype=out_dtype,
        out_shape=out_shape,
        block_size=block_size,
        prime_group=PRIME_GROUP,
        quant_bit=QUANT_BIT,
        axis=axis,
        blocks_per_program=blocks_per_program,
    )
