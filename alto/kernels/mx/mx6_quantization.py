# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Packed MX6 quantization: each 16-element block becomes 12 uint8 bytes.

Per-block layout ``[max_exp(1B) | prime(1B) | sign(2B) | mantissa(8B)]``
= 6 bits/element, byte-compatible with Quark's MX6 export packer (not vendored
here; its ``idx2`` bitmap is ``prime`` below):
  - ``max_exp``  : E8M0 (true exponent + 127)
  - ``prime``    : 1 bit per pair of 2 elements, set when the pair takes a
                   1-exponent demotion
  - ``sign``     : 16 sign bits, LSB = element 0
  - ``mantissa`` : two ``q & 0xF`` nibbles per byte, low nibble = even element

The mantissa nibbles are the low bits of the signed integer, not ``abs(q)``.

MX6 runs the same block algorithm as MX9 and differs only in element width
(``QUANT_BIT`` = 5, so integers clamp to +/-15) and this byte packing; the shared
math lives in ``_mx_common`` and follows the MX6/MX9 fake-quant port in
``alto/modifiers/quantization/mx.py``.

block_size is fixed at 16 so 8 pairs pack into exactly one prime byte. Packed
integers cannot represent NaN, so the sign / mantissa bytes are undefined for
blocks containing NaN.
"""

import torch
import triton
import triton.language as tl

from ._mx_common import (
    _block_coords,
    _calculate_mx_exp,
    _quantize_clamped,
    _pack_bits8,
    _unpack_bits8,
    _dequantize_decoded_q,
    convert_to_mx,
    convert_from_mx,
)

BLOCK_SIZE = 16
QUANT_BIT = 5
PRIME_GROUP = 2
BLOCKS_PER_PROG_DEFAULT = 64

# [max_exp(1B) | prime | sign bitmap | mantissa nibbles] = 6 bits/element.
N_PACKED_BYTES = 1 + (BLOCK_SIZE // PRIME_GROUP) // 8 + BLOCK_SIZE // 8 + BLOCK_SIZE // 2


@triton.jit
def _pack_mx6(
    x,
    shared_exp,
    pair,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    Q_HI: tl.constexpr = (1 << (QUANT_BIT - 1)) - 1

    q = _quantize_clamped(x, shared_exp, QUANT_BIT, Q_HI)

    qi = q.to(tl.int32)
    mantissa = qi & 0xF
    sign = (qi < 0).to(tl.int32)

    sign_bytes = _pack_bits8(sign, BLOCKS_PER_PROG, N_SIGN_BYTES)

    nib_w = (1 << (4 * tl.arange(0, 2))).to(tl.int32)
    mantissa_g = mantissa.reshape(BLOCKS_PER_PROG, N_MANTISSA_BYTES, 2)
    mantissa_bytes = tl.sum(mantissa_g * nib_w[None, None, :], axis=2)

    prime = _pack_bits8(pair, BLOCKS_PER_PROG, N_PRIME_BYTES)
    return sign_bytes, mantissa_bytes.to(tl.uint8), prime


@triton.jit
def _unpack_mx6(
    sign_bytes,
    mantissa_bytes,
    max_exp,
    prime,
    out_dtype: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8

    sign = _unpack_bits8(sign_bytes, BLOCKS_PER_PROG, N_SIGN_BYTES)

    nib_shift = (4 * tl.arange(0, 2)).to(tl.int32)
    mb = mantissa_bytes.to(tl.int32).reshape(BLOCKS_PER_PROG, N_MANTISSA_BYTES, 1)
    mantissa = (mb >> nib_shift[None, None, :]) & 0xF
    mantissa = mantissa.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)

    # Two's complement on 4 bits: sign=1 with mantissa=13 means q=-3.
    q = tl.where(sign == 1, mantissa - 16, mantissa)

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
def _convert_to_mx6_kernel(
    x_ptr,
    packed_ptr,
    n_blocks,
    last,
    blocks_per_row,
    stride_in_row,
    stride_in_col,
    stride_out_blk,
    stride_out_byte,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE % PRIME_GROUP == 0, "BLOCK_SIZE must be divisible by PRIME_GROUP")
    tl.static_assert((BLOCK_SIZE // PRIME_GROUP) % 8 == 0, "MX6 prime bitmap requires pair count divisible by 8")
    tl.static_assert(BLOCK_SIZE % 8 == 0, "MX6 sign bitmap requires BLOCK_SIZE divisible by 8")

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    PRIME_OFFSET: tl.constexpr = 1
    SIGN_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES
    MANTISSA_OFFSET: tl.constexpr = SIGN_OFFSET + N_SIGN_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    blk_mask = blk < n_blocks

    row, col_g = _block_coords(blk, blocks_per_row, BLOCK_SIZE)
    load_mask = blk_mask[:, None] & (col_g < last)
    in_offs = row[:, None] * stride_in_row + col_g * stride_in_col
    x = tl.load(x_ptr + in_offs, mask=load_mask, other=0.0)

    shared_exp, max_exp, pair = _calculate_mx_exp(x, BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP)
    sign_bytes, mantissa_bytes, prime = _pack_mx6(
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
    tl.store(packed_ptr + blk * stride_out_blk, e_store, mask=blk_mask)

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * stride_out_blk + (PRIME_OFFSET + pcol[None, :]) * stride_out_byte
    tl.store(packed_ptr + offs_p, prime, mask=blk_mask[:, None])

    signcol = tl.arange(0, N_SIGN_BYTES)
    sign_offs = blk[:, None] * stride_out_blk + (SIGN_OFFSET + signcol[None, :]) * stride_out_byte
    tl.store(packed_ptr + sign_offs, sign_bytes, mask=blk_mask[:, None])

    mantcol = tl.arange(0, N_MANTISSA_BYTES)
    mant_offs = blk[:, None] * stride_out_blk + (MANTISSA_OFFSET + mantcol[None, :]) * stride_out_byte
    tl.store(packed_ptr + mant_offs, mantissa_bytes, mask=blk_mask[:, None])


@triton.jit
def _convert_from_mx6_kernel(
    packed_ptr,
    y_ptr,
    n_blocks,
    last,
    blocks_per_row,
    stride_in_blk,
    stride_in_byte,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):

    tl.static_assert(BLOCK_SIZE % PRIME_GROUP == 0, "BLOCK_SIZE must be divisible by PRIME_GROUP")
    tl.static_assert((BLOCK_SIZE // PRIME_GROUP) % 8 == 0, "MX6 prime bitmap requires pair count divisible by 8")
    tl.static_assert(BLOCK_SIZE % 8 == 0, "MX6 sign bitmap requires BLOCK_SIZE divisible by 8")

    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    PRIME_OFFSET: tl.constexpr = 1
    SIGN_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES
    MANTISSA_OFFSET: tl.constexpr = SIGN_OFFSET + N_SIGN_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    blk_mask = blk < n_blocks
    mask = blk_mask[:, None]

    # Undo the E8M0 bias to recover the true exponent.
    max_exp_u = tl.load(packed_ptr + blk * stride_in_blk, mask=blk_mask, other=0)
    max_exp = max_exp_u.to(tl.int32) - 127

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * stride_in_blk + (PRIME_OFFSET + pcol[None, :]) * stride_in_byte
    prime = tl.load(packed_ptr + offs_p, mask=mask, other=0)

    signcol = tl.arange(0, N_SIGN_BYTES)
    sign_offs = blk[:, None] * stride_in_blk + (SIGN_OFFSET + signcol[None, :]) * stride_in_byte
    sign_bytes = tl.load(packed_ptr + sign_offs, mask=mask, other=0)

    mantcol = tl.arange(0, N_MANTISSA_BYTES)
    mant_offs = blk[:, None] * stride_in_blk + (MANTISSA_OFFSET + mantcol[None, :]) * stride_in_byte
    mantissa_bytes = tl.load(packed_ptr + mant_offs, mask=mask, other=0)

    y = _unpack_mx6(
        sign_bytes,
        mantissa_bytes,
        max_exp[:, None],
        prime,
        OUT_DTYPE,
        BLOCKS_PER_PROG,
        BLOCK_SIZE,
        PRIME_GROUP,
        QUANT_BIT,
    )

    row, col_g = _block_coords(blk, blocks_per_row, BLOCK_SIZE)
    out_offs = row[:, None] * stride_out_row + col_g * stride_out_col
    tl.store(y_ptr + out_offs, y, mask=mask & (col_g < last))


# MX6 entry points


def convert_to_mx6(data_hp: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """High-precision tensor -> packed MX6 ``[n_blocks, 12]`` uint8 tensor."""
    return convert_to_mx(data_hp, "mx6", axis=axis)


def convert_from_mx6(
    packed: torch.Tensor,
    out_dtype: torch.dtype,
    out_shape,
    axis: int = -1,
) -> torch.Tensor:
    """Packed MX6 ``[n_blocks, 12]`` uint8 tensor -> reconstructed high-precision tensor."""
    return convert_from_mx(packed, "mx6", out_dtype, out_shape, axis=axis)
