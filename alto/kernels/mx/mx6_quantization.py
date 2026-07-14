# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Packed MX6 quantization Triton device kernels (called by grid kernels).

MX6 shares the *exact same block algorithm* as MX9 (exponent extraction, prime
demotion, power-of-two scale); the only numeric difference is the element
integer width ``quant_bit`` (MX9 = 8, MX6 = 5), exactly mirroring Quark's
``fake_quantize_mx6_mx9`` (see ``alto/modifiers/quantization/mx.py``). Everything
in this module is therefore a straightforward specialization of
``mx9_quantization.py``; the two intentional divergences (to be implemented in
later stages) are:

  1. Element width: ``QUANT_BIT = 5`` -> quantized integers are symmetrically
     clamped to +/-15 (=2^(quant_bit-1)-1), instead of MX9's +/-127.
  2. **Python-export-compatible packing** (achieves a true 6 bits/element): MX9
     stores ``q`` as int8 (one byte per element), dense at its 8-bit width. At 5
     bits, one byte per element would waste 3 bits and defeat the "6" in MX6.
     To match the Python export path, each 16-element block is laid out as:
       - max_exp  : 1 byte  (E8M0, true exponent + 127)
       - prime    : 1 byte  (same bitmap as ``idx2`` in the Python packer)
       - sign     : 2 bytes (16 sign bits, LSB = element 0)
       - mantissa : 8 bytes (two ``q & 0xF`` nibbles per byte)
     -> 12 bytes/block = 6 bits/element. The 4-bit mantissa nibbles are the low
     bits of the signed quantized integer, not ``abs(q)``.

The packed bytes are laid out per block as ``[max_exp, prime, sign bitmap ...,
mantissa nibbles ...]``. For the mantissa segment, low nibble = even element and
high nibble = odd element.

Development is incremental. This file currently implements:
  - Stage 1: stateless device helpers ``_sanitize`` / ``_floor_exp`` /
    ``_round_half_even`` (identical to MX9).
  - Stage 2: ``_calculate_mx6_exp`` (per-block shared exponent + per-element
    shared_exp + per-pair prime bits; width-independent, identical to MX9).
  - Stage 3: ``_pack_mx6`` / ``_unpack_mx6`` (Python-export QDQ pack + inverse).
  - Stage 4: ``_convert_to_mx6_kernel`` / ``_convert_from_mx6_kernel`` grid entries
    (stride-addressed load/store, ragged-tail masking, E8M0 bias).

Still to come: the ``convert_to_mx6`` / ``convert_from_mx6`` host wrappers (Stage 5).

See ``mx9_quantization.py`` for the full rationale behind the shared design
decisions (scale exponent clamp to -126 / FTZ independence, exponent extraction
from native dtype bits, E8M0 max_exp, stride addressing, ragged-tail masking).
"""

import triton
import triton.language as tl

BLOCK_SIZE = 16
QUANT_BIT = 5
PRIME_GROUP = 2
BLOCKS_PER_PROG_DEFAULT = 64

# Python export-compatible packing (Stage 3+): per block, mantissa nibbles
# occupy block_size/2 bytes and the sign bitmap occupies block_size/8 bytes.
MANTISSA_BIT = QUANT_BIT - 1   # 4-bit low mantissa bits (q & 0xF)


@triton.jit
def _sanitize(x):
    """Replace NaN / +/-Inf with 0; used only for the amax / exponent statistics
    copy, does not touch the QDQ data path (identical to MX9)."""
    x = tl.where(x != x, 0.0, x)
    x = tl.where(x == float("inf"), 0.0, x)
    x = tl.where(x == float("-inf"), 0.0, x)
    return x


@triton.jit
def _floor_exp(x):
    """floor(log2(|x|)): extract exponent field directly from native float bits
    (fp32: (bits>>23)&0xFF - 127; bf16: (bits>>7)&0xFF - 127). Identical to MX9."""
    if x.type.element_ty == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        return (((bits >> 23) & 0xFF) - 127).to(tl.int32)
    elif x.type.element_ty == tl.bfloat16:
        bits = x.to(tl.int16, bitcast=True)
        return (((bits >> 7) & 0xFF) - 127).to(tl.int32)
    else:
        tl.static_assert(False, "x must be fp32 / bf16")


@triton.jit
def _round_half_even(y):
    """Round-to-nearest-ties-to-even (matches torch.round), pure tl. Identical to MX9."""
    rounded = tl.floor(y + 0.5)
    is_tie = (y - tl.floor(y)) == 0.5
    is_odd = (rounded - 2.0 * tl.floor(rounded * 0.5)) == 1.0
    return tl.where(is_tie & is_odd, rounded - 1.0, rounded)


@triton.jit
def _calculate_mx6_exp(
    x,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
):
    """Compute per-block shared exponent + per-element shared_exp + per-pair
    prime bits.

    The exponent / prime math is *width-independent* (does not involve
    ``quant_bit``), so this is byte-for-byte the MX9 ``_calculate_mx9_exp``.

    Input ``x``: this program's tile ``[BLOCKS_PER_PROG, BLOCK_SIZE]`` in native
    dtype (each row is one MX6 block). Exponent extraction must be done on native
    dtype, so do NOT pre-cast to fp32 on host.

    Returns:
      - ``shared_exp`` : [BPP, BLOCK_SIZE] int32, per-element shared exponent
                         (demoted pairs get max_exp - 1)
      - ``max_exp``    : [BPP] int32, per-block max exponent
      - ``pair``       : [BPP, N_PAIRS] int32(0/1), per-pair prime bit
                         (1 = that pair gets 1-exponent demotion)
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP

    # Sanitized copy used only for exponent statistics (NaN/Inf -> 0).
    clean = _sanitize(x)

    # Block shared exponent: per-row (block) amax -> floor extract exponent.
    # Cast the reduce result back to native dtype so _floor_exp uses the same
    # branch as the per-element t_exp (some backends promote tl.max to fp32).
    amax = tl.max(tl.abs(clean), axis=1, keep_dims=True).to(clean.dtype)   # [BPP, 1] native
    max_exp = _floor_exp(amax)                             # [BPP, 1] int32

    # Per-element exponent + demote flag (at least 1 octave below block max).
    t_exp = _floor_exp(clean)                              # [BPP, BLOCK_SIZE] int32
    demote = (max_exp - t_exp) >= 1                        # [BPP, BLOCK_SIZE] bool (broadcast)

    # Prime: adjacent PRIME_GROUP(=2) elements form a pair; both must be demoted
    # for the pair to be demoted.
    d = demote.to(tl.int32).reshape(BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP)
    pair_keep = tl.sum(d, axis=2, keep_dims=True) == PRIME_GROUP   # [BPP, N_PAIRS, 1] bool

    # Broadcast back to per-element to get per-element shared_exp.
    pair_b = tl.broadcast_to(pair_keep, (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b.to(tl.int32)            # [BPP, BLOCK_SIZE] int32

    # Pair bits (not broadcast) are kept for prime bitmap packing (Stage 3).
    pair = pair_keep.reshape(BLOCKS_PER_PROG, N_PAIRS).to(tl.int32)
    return shared_exp, max_exp.reshape(BLOCKS_PER_PROG), pair


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
    """Quantize + Python-export-compatible 5-bit pack.

    Returns three uint8 tiles used by the final single packed block layout:
      - ``sign_bytes``     : [BPP, BLOCK_SIZE//8] sign bitmap (LSB = element 0)
      - ``mantissa_bytes`` : [BPP, BLOCK_SIZE//2] 4-bit ``q & 0xF`` nibbles
                             (low nibble = even element, high nibble = odd element)
      - ``prime``          : [BPP, N_PAIRS//8]    prime bitmap (identical to MX9)

    Quantized integers are clamped to +/-15, matching the BFPQuantizer path. The
    mantissa is the low 4 bits of the signed integer, matching the Python export
    packer; it is intentionally not ``abs(q)``.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    N_PRIME_BYTES: tl.constexpr = N_PAIRS // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2   # 2 nibbles per byte
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8       # 8 sign bits per byte
    Q_HI: tl.constexpr = (1 << (QUANT_BIT - 1)) - 1    # 15 for quant_bit=5

    # Same scale as MX9 (clamp scale exponent to -126 for FTZ independence).
    scale_exp = tl.maximum(shared_exp - QUANT_BIT + 2, -126)
    scale = tl.exp2(scale_exp.to(tl.float32))             # [BPP, BLOCK_SIZE], always normal

    xf = x.to(tl.float32)
    q = _round_half_even(tl.div_rn(xf, scale))
    q = tl.minimum(tl.maximum(q, -(Q_HI * 1.0)), Q_HI * 1.0)   # clamp +/-15

    qi = q.to(tl.int32)
    mantissa = qi & 0xF                                     # [BPP, BLOCK_SIZE], low 4 bits
    sign = (qi < 0).to(tl.int32)                            # [BPP, BLOCK_SIZE], 0/1

    # Sign bitmap: 8 sign bits per byte via weights [1, 2, ..., 128].
    weights = tl.exp2(tl.arange(0, 8).to(tl.float32)).to(tl.int32)       # [8]
    sign_g = sign.reshape(BLOCKS_PER_PROG, N_SIGN_BYTES, 8)
    sign_bytes = tl.sum(sign_g * weights[None, None, :], axis=2)         # [BPP, N_SIGN_BYTES]

    # Mantissa: pack two 4-bit low-nibble values per byte via weights [1, 16].
    nib_w = tl.exp2((4 * tl.arange(0, 2)).to(tl.float32)).to(tl.int32)   # [2] = [1, 16]
    mantissa_g = mantissa.reshape(BLOCKS_PER_PROG, N_MANTISSA_BYTES, 2)
    mantissa_bytes = tl.sum(mantissa_g * nib_w[None, None, :], axis=2)   # [BPP, N_MANTISSA_BYTES]

    # Prime bitmap (identical to MX9): 8 pair bits -> 1 byte, LSB = pair0.
    pb = pair.reshape(BLOCKS_PER_PROG, N_PRIME_BYTES, 8)                 # [BPP, NB, 8]
    prime = tl.sum(pb * weights[None, None, :], axis=2)                  # [BPP, NB]
    return sign_bytes.to(tl.uint8), mantissa_bytes.to(tl.uint8), prime.to(tl.uint8)


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
    """Dequantize Python-export-compatible tiles + shared exponent.

    Inverse of ``_pack_mx6``: extract 4-bit ``q & 0xF`` mantissa nibbles and
    sign bits, rebuild the signed integer, then apply
    scale = 2^((max_exp - pair) - quant_bit + 2). ``max_exp`` arrives already
    unbiased (int32, [BPP, 1]); ``prime`` is the per-pair bitmap.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    N_PRIME_BYTES: tl.constexpr = N_PAIRS // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8

    # Unpack sign bitmap (same scheme as prime / Python uint16 little-endian view).
    weights = tl.exp2(tl.arange(0, 8).to(tl.float32)).to(tl.int32)       # [8]
    sb = sign_bytes.to(tl.int32).reshape(BLOCKS_PER_PROG, N_SIGN_BYTES, 1)
    sign = (sb // weights[None, None, :]) % 2                            # [BPP, N_SIGN_BYTES, 8]
    sign = sign.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)

    # Unpack mantissa nibbles: (byte // [1,16]) % 16 -> [even, odd] element.
    nib_w = tl.exp2((4 * tl.arange(0, 2)).to(tl.float32)).to(tl.int32)   # [1, 16]
    mb = mantissa_bytes.to(tl.int32).reshape(BLOCKS_PER_PROG, N_MANTISSA_BYTES, 1)
    mantissa = (mb // nib_w[None, None, :]) % 16                         # [BPP, N_MANTISSA_BYTES, 2]
    mantissa = mantissa.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)

    # Rebuild signed q from sign + low 4 bits. sign=1, mantissa=13 -> -3.
    q = tl.where(sign == 1, mantissa - 16, mantissa)                     # [BPP, BLOCK_SIZE] int32

    # Unpack prime bitmap -> pair -> per-element shared_exp (identical to MX9).
    pbytes = prime.to(tl.int32).reshape(BLOCKS_PER_PROG, N_PRIME_BYTES, 1)
    bits = (pbytes // weights[None, None, :]) % 2                        # [BPP, NB, 8]
    pair = bits.reshape(BLOCKS_PER_PROG, N_PAIRS)                        # [BPP, N_PAIRS]

    pair_b = tl.broadcast_to(pair[:, :, None], (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b                                        # [BPP, BLOCK_SIZE]

    scale_exp = tl.maximum(shared_exp - QUANT_BIT + 2, -126)
    scale = tl.exp2(scale_exp.to(tl.float32))
    y = q.to(tl.float32) * scale
    return y.to(out_dtype)


# ============================================================================
# Grid kernel (entry point): 1D grid, each program handles BLOCKS_PER_PROG
# 16-element blocks. High-precision side addressed by stride (no forced copy).
# Python export-compatible single packed tensor layout:
#   packed : [n_blocks, N_PACKED_BYTES]  per block uint8 bytes
#            [0]                         max_exp E8M0 (true exp + 127)
#            [1 : 1 + N_PRIME_BYTES]     prime / idx2 bitmap
#            next N_SIGN_BYTES           sign bitmap
#            last N_MANTISSA_BYTES       4-bit q low nibbles
# For BLOCK_SIZE=16 this is [max_exp, prime, sign0, sign1, mantissa0..7].
# ============================================================================


@triton.jit
def _convert_to_mx6_kernel(
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
):
    """Input ``x`` addressed by stride as 2D ``[rows, last]`` (``last`` = unpadded
    quant-axis length), so the host does not force a ``.contiguous()`` copy. Each
    block maps back to (row, block-within-row); columns beyond ``last`` (the
    ragged tail) are masked to 0. Output is a freshly allocated contiguous packed
    tensor using the same per-block byte order as the Python export path.
    """
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    N_PACKED_BYTES: tl.constexpr = 1 + N_PRIME_BYTES + N_SIGN_BYTES + N_MANTISSA_BYTES
    PRIME_OFFSET: tl.constexpr = 1
    SIGN_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES
    MANTISSA_OFFSET: tl.constexpr = SIGN_OFFSET + N_SIGN_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)   # global block indices [BPP]
    blk_mask = blk < n_blocks

    # Map each block back to its source (row, block-within-row) coordinate.
    row = blk // blocks_per_row                                  # [BPP]
    brow = blk % blocks_per_row                                  # [BPP]
    col = tl.arange(0, BLOCK_SIZE)                               # [BLOCK]
    col_g = brow[:, None] * BLOCK_SIZE + col[None, :]            # column in the row [BPP, BLOCK]

    in_range = col_g < last
    load_mask = blk_mask[:, None] & in_range
    in_offs = row[:, None] * stride_row + col_g * stride_col

    tl.static_assert(
        x_ptr.type.element_ty == tl.float32 or
        x_ptr.type.element_ty == tl.bfloat16,
        "x must be fp32 / bf16",
    )
    x = tl.load(x_ptr + in_offs, mask=load_mask, other=0.0)       # native dtype tile

    shared_exp, max_exp, pair = _calculate_mx6_exp(
        x, BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP
    )
    sign_bytes, mantissa_bytes, prime = _pack_mx6(
        x, shared_exp, pair,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )

    # Byte 0: max_exp true exponent ([BPP] int32) -> E8M0 uint8: +127 bias.
    e_store = (max_exp + 127).to(tl.uint8)
    tl.store(packed_ptr + blk * N_PACKED_BYTES, e_store, mask=blk_mask)

    # Byte 1..: prime / idx2 bitmap.
    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * N_PACKED_BYTES + (PRIME_OFFSET + pcol[None, :])
    tl.store(packed_ptr + offs_p, prime, mask=blk_mask[:, None])

    # Then sign bitmap bytes, matching quantized_weight_sign.view(torch.uint8).
    signcol = tl.arange(0, N_SIGN_BYTES)
    sign_offs = blk[:, None] * N_PACKED_BYTES + (SIGN_OFFSET + signcol[None, :])
    tl.store(packed_ptr + sign_offs, sign_bytes, mask=blk_mask[:, None])

    # Last segment: q low 4-bit mantissa nibbles.
    mantcol = tl.arange(0, N_MANTISSA_BYTES)
    mant_offs = blk[:, None] * N_PACKED_BYTES + (MANTISSA_OFFSET + mantcol[None, :])
    tl.store(packed_ptr + mant_offs, mantissa_bytes, mask=blk_mask[:, None])


@triton.jit
def _convert_from_mx6_kernel(
    packed_ptr,
    y_ptr,
    n_blocks,
    stride_packed_blk,
    stride_packed_col,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Packed input addressed by stride; output ``y`` is freshly allocated
    contiguous ``[n_blocks, BLOCK_SIZE]`` with flat offsets.

    The packed row is split as ``[max_exp, prime/idx2, sign, mantissa]`` before
    dequantizing.
    """
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    PRIME_OFFSET: tl.constexpr = 1
    SIGN_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES
    MANTISSA_OFFSET: tl.constexpr = SIGN_OFFSET + N_SIGN_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    col = tl.arange(0, BLOCK_SIZE)
    blk_mask = blk < n_blocks
    mask = blk_mask[:, None]

    # max_exp stored as E8M0 uint8 (+127 bias); subtract back to true exponent.
    max_exp_u = tl.load(
        packed_ptr + blk * stride_packed_blk,
        mask=blk_mask,
        other=0,
    )
    max_exp = max_exp_u.to(tl.int32) - 127

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * stride_packed_blk + (PRIME_OFFSET + pcol[None, :]) * stride_packed_col
    prime = tl.load(packed_ptr + offs_p, mask=mask, other=0)              # [BPP, NB] uint8

    signcol = tl.arange(0, N_SIGN_BYTES)
    sign_offs = blk[:, None] * stride_packed_blk + (SIGN_OFFSET + signcol[None, :]) * stride_packed_col
    sign_bytes = tl.load(packed_ptr + sign_offs, mask=mask, other=0)      # [BPP, N_SIGN_BYTES]

    mantcol = tl.arange(0, N_MANTISSA_BYTES)
    mant_offs = blk[:, None] * stride_packed_blk + (MANTISSA_OFFSET + mantcol[None, :]) * stride_packed_col
    mantissa_bytes = tl.load(packed_ptr + mant_offs, mask=mask, other=0)  # [BPP, N_MANTISSA_BYTES]

    y = _unpack_mx6(
        sign_bytes, mantissa_bytes, max_exp[:, None], prime, OUT_DTYPE,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )
    out_offs = blk[:, None] * BLOCK_SIZE + col[None, :]                  # contiguous y offsets
    tl.store(y_ptr + out_offs, y, mask=mask)

