# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Packed MX9 quantization Triton device kernels (called by grid kernels).

Unlike the fake-quant path (``alto/kernels/mx/quantize_triton.py`` which outputs
dequantized bf16), this module performs *real packed* quantization: each MX9
block is compressed into a **single packed uint8 tensor**, with each block's
bytes laid out as ``[max_exp(1B) | prime(1B) | q(16B)]`` (mirroring MX6's
prefix convention in ``mx6_quantization.py``):
  - ``max_exp`` : 1 byte, per-block shared exponent, uint8 E8M0
                  (floor(log2(amax)) + 127 bias)
  - ``prime``   : 1 byte, per-block prime bitmap (1 bit per pair of 2 elements,
                  indicates whether that pair gets a 1-exponent demotion)
  - ``q``       : 16 bytes, per-element quantized integer (symmetric clamp
                  +/-127), stored as the bit-identical uint8 reinterpretation of
                  int8 -- reconstruct via ``packed[..., 2:].view(torch.int8)``.
-> 18 bytes/block = 9 bits/element (8b value + 0.5b amortized max_exp + 0.5b prime).

The exponent extraction / scale / prime math is aligned with
``alto/modifiers/quantization/mx.py`` (exponent extracted from native dtype bits
= floor; QDQ computed in fp32). The **only intentional divergence** is the
quantized value clamp:

  True 8-bit variant: quantized integers are symmetrically clamped to +/-127,
  stored as **int8**, yielding exactly 9 bits/element (8b value + 0.5b amortized
  max_exp + 0.5b prime). In contrast, mx.py uses quant_max=255 for demoted pairs
  and can preserve +/-128. The two differ only for elements that are both demoted
  AND whose quantized value reaches +/-128.

Therefore the acceptance reference is **not** ``mx9_fake_quantize`` but its
clamp-127 variant (see tests):
``unpack(pack(x)) == mx9_clamp127_ref(x)`` bit-exact.

Constraints:
  - Blocks are formed along the last dimension only (host transposes the quant
    axis to dim -1). The tensor is passed to the kernel **by stride** (no forced
    ``.contiguous()`` copy); the kernel addresses elements via row/column strides
    and masks the ragged tail block instead of host-side ``F.pad``.
  - ``block_size`` is fixed at 16: each block has 8 pairs which pack into
    exactly 1 byte of prime bitmap.

Key design decisions (lessons learned / non-obvious trade-offs -- understand
these before modifying):
  1. q uses int8 + clamp +/-127, not int16: demoted pairs use half-scale so
     the quantized value can reach +/-128 (quant_max=255), which overflows signed
     int8. We previously used int16 to preserve +/-128 bit-exactness, but that
     yields ~17 bits/element (larger than bf16), defeating the purpose of packing.
     Choosing clamp +/-127 achieves true 9 bits/element, at the cost of a 1-LSB
     divergence from mx.py on the rare "demoted AND reaching +/-128" elements
     (measured ~0.1%).
  2. Scale exponent is clamped to the minimum normal exponent -126 (following
     mxfp4 / PyTorch #125557), rather than a post-hoc scale==0 guard: the latter
     implicitly depends on GPU FTZ flushing subnormal scales to 0, which once
     caused divergence of ~1e-41 for deep-subnormal blocks vs a no-FTZ reference.
     Clamping keeps scale always normal, FTZ-independent, and degenerate blocks
     deterministically output 0.
  3. Exponents are extracted from native dtype bits (fp32/bf16 >>... - 127);
     do NOT pre-cast to fp32 on host.
  4. max_exp is stored as uint8 E8M0 (true exponent + 127), not int32: true
     exponents for any finite input fall in [-127, 127], +127 -> [0, 254] which
     fits uint8 without overflow; this achieves 1 byte/block and 9 bits/element.
  5. All three components live in one uint8 tensor (not three separate tensors):
     ``q`` is computed as int8 in registers (see ``_pack_mx9`` / ``_unpack_mx9``,
     unchanged) but stored/loaded via the packed uint8 tensor using
     ``.to(tl.uint8/tl.int8, bitcast=True)`` -- a pure bit reinterpretation, not a
     value cast (which would incorrectly clamp/wrap negative values). This mirrors
     ``mx6_quantization.py``'s single-tensor, Python-export-compatible layout.

Known limitations:
  - NaN: packed integers cannot represent NaN; blocks containing NaN will have
    garbage q values (only the fake-quant path supports NaN pass-through).
  - The acceptance reference is the custom clamp-127 variant (Quark-divergent),
    not Quark/mx.py itself.
"""

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 16
QUANT_BIT = 8
PRIME_GROUP = 2
BLOCKS_PER_PROG_DEFAULT = 64

# torch dtype -> triton dtype (used by unpack output cast).
_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _sanitize(x):
    """Replace NaN / +/-Inf with 0; used only for the amax / exponent statistics
    copy, does not touch the QDQ data path.

    (Faithfully replicates Quark/reference: exponent statistics use sanitized
    values to prevent Inf from polluting the block scale; the actual round still
    operates on the original x, letting Inf be clamped. NaN is not representable
    in the packed path (q is int8), so NaN inputs yield undefined q values.)
    """
    x = tl.where(x != x, 0.0, x)
    x = tl.where(x == float("inf"), 0.0, x)
    x = tl.where(x == float("-inf"), 0.0, x)
    return x


@triton.jit
def _floor_exp(x):
    """floor(log2(|x|)): extract exponent field directly from native float bits.

    Branches by dtype, aligned with ``mx.py``'s ``_exponent_frexp_no_exception``:
      - fp32 : (bits>>23)&0xFF - 127
      - bf16 : (bits>>7) &0xFF - 127
    The ``& mask`` also clears sign-bit extension from arithmetic right shift,
    so negative values are safe. Always returns int32 to avoid int16 mixing.
    """
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
    """Round-to-nearest-ties-to-even (matches torch.round), pure tl
    implementation to avoid libdevice dependency."""
    rounded = tl.floor(y + 0.5)
    is_tie = (y - tl.floor(y)) == 0.5
    is_odd = (rounded - 2.0 * tl.floor(rounded * 0.5)) == 1.0
    return tl.where(is_tie & is_odd, rounded - 1.0, rounded)


@triton.jit
def _calculate_mx9_exp(
    x,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
):
    """Compute per-block shared exponent + per-element shared_exp + per-pair
    prime bits.

    Input ``x``: this program's tile ``[BLOCKS_PER_PROG, BLOCK_SIZE]`` in
    **native dtype** (each row is one MX9 block). Exponent extraction must be
    done on native dtype, so do NOT pre-cast to fp32 on host.

    Returns:
      - ``shared_exp`` : [BPP, BLOCK_SIZE] int32, per-element shared exponent
                         (demoted pairs get -1)
      - ``max_exp``    : [BPP, 1] int32, per-block max exponent
      - ``pair``       : [BPP, N_PAIRS] int32(0/1), per-pair prime bit
                         (1 = that pair gets 1-exponent demotion)
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP

    # Sanitized copy used only for exponent statistics (NaN/Inf -> 0).
    clean = _sanitize(x)

    # Block shared exponent: per-row (block) amax -> floor extract exponent.
    # tl.max on some backends promotes the reduce result to fp32; cast back to
    # native dtype to ensure _floor_exp uses the same branch as per-element
    # t_exp, consistent with the mxfp4 kernel (see mxfp_quantization.py:72).
    amax = tl.max(tl.abs(clean), axis=1, keep_dims=True).to(clean.dtype)   # [BPP, 1] native dtype
    max_exp = _floor_exp(amax)                             # [BPP, 1] int32

    # Per-element exponent + demote flag: whether at least 1 octave below block max.
    t_exp = _floor_exp(clean)                              # [BPP, BLOCK_SIZE] int32
    demote = (max_exp - t_exp) >= 1                        # [BPP, BLOCK_SIZE] bool (max_exp broadcasts)

    # Prime: adjacent PRIME_GROUP(=2) elements form a pair; both must be demoted
    # for the pair to be demoted.
    d = demote.to(tl.int32).reshape(BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP)
    pair_keep = tl.sum(d, axis=2, keep_dims=True) == PRIME_GROUP   # [BPP, N_PAIRS, 1] bool

    # Broadcast back to per-element to get per-element shared_exp.
    pair_b = tl.broadcast_to(pair_keep, (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b.to(tl.int32)            # [BPP, BLOCK_SIZE] int32

    # Pair bits (not broadcast) are kept for prime bitmap packing.
    pair = pair_keep.reshape(BLOCKS_PER_PROG, N_PAIRS).to(tl.int32)
    return shared_exp, max_exp.reshape(BLOCKS_PER_PROG), pair


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
    """Quantize + pack: original x -> (q:int8 values, prime:uint8 bitmap).

    True 8-bit variant: quantized integers are symmetrically clamped to +/-127,
    stored as int8 (achieving 9 bits/element). This intentionally diverges from
    mx.py (which uses quant_max=255 for demoted pairs, preserving +/-128) at
    the +/-128 boundary -- this variant serves as its own reference (see the
    clamp-127 reference in tests). ``prime`` packs every 8 pair bits into one
    byte.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    N_PRIME_BYTES: tl.constexpr = N_PAIRS // 8   # block 16 -> 1; block 32 -> 2

    # Per-element scale = 2^(shared_exp - quant_bit + 2).
    # Following mxfp4 (mxfp_quantization.py:81-89 / PyTorch #125557): clamp the
    # scale exponent to fp32 minimum normal exponent -126, ensuring scale is
    # always normal (never subnormal or zero), eliminating 0/0 and GPU FTZ
    # dependence from the source (only affects blocks with amax < 2^-120, which
    # real data never reaches).
    scale_exp = tl.maximum(shared_exp - QUANT_BIT + 2, -126)
    scale = tl.exp2(scale_exp.to(tl.float32))   # [BPP, BLOCK_SIZE], always normal (!=0)

    # Q part of QDQ operates on the *original* x (NaN will produce garbage, see below).
    # div_rn (IEEE RN) matches PyTorch eager x/scale; plain / lowers to fast div
    # (max 2 ULP error) which can flip round at .5 boundaries.
    xf = x.to(tl.float32)
    q = _round_half_even(tl.div_rn(xf, scale))
    # True 8-bit: symmetric clamp to +/-127.
    #   - Normal data: demoted elements have |q|<128 mathematically; only rounding
    #     edge cases can produce 128, which is clamped to 127.
    #   - +/-Inf input: Inf/scale=Inf, clamped to +/-127 (does not pollute block
    #     scale since _sanitize already cleared it).
    #   - Intentional divergence from mx.py: mx.py uses quant_max=255 for demoted
    #     pairs, preserving +/-128.
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)

    # NaN limitation: packed integers cannot represent NaN; blocks containing NaN
    # will have garbage q values (round-trip tests must exclude NaN inputs; only
    # the fake-quant path supports NaN pass-through).
    q_int = q.to(tl.int8)

    # Prime packing: N_PAIRS pair bits -> every 8 bits packed into 1 byte.
    # weights = [1,2,4,...,128], generated via exp2 to avoid 1<<tensor syntax.
    weights = tl.exp2(tl.arange(0, 8).to(tl.float32)).to(tl.int32)        # [8]
    pb = pair.reshape(BLOCKS_PER_PROG, N_PRIME_BYTES, 8)                  # [BPP, NB, 8]
    prime = tl.sum(pb * weights[None, None, :], axis=2)                  # [BPP, NB]
    return q_int, prime.to(tl.uint8)


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
    """Dequantize: three-part tuple (q:int8, max_exp:[BPP,1] int32 unbiased,
    prime:[BPP,NB] uint8) -> reconstructed values.

    Rebuilds scale = 2^((max_exp - pair) - quant_bit + 2), then y = q * scale.
    Note: max_exp has already been unbiased from E8M0 uint8 at the kernel entry;
    it arrives here as int32.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    N_PRIME_BYTES: tl.constexpr = N_PAIRS // 8

    # Unpack prime bitmap: extract 8 bits from each byte -> one flag per pair.
    weights = tl.exp2(tl.arange(0, 8).to(tl.float32)).to(tl.int32)        # [8]
    pbytes = prime.to(tl.int32).reshape(BLOCKS_PER_PROG, N_PRIME_BYTES, 1)
    bits = (pbytes // weights[None, None, :]) % 2                         # [BPP, NB, 8]
    pair = bits.reshape(BLOCKS_PER_PROG, N_PAIRS)                         # [BPP, N_PAIRS]

    # Broadcast back to per-element, rebuild per-element shared_exp.
    pair_b = tl.broadcast_to(pair[:, :, None], (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b                                         # [BPP, BLOCK_SIZE] (max_exp broadcasts)

    # Same scale exponent clamping as pack (-126 minimum) to ensure consistency.
    scale_exp = tl.maximum(shared_exp - QUANT_BIT + 2, -126)
    scale = tl.exp2(scale_exp.to(tl.float32))
    y = q.to(tl.float32) * scale
    return y.to(out_dtype)


# ============================================================================
# Grid kernel (entry point): 1D grid, each program handles BLOCKS_PER_PROG
# 16-element blocks. The high-precision side is addressed by stride (no forced
# copy). Single packed tensor layout (mirrors mx6_quantization.py):
#   packed : [n_blocks, N_PACKED_BYTES]  per block uint8 bytes
#            [0]                          max_exp E8M0 (true exp + 127)
#            [1 : 1 + N_PRIME_BYTES]      prime bitmap
#            [1 + N_PRIME_BYTES : ]       q values (int8, bit-reinterpreted uint8)
# For BLOCK_SIZE=16 this is [max_exp, prime, q0..q15] = 18 bytes/block.
# ============================================================================


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
):
    """Input ``x`` is addressed **by stride** as a 2D ``[rows, last]`` tensor
    (``last`` = unpadded quant-axis length), so the host does not force a
    ``.contiguous()`` copy. Each of the ``BLOCKS_PER_PROG`` blocks maps back to
    ``(row, block-within-row)``; columns beyond ``last`` (the ragged tail block)
    are masked to 0, replacing host-side ``F.pad``.

    Output is a freshly allocated contiguous packed tensor; ``q`` (int8 in
    registers) is bit-reinterpreted to uint8 before storing (a bitcast, not a
    value cast, so negative values are preserved exactly).
    """
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    N_PACKED_BYTES: tl.constexpr = 1 + N_PRIME_BYTES + BLOCK_SIZE
    PRIME_OFFSET: tl.constexpr = 1
    Q_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)   # global block indices [BPP]
    blk_mask = blk < n_blocks

    # Map each block back to its source (row, block-within-row) coordinate.
    row = blk // blocks_per_row                                  # [BPP]
    brow = blk % blocks_per_row                                  # [BPP]
    col = tl.arange(0, BLOCK_SIZE)                               # [BLOCK]
    col_g = brow[:, None] * BLOCK_SIZE + col[None, :]           # column in the (unpadded) row [BPP, BLOCK]

    # Strided gather: address = row*stride_row + col*stride_col. Columns past the
    # unpadded width (last) are the padding tail -> masked to 0 (same statistics
    # as the previous F.pad-with-zeros behaviour).
    in_range = col_g < last
    load_mask = blk_mask[:, None] & in_range
    in_offs = row[:, None] * stride_row + col_g * stride_col

    tl.static_assert(
        x_ptr.type.element_ty == tl.float32 or
        x_ptr.type.element_ty == tl.bfloat16,
        "x must be fp32 / bf16",
    )
    x = tl.load(x_ptr + in_offs, mask=load_mask, other=0.0)     # native dtype tile

    shared_exp, max_exp, pair = _calculate_mx9_exp(
        x, BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP
    )
    q_int, prime = _pack_mx9(
        x, shared_exp, pair,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )

    # Byte 0: max_exp true exponent ([BPP] int32) -> E8M0 uint8: +127 bias.
    e_store = (max_exp + 127).to(tl.uint8)
    tl.store(packed_ptr + blk * N_PACKED_BYTES, e_store, mask=blk_mask)

    # Byte 1..: prime bitmap.
    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * N_PACKED_BYTES + (PRIME_OFFSET + pcol[None, :])
    tl.store(packed_ptr + offs_p, prime, mask=blk_mask[:, None])

    # Last BLOCK_SIZE bytes: q values, bit-reinterpreted int8 -> uint8.
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
):
    """Packed input is addressed **by stride** so the host does not force a
    ``.contiguous()`` copy. The output ``y`` is freshly allocated contiguous
    ``[n_blocks, BLOCK_SIZE]`` and uses classic flat offsets.

    The packed row is split as ``[max_exp, prime, q]``; ``q`` bytes are loaded
    as uint8 then bit-reinterpreted back to int8 (a bitcast, not a value cast)
    before dequantizing.
    """
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8
    PRIME_OFFSET: tl.constexpr = 1
    Q_OFFSET: tl.constexpr = PRIME_OFFSET + N_PRIME_BYTES

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    col = tl.arange(0, BLOCK_SIZE)
    blk_mask = blk < n_blocks
    mask = blk_mask[:, None]

    # max_exp stored as E8M0 uint8 (+127 bias); subtract back to true exponent.
    max_exp_u = tl.load(packed_ptr + blk * stride_blk, mask=blk_mask, other=0)   # [BPP] uint8
    max_exp = max_exp_u.to(tl.int32) - 127

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * stride_blk + (PRIME_OFFSET + pcol[None, :]) * stride_col
    prime = tl.load(packed_ptr + offs_p, mask=mask, other=0)       # [BPP, NB] uint8

    q_offs = blk[:, None] * stride_blk + (Q_OFFSET + col[None, :]) * stride_col
    q_u8 = tl.load(packed_ptr + q_offs, mask=mask, other=0)        # uint8
    q = q_u8.to(tl.int8, bitcast=True)                             # bit-reinterpret -> int8

    y = _unpack_mx9(
        q, max_exp[:, None], prime, OUT_DTYPE,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )
    out_offs = blk[:, None] * BLOCK_SIZE + col[None, :]       # contiguous y offsets
    tl.store(y_ptr + out_offs, y, mask=mask)


# ============================================================================
# Host wrappers (not yet registered as @triton_op / register_fake; using bare
# launch for ease of round-trip verification). Single packed uint8 tensor
# in/out (mirrors mx6_quantization.py), not a three-tensor tuple.
# ============================================================================


def _mx9_packed_bytes(block_size: int) -> int:
    """Bytes per block for the packed layout [max_exp(1B) | prime | q(block_size)B]."""
    n_prime_bytes = (block_size // 2) // 8
    return 1 + n_prime_bytes + block_size


def convert_to_mx9(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
) -> torch.Tensor:
    """High-precision tensor -> packed MX9 tensor (single uint8 tensor, byte
    layout ``[max_exp(1B) | prime(1B) | q(block_size bytes)]`` per block).

    Blocks are formed along the ``axis`` dimension (axis is transposed to the last
    dim internally). The returned tensor is a flattened view:
      packed : [n_blocks, n_packed_bytes]   uint8
    Shape reconstruction info (original shape / axis / padding) must be passed
    back by the caller in convert_from_mx9. ``q`` is recoverable bit-exact via
    ``packed[..., 2:].view(torch.int8)`` (int8, clamp +/-127).
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16), \
        f"mx9_quantization only supports fp32 / bf16, got {data_hp.dtype}"
    assert block_size == 16, f"block_size only supports 16, got {block_size}"

    # Transpose quant axis to last dim, matching mxfp4 host processing.
    # No forced .contiguous(): reshape returns a (possibly non-contiguous) view
    # when possible, and the kernel gathers by stride. Padding of the ragged tail
    # block is handled in-kernel via column masking instead of F.pad.
    data_hp = data_hp.transpose(axis, -1)
    last = data_hp.shape[-1]
    x2d = data_hp.reshape(-1, last)
    rows = x2d.shape[0]
    blocks_per_row = triton.cdiv(last, block_size)
    n_blocks = rows * blocks_per_row

    n_packed_bytes = _mx9_packed_bytes(block_size)
    packed = torch.empty((n_blocks, n_packed_bytes), dtype=torch.uint8, device=data_hp.device)

    stride_row, stride_col = x2d.stride()
    grid = (triton.cdiv(n_blocks, blocks_per_program),)
    _convert_to_mx9_kernel[grid](
        x2d, packed, n_blocks, last, blocks_per_row,
        stride_row, stride_col,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROG=blocks_per_program,
        PRIME_GROUP=PRIME_GROUP,
        QUANT_BIT=QUANT_BIT,
    )
    return packed


def convert_from_mx9(
    packed: torch.Tensor,
    out_dtype: torch.dtype,
    out_shape,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
) -> torch.Tensor:
    """Packed MX9 tensor -> reconstructed high-precision tensor.

    out_shape / axis must match the original convert_to_mx9 call, used to
    reverse the transpose and padding. Returns shape = out_shape, dtype = out_dtype.
    """
    assert out_dtype in _TORCH_TO_TL, \
        f"out_dtype must be one of {tuple(_TORCH_TO_TL)}, got {out_dtype}"
    assert packed.dtype == torch.uint8, f"packed dtype must be uint8, got {packed.dtype}"
    assert block_size == 16, f"block_size only supports 16, got {block_size}"

    # Layout consistency check: packed must carry a whole number of blocks with
    # the expected per-block byte count; otherwise the kernel addressing would
    # read misaligned bytes or raise an unhelpful dimension error downstream.
    n_blocks = packed.shape[0]
    n_packed_bytes = _mx9_packed_bytes(block_size)
    assert packed.shape == (n_blocks, n_packed_bytes), \
        f"packed shape should be ({n_blocks}, {n_packed_bytes}), got {tuple(packed.shape)}"

    y_blocks = torch.empty((n_blocks, block_size), dtype=out_dtype, device=packed.device)

    # No forced .contiguous(): pass the packed input by stride so an already
    # laid-out (or non-contiguous view) tensor is consumed without an extra copy.
    stride_blk, stride_col = packed.stride()
    grid = (triton.cdiv(n_blocks, blocks_per_program),)
    _convert_from_mx9_kernel[grid](
        packed, y_blocks, n_blocks,
        stride_blk, stride_col,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROG=blocks_per_program,
        PRIME_GROUP=PRIME_GROUP,
        QUANT_BIT=QUANT_BIT,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
    )

    # Reverse: remove padding, reshape back to post-transpose shape, then
    # transpose back to original axis.
    # out_shape is the original (pre-transpose) shape; axis indicates which
    # dimension was the quant axis.
    transposed_shape = list(out_shape)
    transposed_shape[axis], transposed_shape[-1] = transposed_shape[-1], transposed_shape[axis]
    last = transposed_shape[-1]
    rows = 1
    for s in transposed_shape[:-1]:
        rows *= s
    pad = (block_size - last % block_size) % block_size
    padded_cols = last + pad
    # out_shape/axis must be consistent with convert_to: if the inferred block
    # count doesn't match the packed tensor, intercept here with a readable
    # error rather than letting the reshape below raise a cryptic dimension error.
    assert rows * padded_cols == n_blocks * block_size, (
        f"out_shape/axis/block_size inconsistent with packed tensor: "
        f"inferred rows*padded_cols={rows * padded_cols}, "
        f"but tuple has n_blocks*block_size={n_blocks * block_size}"
    )
    # Aligned with mxfp4 (convert_from_mxfp4): return the transposed stride view
    # without a forced .contiguous() copy. Only the ragged-tail slice reshape may
    # trigger an internal copy; the common (no-pad / axis=-1) path stays a view.
    y2d = y_blocks.reshape(rows, padded_cols)
    y2d = y2d[:, :last]
    return y2d.reshape(transposed_shape).transpose(axis, -1)
