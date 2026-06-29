# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Packed MX9 quantization Triton device kernels (called by grid kernels).

Unlike the fake-quant path (``alto/kernels/mx/quantize_triton.py`` which outputs
dequantized bf16), this module performs *real packed* quantization: each MX9 block
is compressed into a three-part tuple stored separately:
  - ``q``       : per-element quantized integer (int8, symmetric clamp +/-127)
  - ``max_exp`` : per-block shared exponent, uint8 E8M0 (floor(log2(amax)) + 127 bias)
  - ``prime``   : per-block prime bitmap (1 bit per pair of 2 elements, indicates
                  whether that pair gets a 1-exponent demotion)

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
    axis to dim -1 then calls .contiguous()).
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

Known limitations:
  - NaN: packed integers cannot represent NaN; blocks containing NaN will have
    garbage q values (only the fake-quant path supports NaN pass-through).
  - The acceptance reference is the custom clamp-127 variant (Quark-divergent),
    not Quark/mx.py itself.
"""

import torch
import torch.nn.functional as F
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
    operates on the original x, letting Inf be clamped and NaN pass through.)
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
    xf = x.to(tl.float32)
    q = _round_half_even(xf / scale)
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
# complete 16-element blocks. Inputs/outputs are addressed as flattened
# [n_blocks, BLOCK_SIZE] views; the three-part tuple layout:
#   q       : [n_blocks, BLOCK_SIZE]      (per-element, int8, clamp +/-127)
#   max_exp : [n_blocks]                  (per-block, uint8 E8M0, true exp + 127)
#   prime   : [n_blocks, N_PRIME_BYTES]   (per-block, N_PRIME_BYTES bytes, uint8 bitmap)
# ============================================================================


@triton.jit
def _convert_to_mx9_kernel(
    x_ptr,
    q_ptr,
    e_ptr,
    p_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)   # block indices for this program [BPP]
    col = tl.arange(0, BLOCK_SIZE)
    offs = blk[:, None] * BLOCK_SIZE + col[None, :]              # input/output q offsets [BPP, BLOCK]
    mask = blk[:, None] < n_blocks                              # tail block-level mask
    blk_mask = blk < n_blocks

    tl.static_assert(
        x_ptr.type.element_ty == tl.float32 or
        x_ptr.type.element_ty == tl.bfloat16,
        "x must be fp32 / bf16",
    )
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)            # native dtype tile

    shared_exp, max_exp, pair = _calculate_mx9_exp(
        x, BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP
    )
    q_int, prime = _pack_mx9(
        x, shared_exp, pair,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )

    tl.store(q_ptr + offs, q_int, mask=mask)
    # max_exp is the true exponent ([BPP] int32); store as E8M0 uint8: +127 bias.
    e_store = (max_exp + 127).to(tl.uint8)
    tl.store(e_ptr + blk, e_store, mask=blk_mask)

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * N_PRIME_BYTES + pcol[None, :]
    tl.store(p_ptr + offs_p, prime, mask=blk_mask[:, None])


@triton.jit
def _convert_from_mx9_kernel(
    q_ptr,
    e_ptr,
    p_ptr,
    y_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    N_PRIME_BYTES: tl.constexpr = (BLOCK_SIZE // PRIME_GROUP) // 8

    pid = tl.program_id(0)
    blk = pid * BLOCKS_PER_PROG + tl.arange(0, BLOCKS_PER_PROG)
    col = tl.arange(0, BLOCK_SIZE)
    offs = blk[:, None] * BLOCK_SIZE + col[None, :]
    mask = blk[:, None] < n_blocks
    blk_mask = blk < n_blocks

    q = tl.load(q_ptr + offs, mask=mask, other=0)              # int8
    # max_exp is stored as E8M0 uint8 (with +127 bias); subtract back to true exponent.
    max_exp_u = tl.load(e_ptr + blk, mask=blk_mask, other=0)   # [BPP] uint8
    max_exp = max_exp_u.to(tl.int32) - 127

    pcol = tl.arange(0, N_PRIME_BYTES)
    offs_p = blk[:, None] * N_PRIME_BYTES + pcol[None, :]
    prime = tl.load(p_ptr + offs_p, mask=blk_mask[:, None], other=0)          # [BPP, NB] uint8

    y = _unpack_mx9(
        q, max_exp[:, None], prime, OUT_DTYPE,
        BLOCKS_PER_PROG, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT,
    )
    tl.store(y_ptr + offs, y, mask=mask)


# ============================================================================
# Host wrappers (not yet registered as @triton_op / register_fake; using bare
# launch for ease of round-trip verification).
# ============================================================================


def convert_to_mx9(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
):
    """High-precision tensor -> packed MX9 three-part tuple (q:int8, max_exp:uint8
    E8M0, prime:uint8).

    Blocks are formed along the ``axis`` dimension (axis is transposed to the last
    dim internally). The returned three-part tuple is a flattened view:
      q       : [n_blocks, block_size]      int8 (clamp +/-127)
      max_exp : [n_blocks]                  uint8 (E8M0, true exponent + 127)
      prime   : [n_blocks, n_prime_bytes]   uint8
    Shape reconstruction info (original shape / axis / padding) must be passed
    back by the caller in convert_from_mx9.
    """
    assert data_hp.dtype in (torch.float32, torch.bfloat16), \
        f"mx9_quantization only supports fp32 / bf16, got {data_hp.dtype}"
    assert block_size == 16, f"block_size only supports 16, got {block_size}"

    # Transpose quant axis to last dim, matching mxfp4 host processing.
    data_hp = data_hp.transpose(axis, -1)
    last = data_hp.shape[-1]
    x2d = data_hp.contiguous().reshape(-1, last)
    pad = (block_size - last % block_size) % block_size
    if pad:
        x2d = F.pad(x2d, (0, pad))
    rows, cols = x2d.shape
    n_blocks = rows * (cols // block_size)
    x_blocks = x2d.reshape(n_blocks, block_size).contiguous()

    n_prime_bytes = (block_size // 2) // 8
    q = torch.empty((n_blocks, block_size), dtype=torch.int8, device=data_hp.device)
    max_exp = torch.empty((n_blocks,), dtype=torch.uint8, device=data_hp.device)
    prime = torch.empty((n_blocks, n_prime_bytes), dtype=torch.uint8, device=data_hp.device)

    grid = (triton.cdiv(n_blocks, blocks_per_program),)
    _convert_to_mx9_kernel[grid](
        x_blocks, q, max_exp, prime, n_blocks,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROG=blocks_per_program,
        PRIME_GROUP=PRIME_GROUP,
        QUANT_BIT=QUANT_BIT,
    )
    return q, max_exp, prime


def convert_from_mx9(
    q: torch.Tensor,
    max_exp: torch.Tensor,
    prime: torch.Tensor,
    out_dtype: torch.dtype,
    out_shape,
    block_size: int = BLOCK_SIZE,
    axis: int = -1,
    blocks_per_program: int = BLOCKS_PER_PROG_DEFAULT,
) -> torch.Tensor:
    """Packed MX9 three-part tuple -> reconstructed high-precision tensor.

    out_shape / axis must match the original convert_to_mx9 call, used to
    reverse the transpose and padding. Returns shape = out_shape, dtype = out_dtype.
    """
    assert out_dtype in _TORCH_TO_TL
    assert block_size == 16, f"block_size only supports 16, got {block_size}"

    # Three-part consistency check: q/max_exp/prime must share the same n_blocks
    # and shapes must match the storage spec; otherwise the kernel addressing /
    # reshape would read misaligned data or raise unhelpful dimension errors.
    n_blocks = q.shape[0]
    n_prime_bytes = (block_size // 2) // 8
    assert q.shape == (n_blocks, block_size), \
        f"q shape should be ({n_blocks}, {block_size}), got {tuple(q.shape)}"
    assert max_exp.shape == (n_blocks,), \
        f"max_exp shape should be ({n_blocks},), got {tuple(max_exp.shape)}"
    assert prime.shape == (n_blocks, n_prime_bytes), \
        f"prime shape should be ({n_blocks}, {n_prime_bytes}), got {tuple(prime.shape)}"

    y_blocks = torch.empty((n_blocks, block_size), dtype=out_dtype, device=q.device)

    grid = (triton.cdiv(n_blocks, blocks_per_program),)
    _convert_from_mx9_kernel[grid](
        q.contiguous(), max_exp.contiguous(), prime.contiguous(), y_blocks, n_blocks,
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
    # count doesn't match the three-part tuple, intercept here with a readable
    # error rather than letting the reshape below raise a cryptic dimension error.
    assert rows * padded_cols == n_blocks * block_size, (
        f"out_shape/axis/block_size inconsistent with three-part tuple: "
        f"inferred rows*padded_cols={rows * padded_cols}, "
        f"but tuple has n_blocks*block_size={n_blocks * block_size}"
    )
    y2d = y_blocks.reshape(rows, padded_cols)
    y2d = y2d[:, :last]
    return y2d.reshape(transposed_shape).transpose(axis, -1).contiguous()
