# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Shared MX packed-quantization device helpers plus the ``convert_*`` entry points.

MX6 and MX9 share the same block algorithm -- exponent extraction, prime/pair
demotion, and the power-of-two scale (with its -126 clamp) -- and differ only in
element integer width (``quant_bit`` = 8 for MX9, 5 for MX6) and byte packing.
The width-/packing-independent pieces are centralised here so both formats use
identical math; only the per-format ``_pack_*`` / ``_unpack_*``, the grid
kernels, and the byte layout stay in their own modules.

``convert_to_mx`` / ``convert_from_mx`` dispatch on a ``target_dtype`` string,
resolving that format's kernels and packing constants internally.
"""

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl

_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def _floor_exp(x):
    """Extract the unbiased exponent field directly from native float bits."""
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
    """Round-to-nearest-ties-to-even (matches torch.round)."""
    rounded = tl.floor(y + 0.5)
    is_tie = (y - tl.floor(y)) == 0.5
    is_odd = (rounded - 2.0 * tl.floor(rounded * 0.5)) == 1.0
    return tl.where(is_tie & is_odd, rounded - 1.0, rounded)


@triton.jit
def _block_coords(blk, blocks_per_row, BLOCK_SIZE: tl.constexpr):
    """Map block ids to their coordinates in the caller's 2D [rows, last] tensor.

    Block b covers columns [brow*BLOCK_SIZE, +BLOCK_SIZE) of row b //
    blocks_per_row. Returns (row [BPP], col_g [BPP, BLOCK_SIZE]); the tail
    block's col_g runs past ``last``, so callers mask on it.
    """
    row = blk // blocks_per_row
    brow = blk % blocks_per_row
    col_g = brow[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    return row, col_g


@triton.jit
def _calculate_mx_exp(
    x,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
):
    """Per-block shared exponent, per-element shared_exp, and per-pair prime bits.

    x is [BPP, BLOCK_SIZE] in native dtype; do NOT pre-cast to fp32 on the host
    since exponent extraction must match the per-element branch in _floor_exp.

    Returns (shared_exp [BPP, BLOCK_SIZE], max_exp [BPP], pair [BPP, N_PAIRS]),
    all int32. pair[i,j]=1 means that pair gets a 1-exponent demotion.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP

    # Sanitize the exponent statistics only. Quantization keeps the original x,
    # so Inf is clamped rather than replaced by 0.
    clean = tl.where(x != x, 0.0, x)
    clean = tl.where(tl.abs(clean) == float("inf"), 0.0, clean)

    # tl.max on some backends promotes the reduce to fp32; cast back to native
    # dtype so _floor_exp takes the same branch as the per-element call below.
    amax = tl.max(tl.abs(clean), axis=1, keep_dims=True).to(clean.dtype)
    max_exp = _floor_exp(amax)

    t_exp = _floor_exp(clean)
    demote = (max_exp - t_exp) >= 1

    demote_groups = demote.to(tl.int32).reshape(BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP)
    pair_demote = tl.sum(demote_groups, axis=2, keep_dims=True) == PRIME_GROUP

    pair_b = tl.broadcast_to(pair_demote, (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b.to(tl.int32)

    pair = pair_demote.reshape(BLOCKS_PER_PROG, N_PAIRS).to(tl.int32)
    return shared_exp, max_exp.reshape(BLOCKS_PER_PROG), pair


@triton.jit
def _scale_from_shared_exp(shared_exp, QUANT_BIT: tl.constexpr):
    """2^(shared_exp - quant_bit + 2), scale exponent clamped to -126.

    Clamping to the minimum normal fp32 exponent keeps the scale nonzero and
    avoids backend-dependent subnormal/FTZ behavior.
    """
    scale_exp = tl.maximum(shared_exp - QUANT_BIT + 2, -126)
    return tl.exp2(scale_exp.to(tl.float32))


@triton.jit
def _quantize_clamped(x, shared_exp, QUANT_BIT: tl.constexpr, Q_MAX: tl.constexpr):
    """Round-half-even to [-Q_MAX, Q_MAX] and return fp32 q for format packing."""
    scale = _scale_from_shared_exp(shared_exp, QUANT_BIT)
    q = _round_half_even(tl.div_rn(x.to(tl.float32), scale))
    return tl.minimum(tl.maximum(q, -(Q_MAX * 1.0)), Q_MAX * 1.0)


@triton.jit
def _pack_bits8(bits, BLOCKS_PER_PROG: tl.constexpr, N_BYTES: tl.constexpr):
    """Pack a [BPP, N_BYTES*8] 0/1 tensor into [BPP, N_BYTES] uint8, LSB first."""
    weights = (1 << tl.arange(0, 8)).to(tl.int32)
    g = bits.reshape(BLOCKS_PER_PROG, N_BYTES, 8)
    return tl.sum(g * weights[None, None, :], axis=2).to(tl.uint8)


@triton.jit
def _unpack_bits8(packed, BLOCKS_PER_PROG: tl.constexpr, N_BYTES: tl.constexpr):
    """Inverse of ``_pack_bits8``: [BPP, N_BYTES] uint8 -> [BPP, N_BYTES*8] 0/1 int32."""
    shifts = tl.arange(0, 8).to(tl.int32)
    p = packed.to(tl.int32).reshape(BLOCKS_PER_PROG, N_BYTES, 1)  # nonnegative, so >> is safe
    bits = (p >> shifts[None, None, :]) & 1
    return bits.reshape(BLOCKS_PER_PROG, N_BYTES * 8)


@triton.jit
def _dequantize_decoded_q(
    q,
    pair,
    max_exp,
    out_dtype: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr,
    QUANT_BIT: tl.constexpr,
):
    """Reconstruct values from an already-decoded integer ``q`` and per-pair
    ``pair`` demotion bitmap: rebuild per-element shared_exp, then
    y = q * 2^(shared_exp - quant_bit + 2).

    ``q`` may be int8 (MX9, straight from storage) or int32 (MX6, rebuilt from
    sign + mantissa nibble); both are cast to fp32 here before scaling.
    """
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    pair_b = tl.broadcast_to(pair[:, :, None], (BLOCKS_PER_PROG, N_PAIRS, PRIME_GROUP))
    pair_b = pair_b.reshape(BLOCKS_PER_PROG, BLOCK_SIZE)
    shared_exp = max_exp - pair_b
    scale = _scale_from_shared_exp(shared_exp, QUANT_BIT)
    return (q.to(tl.float32) * scale).to(out_dtype)


class _MxFormat(NamedTuple):
    """Per-format constants and kernels resolved from a ``target_dtype``.

    block_size / prime_group / blocks_per_program happen to agree between MX6
    and MX9, but are read per format rather than shared.
    """
    block_size: int
    prime_group: int
    quant_bit: int
    n_packed_bytes: int
    blocks_per_program: int
    to_kernel: object
    from_kernel: object


def _mx_format(target_dtype: str) -> _MxFormat:
    """Resolve ``target_dtype`` to its kernels and packing constants."""
    if target_dtype == "mx6":
        from . import mx6_quantization as mod

        to_kernel, from_kernel = mod._convert_to_mx6_kernel, mod._convert_from_mx6_kernel
    elif target_dtype == "mx9":
        from . import mx9_quantization as mod

        to_kernel, from_kernel = mod._convert_to_mx9_kernel, mod._convert_from_mx9_kernel
    else:
        raise ValueError(f"target_dtype must be 'mx6' or 'mx9', got {target_dtype!r}")

    return _MxFormat(
        mod.BLOCK_SIZE,
        mod.PRIME_GROUP,
        mod.QUANT_BIT,
        mod.N_PACKED_BYTES,
        mod.BLOCKS_PER_PROG_DEFAULT,
        to_kernel,
        from_kernel,
    )


def convert_to_mx(data_hp: torch.Tensor, target_dtype: str, axis: int = -1) -> torch.Tensor:
    """High-precision tensor -> packed ``[n_blocks, n_packed_bytes]`` uint8 tensor.

    ``target_dtype`` ("mx6" / "mx9") is the only thing the caller chooses about
    the format: element width, block size, byte layout and launch geometry are
    fixed per format and resolved here.

    Blocks are formed along ``axis``, which is transposed to the last dim
    internally. The original shape / axis are not stored, so the caller must pass
    them back to convert_from_mx.

    Launches a Triton kernel, so ``data_hp`` must be on a CUDA/HIP device. No
    ``.contiguous()`` is forced: ``reshape`` keeps a view where it can and the
    kernel gathers by stride, masking the ragged tail block.
    """
    spec = _mx_format(target_dtype)
    assert data_hp.dtype in (torch.float32, torch.bfloat16), \
        f"{target_dtype} only supports fp32 / bf16, got {data_hp.dtype}"

    data_hp = data_hp.transpose(axis, -1)
    last = data_hp.shape[-1]
    assert last > 0, f"{target_dtype} requires a non-empty quantization axis"
    x2d = data_hp.reshape(-1, last)
    rows = x2d.shape[0]
    blocks_per_row = triton.cdiv(last, spec.block_size)
    n_blocks = rows * blocks_per_row

    packed = torch.empty((n_blocks, spec.n_packed_bytes), dtype=torch.uint8, device=data_hp.device)

    stride_in_row, stride_in_col = x2d.stride()
    stride_out_blk, stride_out_byte = packed.stride()
    grid = (triton.cdiv(n_blocks, spec.blocks_per_program),)
    spec.to_kernel[grid](
        x2d,
        packed,
        n_blocks,
        last,
        blocks_per_row,
        stride_in_row,
        stride_in_col,
        stride_out_blk,
        stride_out_byte,
        BLOCK_SIZE=spec.block_size,
        BLOCKS_PER_PROG=spec.blocks_per_program,
        PRIME_GROUP=spec.prime_group,
        QUANT_BIT=spec.quant_bit,
    )
    return packed


def convert_from_mx(
    packed: torch.Tensor,
    target_dtype: str,
    out_dtype: torch.dtype,
    out_shape,
    axis: int = -1,
) -> torch.Tensor:
    """Packed MX tensor -> reconstructed high-precision tensor.

    ``target_dtype``, ``out_shape`` and ``axis`` must exactly match the
    convert_to_mx call that produced ``packed``; none of them are stored in the
    packed bytes. Some mismatches have the same block count and cannot be
    detected, so violating this requirement may silently reorder values.

    Both tensors are addressed by stride, so an existing non-contiguous ``packed``
    view is consumed without a copy and the kernel scatters straight into the
    ``[rows, last]`` output.
    """
    spec = _mx_format(target_dtype)
    assert out_dtype in _TORCH_TO_TL, \
        f"out_dtype must be one of {tuple(_TORCH_TO_TL)}, got {out_dtype}"
    assert packed.dtype == torch.uint8, f"packed dtype must be uint8, got {packed.dtype}"
    assert packed.ndim == 2 and packed.shape[1] == spec.n_packed_bytes, \
        f"{target_dtype} packed tensor must be [n_blocks, {spec.n_packed_bytes}], got {tuple(packed.shape)}"

    n_blocks = packed.shape[0]
    transposed_shape = list(out_shape)
    assert transposed_shape, f"{target_dtype} out_shape must have at least one dimension"
    assert -len(transposed_shape) <= axis < len(transposed_shape), \
        f"axis {axis} is out of range for out_shape={tuple(out_shape)}"
    transposed_shape[axis], transposed_shape[-1] = transposed_shape[-1], transposed_shape[axis]
    last = transposed_shape[-1]
    assert last > 0, f"{target_dtype} requires a non-empty quantization axis"
    rows = math.prod(transposed_shape[:-1])
    blocks_per_row = triton.cdiv(last, spec.block_size)
    assert rows * blocks_per_row == n_blocks, (f"out_shape/axis inconsistent with packed tensor: "
                                               f"inferred rows*blocks_per_row={rows * blocks_per_row}, "
                                               f"but packed holds n_blocks={n_blocks}")

    y2d = torch.empty((rows, last), dtype=out_dtype, device=packed.device)

    stride_in_blk, stride_in_byte = packed.stride()
    stride_out_row, stride_out_col = y2d.stride()
    grid = (triton.cdiv(n_blocks, spec.blocks_per_program),)
    spec.from_kernel[grid](
        packed,
        y2d,
        n_blocks,
        last,
        blocks_per_row,
        stride_in_blk,
        stride_in_byte,
        stride_out_row,
        stride_out_col,
        BLOCK_SIZE=spec.block_size,
        BLOCKS_PER_PROG=spec.blocks_per_program,
        PRIME_GROUP=spec.prime_group,
        QUANT_BIT=spec.quant_bit,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
    )
    return y2d.reshape(transposed_shape).transpose(axis, -1)
