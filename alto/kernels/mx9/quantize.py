# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX9 fake-quantize kernel (block-wise emulation; no real packing/kernel inference).

This is a port of Quark's ``fake_quantize_mx6_mx9`` for the ``quant_bit=8`` (MX9)
case, with its helpers (``_t_exponent`` / ``_reshape_to_blocks`` / ``_pad_to_blocks``)
inlined here.

MX9 in a nutshell:
  1. Split the input along ``axis`` into blocks of ``block_size`` elements.
  2. Each block derives a shared exponent ``max_exp = floor(log2(amax))`` from its
     max absolute value.
  3. Every ``SHARED_PRIME_BIT_GROUP`` (=2) adjacent elements share one prime bit:
     if both elements in a pair sit at least one exponent below the block max, the
     pair uses ``max_exp - 1`` (finer resolution); otherwise ``max_exp``.
  4. The shared exponent yields a power-of-two scale; the block is then
     round -> clamp -> dequant.

The "9" is not ``quant_bit=9``: elements are still quantized to 8-bit integers, and
the extra "1" is the per-pair prime bit that decides the one-exponent demotion.
Intuitively: 8-bit signed integer value + a shared prime bit for exponent refinement.
  - ``QUANT_BIT = 8``: elements round to an 8-bit integer grid.
  - ``SHARED_PRIME_BIT_GROUP = 2``: every two adjacent elements share a prime bit.

This is fake quantization: the output keeps the input dtype (bf16/fp16/fp32) with
values projected onto the MX9-representable grid. The scale is computed from the
data at runtime; any externally supplied compressed_tensors scale is ignored here.
"""

import torch

from .format import BLOCK_SIZE, QUANT_BIT, SHARED_PRIME_BIT_GROUP


def _pad_to_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Right-pad the last dim with zeros to a multiple of ``block_size``."""
    pad = block_size - x.size(-1) % block_size
    if pad == block_size:
        return x
    return torch.nn.functional.pad(x, (0, pad))


def _reshape_to_blocks(x: torch.Tensor, block_size: int, axis: int) -> torch.Tensor:
    """Reshape into ``[num_rows, num_blocks, block_size]``.

    Blocking always happens along the last dim, so the requested ``axis`` is moved
    there first (e.g. weight ``[out, in]`` with axis=-1 blocks each output channel's
    input dim; activation ``[batch, seq, hidden]`` blocks each token's hidden vector).
    """
    if axis > x.dim() - 1:
        raise IndexError("Axis is larger than number of tensor dimensions")

    # Move the quantized axis to the last dim, flatten the rest, pad, then split
    # the last dim into blocks. Padding is trimmed after quantization.
    x = x.transpose(axis, -1)
    x = x.reshape(-1, x.size(-1))
    x = _pad_to_blocks(x, block_size)
    return x.reshape(x.size(0), x.size(1) // block_size, block_size)


def _exponent_frexp_no_exception(t: torch.Tensor) -> torch.Tensor:
    """Read the exponent straight from the float bit pattern (~ floor(log2(|x|)))."""
    with torch.no_grad():
        if t.dtype == torch.float32:
            return ((t.view(torch.int32) >> 23) & 0xFF) - 127
        if t.dtype == torch.bfloat16:
            return ((t.view(torch.int16) >> 7) & 0xFF) - 127
        if t.dtype == torch.float16:
            # Matches Quark: fp16 exponent bias treated as -15.
            return ((t.view(torch.int16) >> 10) & 0x1F) - 15
        raise ValueError(f"Unsupported data type: {t.dtype}")


def _t_exponent(t: torch.Tensor) -> torch.Tensor:
    """Per-element exponent; NaN/Inf are zeroed before extraction."""
    with torch.no_grad():
        t = torch.nan_to_num(t, nan=0, posinf=0, neginf=0)
        return _exponent_frexp_no_exception(t)


def mx9_fake_quantize(
    input_tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    quant_bit: int = QUANT_BIT,
    axis: int = -1,
) -> torch.Tensor:
    """Block-wise MX9 fake quantization (QDQ), bit-exact with Quark's
    ``fake_quantize_mx6_mx9(quant_bit=8)``.

    Args:
        input_tensor: tensor to quantize (weight or activation).
        block_size: elements per MX9 block.
        quant_bit: element integer width (8 for MX9, 5 for MX6).
        axis: axis to block along. Only ``axis=-1`` is supported on the ALTO path.
    """
    if axis != -1:
        raise NotImplementedError("mx9_fake_quantize supports axis=-1 only")

    input_dtype = input_tensor.dtype

    # Shape to restore after blocking: _reshape_to_blocks moves axis to the last dim,
    # so record the post-transpose shape to rebuild before transposing back.
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    # Detached block view used only for the per-block max exponent (no gradient).
    block_x = _reshape_to_blocks(input_tensor.detach(), block_size, axis)
    # Out-of-place nan_to_num: in-place breaks torch.compile/inductor functional graphs.
    block_x = torch.nan_to_num(block_x, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-block max abs -> shared block exponent max_exp = floor(log2(amax)).
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    max_exp = _t_exponent(amax)  # [num_rows, num_blocks, 1]

    # Block the original input for the per-element QDQ.
    input_tensor = _reshape_to_blocks(input_tensor, block_size, axis)
    t_exp = _t_exponent(input_tensor)  # [num_rows, num_blocks, block_size]

    # Per element: is it at least one exponent below its block max? (max_exp broadcasts)
    demote = max_exp - t_exp >= 1

    # Prime bit: a pair is demoted only if BOTH its elements are below the block max.
    # Reshape last dim into pairs, AND within each pair, then broadcast back.
    n = SHARED_PRIME_BIT_GROUP
    assert demote.shape[-1] % n == 0
    flat_shape = demote.shape
    demote = demote.reshape(*flat_shape[:-1], flat_shape[-1] // n, n)
    demote = torch.sum(demote, -1, keepdim=True) == n
    demote = demote.repeat(*([1] * (demote.dim() - 1)), n).reshape(flat_shape)

    # Demoted pairs drop one exponent (the prime-bit refinement).
    shared_exp = max_exp - demote.long()

    # Final power-of-two scale: 2 ** (shared_exp - quant_bit + 2)  (=2**(shared_exp-6) for MX9).
    scale = torch.pow(2.0, shared_exp - quant_bit + 2)

    # Max representable integer magnitude. clamp_max guards against float overflow
    # when max_exp is large: quant_max = (2**(max_exp+1) - scale) / scale.
    quant_max = (
        torch.clamp_max(
            torch.pow(2.0, max_exp.to(torch.float64) + 1) - scale,
            torch.finfo(torch.float32).max,
        ).to(torch.float32)
        / scale
    )

    # QDQ: round to the integer grid, clamp to range, dequant back to float.
    output_tensor = torch.round(input_tensor / scale)
    output_tensor = torch.clamp(output_tensor, -quant_max, quant_max) * scale

    # Flatten blocks, drop padding, restore shape/dtype, and transpose axis back.
    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return output_tensor.transpose(axis, -1)
