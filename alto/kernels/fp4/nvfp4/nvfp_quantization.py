# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from alto.kernels.fp4.fp4_common import (
    make_dequantize_e2m1,
    make_generate_philox_randval_2x,
    make_quantize_e2m1,
)


BLOCK_SIZE_DEFAULT = 16
F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny

# Naming convention for the NVFP4 scale hierarchy:
#   inner_scale -- per-block scale stored alongside the packed FP4 data
#                  (NVFP4 spec ``s_block``).  Value lives on the E4M3 grid
#                  in an FP32 container (see outer-side notes below).
#   outer_scale -- the outer-level scale factor that sits above
#                  ``inner_scale`` (NVFP4 spec ``s_global``).  Currently a
#                  per-tensor FP32 scalar; the name is intentionally
#                  agnostic so the same API can later carry an
#                  outer-blockwise layout (e.g. one scale per 128x128
#                  tile) without renaming the public surface.
#
# Per spec, ``outer_scale`` lives in FP32; this floor is a div-by-zero
# guard for the downstream ``max_abs / outer_scale`` when ``amax == 0``.
# We pick ``1e-30`` (well above FP32 denormal range and ~22 orders below
# any natural training-time outer scale) so that the effective per-block
# divisor ``quant_scale = inner_scale * outer_scale`` stays an FP32
# *normal* in the worst case (``inner_scale == E4M3_EPS``,
# ``outer_scale == _OUTER_SCALE_DIVZERO_FLOOR``):
#   1e-30 * 2**-6 ≈ 1.56e-32  >  FP32 smallest normal (~1.18e-38)
_OUTER_SCALE_DIVZERO_FLOOR = 1.0e-30

SUPPORTED_SCALE_FORMATS = ("e4m3",)


def _check_scale_format(scale_format: str) -> None:
    if scale_format != "e4m3":
        raise NotImplementedError(
            f"scale_format={scale_format!r} is not yet supported. "
            f"Currently supported: {SUPPORTED_SCALE_FORMATS}"
        )
_dequantize_e2m1 = make_dequantize_e2m1()
_generate_philox_randval_2x = make_generate_philox_randval_2x()
_quantize_e2m1 = make_quantize_e2m1()


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


@triton.jit
def _pack_fp4(
    x,
    scales_fp32,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Quantize and pack a tile into nibble-packed uint8.

    Mirrors the API of ``mxfp4._pack_fp4``.  The key difference is that
    *scales_fp32* is already a per-block **float32** tensor, whereas MXFP4
    passes uint8 exponents and converts them internally.

    ``USE_ASM`` is accepted for API parity but has no effect — CDNA4 FP4
    ASM instructions only honour the biased exponent of the scale operand,
    making them incompatible with NVFP4's general float32 scales.

    Args:
        x:            input tile ``[BLOCK_M, BLOCK_N]``  (float32 | bfloat16)
        scales_fp32:  per-block float32 scales.
                      1D: ``[BLOCK_M, SCALE_BLOCK_N]``;
                      2D: ``[SCALE_BLOCK_M, SCALE_BLOCK_N]``.
        BLOCK_M / BLOCK_N / QUANT_BLOCK_SIZE: tile constants
        IS_2D_BLOCK:  if True, use 2D (square) blocks for scaling
        USE_SR:       enable stochastic rounding
        USE_ASM:      (no-op) kept for API consistency with MXFP4

    Returns:
        packed uint8 tile ``[BLOCK_M, HALF_BLOCK_N]``
    """
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    if IS_2D_BLOCK:
        scales_bc = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
            SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        scales_bc = scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)

    x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

    if USE_SR:
        randval0, randval1 = _generate_philox_randval_2x(
            BLOCK_M, HALF_BLOCK_N, philox_seed, philox_offset)
    else:
        randval0 = 0
        randval1 = 0

    y0 = _quantize_e2m1(x0, scales_bc, randval0, USE_SR=USE_SR)
    y1 = _quantize_e2m1(x1, scales_bc, randval1, USE_SR=USE_SR)
    y = y0 | (y1 << 4)

    return y.to(tl.uint8)


@triton.jit
def _unpack_fp4(
    x,
    scales_fp32,
    output_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Unpack and dequantize a nibble-packed uint8 tile back to float.

    Mirrors the API of ``mxfp4._unpack_fp4``.  *scales_fp32* is a per-block
    **float32** tensor; see :func:`_pack_fp4` for details on the
    MXFP4/NVFP4 scale difference.

    ``USE_ASM`` is accepted for API parity but has no effect.

    Args:
        x:            packed uint8 tile ``[BLOCK_M, HALF_BLOCK_N]``
        scales_fp32:  per-block float32 scales.
                      1D: ``[BLOCK_M, SCALE_BLOCK_N]``;
                      2D: ``[SCALE_BLOCK_M, SCALE_BLOCK_N]``.
        output_dtype: target element type (tl.float32 | tl.bfloat16).
                      Kept for API parity with MXFP4; the software path
                      always computes in float32.
        BLOCK_M / BLOCK_N / QUANT_BLOCK_SIZE: tile constants
        IS_2D_BLOCK:  if True, use 2D (square) blocks for scaling
        USE_ASM:      (no-op) kept for API consistency with MXFP4

    Returns:
        unpacked float32 tile ``[BLOCK_M, BLOCK_N]``
    """
    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    if IS_2D_BLOCK:
        scales_bc = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
            SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)
    else:
        scales_bc = scales_fp32.expand_dims(axis=2).broadcast_to(
            BLOCK_M, SCALE_BLOCK_N,
            HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)

    x0 = x & 0xF
    x1 = (x & 0xF0) >> 4
    y0 = _dequantize_e2m1(x0, scales_bc)
    y1 = _dequantize_e2m1(x1, scales_bc)

    y = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
    return y


# ---- scale calculation (NVFP4-specific) -----------------------------------

@triton.jit
def _calculate_nvfp4_scales(
    x,
    outer_scale_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_OUTER_SCALE: tl.constexpr = False,
):
    """Compute per-block E4M3-quantised scales for NVFP4 quantization.

    The returned ``inner_scale`` is the NVFP4 spec ``s_block``; the
    ``outer_scale`` operand is the spec ``s_global``.  See the
    naming-convention block at the top of this module for the rationale.

    Per spec, the only value stored as FP8 E4M3 is the per-block scale
    written next to the packed FP4 data; the outer scale and intermediates
    stay in FP32.  With ``USE_OUTER_SCALE=True`` the spec order is::

        inner_scale_raw = block_amax(x) / outer_scale / F4_E2M1_MAX
        inner_scale     = round_e4m3(clamp(inner_scale_raw,
                                           [E4M3_EPS, F8E4M3_MAX]))
        quant_scale     = inner_scale * outer_scale

    i.e. clamp + E4M3 round are applied exactly once, on the final stored
    block (inner) scale.

    When ``IS_2D_BLOCK`` is True, one scale covers a
    ``QUANT_BLOCK_SIZE x QUANT_BLOCK_SIZE`` tile, yielding output shapes
    ``[SCALE_BLOCK_M, SCALE_BLOCK_N]`` instead of ``[BLOCK_M, SCALE_BLOCK_N]``.
    """
    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x_grouped = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE,
                              NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x_grouped), axis=-1)
        max_abs = tl.max(max_abs, axis=-2).to(tl.float32)
    else:
        x_grouped = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x_grouped), axis=-1).to(tl.float32)

    if USE_OUTER_SCALE:
        outer_scale = tl.load(outer_scale_ptr)
        inner_scale_raw = max_abs / outer_scale / F4_E2M1_MAX
        inner_scale_raw = tl.minimum(tl.maximum(inner_scale_raw, E4M3_EPS), F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(tl.float8e4nv).to(tl.float32)
        quant_scale = inner_scale * outer_scale
    else:
        inner_scale_raw = max_abs / F4_E2M1_MAX
        inner_scale_raw = tl.minimum(tl.maximum(inner_scale_raw, E4M3_EPS), F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(tl.float8e4nv).to(tl.float32)
        quant_scale = inner_scale

    return inner_scale, quant_scale


# ---- top-level Triton kernels ---------------------------------------------

@triton.jit
def _convert_to_nvfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    outer_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    philox_seed,
    philox_offset,
    M_ACTUAL,
    N_ACTUAL,
    PACKED_N_ACTUAL,
    SCALE_M_ACTUAL,
    SCALE_N_ACTUAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_OUTER_SCALE: tl.constexpr,
    USE_SR: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_yn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    tl.static_assert(
        (x_ptr.type.element_ty == tl.float32) | (x_ptr.type.element_ty == tl.bfloat16)
    )
    x = tl.load(
        x_ptr + offs_x,
        mask=(offs_m[:, None] < M_ACTUAL) & (offs_xn[None, :] < N_ACTUAL),
        other=0,
    )

    inner_scale, quant_scale = _calculate_nvfp4_scales(
        x,
        outer_scale_ptr,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_OUTER_SCALE=USE_OUTER_SCALE,
    )

    y = _pack_fp4(
        x,
        quant_scale,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_SR=USE_SR,
    )

    tl.store(
        y_ptr + offs_y,
        y.to(y_ptr.type.element_ty),
        mask=(offs_m[:, None] < M_ACTUAL) & (offs_yn[None, :] < PACKED_N_ACTUAL),
    )
    tl.store(
        s_ptr + offs_s,
        inner_scale,
        mask=(offs_sm[:, None] < SCALE_M_ACTUAL) & (offs_sn[None, :] < SCALE_N_ACTUAL),
    )


@triton.jit
def _convert_from_nvfp4_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    outer_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M_ACTUAL,
    N_ACTUAL,
    PACKED_N_ACTUAL,
    SCALE_M_ACTUAL,
    SCALE_N_ACTUAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_OUTER_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
    SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * HALF_BLOCK_N + tl.arange(0, HALF_BLOCK_N)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    if IS_2D_BLOCK:
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
    else:
        offs_sm = offs_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(
        x_ptr + offs_x,
        mask=(offs_m[:, None] < M_ACTUAL) & (offs_xn[None, :] < PACKED_N_ACTUAL),
        other=0,
    )
    s = tl.load(
        s_ptr + offs_s,
        mask=(offs_sm[:, None] < SCALE_M_ACTUAL) & (offs_sn[None, :] < SCALE_N_ACTUAL),
        other=0,
    )

    if USE_OUTER_SCALE:
        outer_scale = tl.load(outer_scale_ptr)
        s = s * outer_scale

    y = _unpack_fp4(
        x,
        s,
        y_ptr.type.element_ty,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    tl.store(
        y_ptr + offs_y,
        y.to(y_ptr.type.element_ty),
        mask=(offs_m[:, None] < M_ACTUAL) & (offs_yn[None, :] < N_ACTUAL),
    )


def compute_dynamic_outer_scale(
    data_hp: torch.Tensor,
    scale_format: str = "e4m3",
) -> torch.Tensor:
    """Compute the FP32 outer-level scale ``amax / (F8E4M3_MAX * F4_E2M1_MAX)``.

    The "outer" naming reflects this scalar's position in the NVFP4
    hierarchy: it sits above the per-block ``inner_scale`` and is shared
    across the entire tensor today (NVFP4 spec ``s_global``).  Future
    extensions may produce one outer scale per outer-block tile; this
    function and the surrounding API are named to accommodate that
    without further renames.

    Per spec, the outer scale stays in FP32 with only a
    ``_OUTER_SCALE_DIVZERO_FLOOR`` div-by-zero guard (``amax == 0``).

    The *scale_format* parameter is reserved for future scale representations
    (e.g. ``"e5m3"``).  Currently only ``"e4m3"`` is supported.
    """
    _check_scale_format(scale_format)
    amax = data_hp.float().abs().max()
    outer_scale = (amax / (F8E4M3_MAX * F4_E2M1_MAX)).clamp(min=_OUTER_SCALE_DIVZERO_FLOOR)
    return outer_scale.to(dtype=torch.float32).reshape(1)


@triton_op("alto::convert_to_nvfp4", mutates_args={})
def convert_to_nvfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    update_outer_scale: bool = True,
    scale_format: str = "e4m3",
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a high-precision tensor to NVFP4 (E2M1) format.

    Outer-level (NVFP4 spec ``s_global``) scaling is controlled by
    ``outer_scale`` and ``update_outer_scale``.  ``outer_scale`` is today a
    1-element FP32 tensor representing a per-tensor scale; the parameter is
    deliberately named ``outer_scale`` (not ``per_tensor_scale``) so a future
    outer-blockwise layout can reuse the same surface without renaming.

    * ``outer_scale`` given, ``update_outer_scale=True`` (default):
      recompute the scale from ``data_hp``'s amax and write it back into the
      caller's tensor **in place** (no clone).  The caller reads the updated
      value back through the same tensor.  This is the recommended path for
      training, where the outer scale tracks the current tensor's range.
    * ``outer_scale`` given, ``update_outer_scale=False``:
      use the caller-provided scale as-is (for calibrated / frozen scales).
    * ``outer_scale=None``, ``update_outer_scale=True`` (default):
      compute a dynamic scale internally and apply it for this call only.
      The scale is not returned; if the caller wants to track it across calls
      they should pre-allocate a buffer and pass it in.
    * ``outer_scale=None``, ``update_outer_scale=False``:
      outer-level scaling is disabled.

    *scale_format* selects the per-block scale representation.  Currently
    supported: ``"e4m3"`` (default).  ``"e5m3"`` is accepted for forward
    compatibility but is not yet validated on hardware.

    ``use_asm`` is accepted for API consistency with MXFP4 but has no effect.
    """
    torch._check(
        data_hp.shape[axis] % block_size == 0,
        lambda: f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}",
    )
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    assert block_size % 2 == 0 and block_size >= 2, (
        f"block_size must be a positive even number, got {block_size}"
    )
    assert not is_2d_block or data_hp.size(-2) % block_size == 0, (
        f"2D block requires dim -2 ({data_hp.size(-2)}) divisible by block_size ({block_size})"
    )
    _check_scale_format(scale_format)

    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, ori_shape[-1])

    new_shape = (*ori_shape[:-1], ori_shape[-1] // 2)
    if is_2d_block:
        scale_shape = (*ori_shape[:-2], ori_shape[-2] // block_size,
                       ori_shape[-1] // block_size)
    else:
        scale_shape = (*ori_shape[:-1], ori_shape[-1] // block_size)
    data_lp = torch.empty(new_shape, dtype=torch.uint8, device=data_hp.device).reshape(
        -1, new_shape[-1]
    )
    scales = torch.empty(scale_shape, dtype=torch.float32, device=data_hp.device).reshape(
        -1, scale_shape[-1]
    )

    # Resolve outer-scale I/O on the caller's own buffer — no clone.
    if outer_scale is not None:
        assert outer_scale.numel() == 1, "outer_scale must be a scalar tensor"
        assert outer_scale.dtype == torch.float32, (
            "outer_scale must be float32"
        )
        assert outer_scale.device == data_hp.device, (
            f"outer_scale device ({outer_scale.device}) must match "
            f"data_hp device ({data_hp.device})"
        )
        if update_outer_scale:
            outer_scale.copy_(
                compute_dynamic_outer_scale(data_hp).reshape_as(outer_scale)
            )
        outer_scale_buf = outer_scale.reshape(())
        use_outer_scale = True
    elif update_outer_scale:
        # No buffer supplied — compute an ephemeral scale for this call.
        outer_scale_buf = compute_dynamic_outer_scale(data_hp).reshape(())
        use_outer_scale = True
    else:
        outer_scale_buf = torch.ones((), dtype=torch.float32, device=data_hp.device)
        use_outer_scale = False

    stride_xm, stride_xn = data_hp.stride()
    stride_ym, stride_yn = data_lp.stride()
    stride_sm, stride_sn = scales.stride()

    M, N = data_hp.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N

    if philox_seed is None:
        philox_seed = torch.randint(0, 2**31 - 1, (1,)).item()
    if philox_offset is None:
        philox_offset = torch.randint(0, 2**31 - 1, (1,)).item()

    wrap_triton(_convert_to_nvfp4_kernel)[grid](
        data_hp,
        data_lp,
        scales,
        outer_scale_buf,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        philox_seed,
        philox_offset,
        M,
        N,
        data_lp.shape[1],
        scales.shape[0],
        scales.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_OUTER_SCALE=use_outer_scale,
        USE_SR=use_sr,
    )

    return (
        data_lp.reshape(new_shape).transpose(axis, -1),
        scales.reshape(scale_shape).transpose(axis, -1),
    )


@triton_op("alto::convert_from_nvfp4", mutates_args={})
def convert_from_nvfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    scale_format: str = "e4m3",
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    """Dequantize NVFP4 (E2M1) data back to high-precision format.

    ``outer_scale`` is the optional outer-level FP32 scalar (NVFP4 spec
    ``s_global``); see :func:`convert_to_nvfp4` for the naming rationale.

    *scale_format* is accepted for API symmetry with :func:`convert_to_nvfp4`.
    The dequantization path only multiplies by the stored float32 scale, so
    the format parameter does not affect the computation.

    ``use_asm`` is accepted for API consistency with MXFP4 but has no effect.
    """
    assert output_dtype in [torch.float32, torch.bfloat16]
    assert block_size % 2 == 0 and block_size >= 2, (
        f"block_size must be a positive even number, got {block_size}"
    )

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape_lp = data_lp.shape

    data_lp = data_lp.reshape(-1, orig_shape_lp[-1])
    scales = scales.reshape(-1, (orig_shape_lp[-1] * 2) // block_size).to(torch.float32)
    orig_shape_hp = (*orig_shape_lp[:-1], orig_shape_lp[-1] * 2)
    data_hp = data_lp.new_empty(orig_shape_hp, dtype=output_dtype).reshape(-1, orig_shape_hp[-1])

    if outer_scale is None:
        outer_scale_buf = torch.ones((), dtype=torch.float32, device=data_lp.device)
        use_outer_scale = False
    else:
        outer_scale = outer_scale.to(device=data_lp.device, dtype=torch.float32)
        assert outer_scale.numel() == 1, "outer_scale must be a scalar tensor"
        outer_scale_buf = outer_scale.reshape(())
        use_outer_scale = True

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_hp.shape

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N

    wrap_triton(_convert_from_nvfp4_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        outer_scale_buf,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        data_lp.shape[1],
        scales.shape[0],
        scales.shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_OUTER_SCALE=use_outer_scale,
    )

    return data_hp.reshape(orig_shape_hp).transpose(axis, -1)


@convert_to_nvfp4.register_fake
def _fake_convert_to_nvfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    update_outer_scale: bool = True,
    scale_format: str = "e4m3",
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    new_shape = (*orig_shape[:-1], orig_shape[-1] // 2)
    if is_2d_block:
        scale_shape = (*orig_shape[:-2], orig_shape[-2] // block_size,
                       orig_shape[-1] // block_size)
    else:
        scale_shape = (*orig_shape[:-1], orig_shape[-1] // block_size)
    data_lp = data_hp.new_empty(new_shape, dtype=torch.uint8)
    scales = data_hp.new_empty(scale_shape, dtype=torch.float32)
    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


@convert_from_nvfp4.register_fake
def _fake_convert_from_nvfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    scale_format: str = "e4m3",
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    data_hp = data_lp.new_empty(data_lp.shape, dtype=output_dtype)
    return torch.cat((data_hp, data_hp), dim=axis)


def _qdq(
    tensor: torch.Tensor,
    *,
    axis: int,
    is_2d_block: bool,
    use_outer_scale: bool,
    use_sr: bool = False,
    block_size: int = 16,
    return_raw: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Quantize to NVFP4 then immediately dequantize back (QDQ round-trip).

    This helper is the core building block used to emulate the numerical effect
    of a future pure-NVFP4 GEMM path. Even though the current implementation
    executes the final matrix multiplications in BF16, every operand first goes
    through an explicit NVFP4 round-trip so the surrounding autograd code can
    model how a native FP4 kernel would see its inputs.

    When ``return_raw=True``, the packed FP4 data tensor (``uint8``) and the
    per-block scales (``float32`` held as E4M3-round-tripped values) are also
    returned alongside the dequantized BF16 view. This is used by the DGE
    backward path to recover the exact FP4 bin each weight element fell into
    without paying a second quantization pass.
    """
    outer_scale = (
        torch.empty(1, dtype=torch.float32, device=tensor.device)
        if use_outer_scale
        else None
    )
    data_lp, scales = convert_to_nvfp4(
        tensor,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        # With ``outer_scale`` provided, refresh it in-place from tensor.amax();
        # without it, skip outer-level scaling entirely.
        update_outer_scale=use_outer_scale,
        use_sr=use_sr,
    )
    dq = convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=tensor.dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
    )
    if return_raw:
        return dq, data_lp, scales
    return dq
