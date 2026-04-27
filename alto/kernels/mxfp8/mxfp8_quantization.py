# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Tuple, Optional
import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl

BLOCK_SIZE_DEFAULT = 32


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


SUPPORTED_FORMATS = ["e4m3", "e5m2"]

# target_max_pow2: max representable power-of-2 exponent (unbiased)
# Reference: torchao/prototype/mx_formats/constants.py (F8E4M3_MAX_POW2, F8E5M2_MAX_POW2)
FORMAT_TO_TARGET_MAX = {
    "e4m3": 8,  # E4M3FN: max_exp=15, bias=7, unbiased=15-7=8
    "e5m2": 15,  # E5M2: max_exp=30, bias=15, unbiased=30-15=15
}

# Mantissa bits for each format (used in scale rounding)
FORMAT_TO_MBITS = {
    "e4m3": 3,  # E4M3: 3-bit mantissa
    "e5m2": 2,  # E5M2: 2-bit mantissa
}


@triton.jit
def _calculate_scales(
    x,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    target_max_pow2: tl.constexpr,
    mbits: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
):
    """Calculate power-of-2 block scales per OCP Microscaling spec."""
    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = tl.int16
        hp_mbits = 7
        hp_ebits = 8

    INPUT_SIGN_BIT = 1

    # Compute max absolute value per block
    NEW_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    if IS_2D_BLOCK:
        NEW_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        x = x.reshape(NEW_BLOCK_M, QUANT_BLOCK_SIZE, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x), axis=-1)
        max_abs = tl.max(max_abs, axis=-2)
    else:
        x = x.reshape(BLOCK_M, NEW_BLOCK_N, QUANT_BLOCK_SIZE)
        max_abs = tl.max(tl.abs(x), axis=-1)
    max_abs = max_abs.to(x.type.element_ty)

    # Extract exponent via bit manipulation (rounding based on target format's mantissa bits)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + INPUT_SIGN_BIT)) - 1) << hp_mbits
    max_abs = ((max_abs + val_to_add) & mask) >> hp_mbits

    # scale = max_abs_exponent - target_max_exponent
    scales = max_abs - target_max_pow2
    scales = tl.where(scales < 1, 1, scales)
    return scales.to(tl.uint8)


@triton.jit
def _generate_randval(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, _, _, _ = tl.randint4x(philox_seed, rng_offsets)
    return r1


@triton.jit
def _quantize_fp8(
    x,
    scales,
    philox_seed,
    philox_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
    USE_SR: tl.constexpr = False,
):
    """Quantize to FP8: qx = x / 2^scales. FP8_FORMAT: 0=e4m3, 1=e5m2."""
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    if USE_ASM:
        if USE_SR:
            # SR uses non-pack instructions: broadcast scales to full block shape
            if IS_2D_BLOCK:
                SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
                scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
                    SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)
            else:
                scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(BLOCK_M, SCALE_BLOCK_N,
                                                                           QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)

            randval = _generate_randval(BLOCK_M, BLOCK_N, philox_seed, philox_offset)

            if x.type.element_ty == tl.float32:
                if FP8_FORMAT == 0:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_sr_fp8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x, randval, scales_fp32],
                        dtype=tl.uint32,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_sr_bf8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x, randval, scales_fp32],
                        dtype=tl.uint32,
                        is_pure=True,
                        pack=1,
                    )
            else:
                if FP8_FORMAT == 0:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_sr_fp8_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x, randval, scales_fp32],
                        dtype=tl.uint32,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_sr_bf8_bf16 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x, randval, scales_fp32],
                        dtype=tl.uint32,
                        is_pure=True,
                        pack=1,
                    )

            # Hardware may produce NaN/Inf; saturate to max finite, preserving sign bit
            y = (y & 0xFF).to(tl.uint8)
            if FP8_FORMAT == 0:
                # e4m3fn: NaN is 0x7F/0xFF; clamp to 0x7E/0xFE
                y = tl.where((y & 0x7F) == 0x7F, ((y & 0x80) | 0x7E).to(tl.uint8), y)
            else:
                # e5m2: Inf/NaN are 0x7C-0x7F/0xFC-0xFF; clamp to 0x7B/0xFB
                y = tl.where((y & 0x7C) == 0x7C, ((y & 0x80) | 0x7B).to(tl.uint8), y)
            if FP8_FORMAT == 0:
                return y.to(tl.float8e4nv, bitcast=True)
            else:
                return y.to(tl.float8e5, bitcast=True)
        else:
            # RTNE uses pack instructions: broadcast scales to half block shape
            HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
            HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2

            if IS_2D_BLOCK:
                SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
                scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(SCALE_BLOCK_M, QUANT_BLOCK_SIZE,
                                                                                SCALE_BLOCK_N,
                                                                                HALF_QUANT_BLOCK_SIZE).reshape(
                                                                                    BLOCK_M, HALF_BLOCK_N)
            else:
                scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(BLOCK_M, SCALE_BLOCK_N,
                                                                           HALF_QUANT_BLOCK_SIZE).reshape(
                                                                               BLOCK_M, HALF_BLOCK_N)

            x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))

            if x0.type.element_ty == tl.float32:
                if FP8_FORMAT == 0:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_pk_fp8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x0, x1, scales_fp32],
                        dtype=tl.uint16,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_pk_bf8_f32 $0, $1, $2, $3 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v,v",
                        args=[x0, x1, scales_fp32],
                        dtype=tl.uint16,
                        is_pure=True,
                        pack=1,
                    )
            else:
                x_packed = (x1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | x0.to(tl.uint16, bitcast=True)
                if FP8_FORMAT == 0:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_pk_fp8_bf16 $0, $1, $2 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v",
                        args=[x_packed, scales_fp32],
                        dtype=tl.uint16,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    y = tl.inline_asm_elementwise(
                        asm="""
                        v_cvt_scalef32_pk_bf8_bf16 $0, $1, $2 op_sel:[0,0,0,0];
                        """,
                        constraints="=&v,v,v",
                        args=[x_packed, scales_fp32],
                        dtype=tl.uint16,
                        is_pure=True,
                        pack=1,
                    )

            # Hardware may produce NaN/Inf; saturate to max finite, preserving sign bit
            y0 = (y & 0xFF).to(tl.uint8)
            y1 = ((y >> 8) & 0xFF).to(tl.uint8)
            if FP8_FORMAT == 0:
                # e4m3fn: NaN is 0x7F/0xFF; clamp to 0x7E/0xFE
                y0 = tl.where((y0 & 0x7F) == 0x7F, ((y0 & 0x80) | 0x7E).to(tl.uint8), y0)
                y1 = tl.where((y1 & 0x7F) == 0x7F, ((y1 & 0x80) | 0x7E).to(tl.uint8), y1)
            else:
                # e5m2: Inf/NaN are 0x7C-0x7F/0xFC-0xFF; clamp to 0x7B/0xFB
                y0 = tl.where((y0 & 0x7C) == 0x7C, ((y0 & 0x80) | 0x7B).to(tl.uint8), y0)
                y1 = tl.where((y1 & 0x7C) == 0x7C, ((y1 & 0x80) | 0x7B).to(tl.uint8), y1)
            qx = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
            if FP8_FORMAT == 0:
                return qx.to(tl.float8e4nv, bitcast=True)
            else:
                return qx.to(tl.float8e5, bitcast=True)
    else:
        tl.static_assert(not USE_SR, "USE_SR requires USE_ASM=True")
        if IS_2D_BLOCK:
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
            scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(SCALE_BLOCK_M, QUANT_BLOCK_SIZE,
                                                                            SCALE_BLOCK_N,
                                                                            QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)
        else:
            scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(BLOCK_M, SCALE_BLOCK_N,
                                                                       QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)

        qx = x.to(tl.float32) / scales_fp32

        if FP8_FORMAT == 0:
            qx = qx.to(tl.float8e4nv)
        else:
            qx = qx.to(tl.float8e5)

        return qx


@triton.jit
def _dequantize_fp8(
    x,
    scales,
    output_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr = False,
    USE_ASM: tl.constexpr = False,
):
    """Dequantize FP8: y = x * 2^scales. FP8_FORMAT: 0=e4m3, 1=e5m2."""
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE

    scales_fp32 = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)

    if USE_ASM:
        HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
        HALF_QUANT_BLOCK_SIZE: tl.constexpr = QUANT_BLOCK_SIZE // 2

        if IS_2D_BLOCK:
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
            scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(
                SCALE_BLOCK_M, QUANT_BLOCK_SIZE, SCALE_BLOCK_N, HALF_QUANT_BLOCK_SIZE).reshape(BLOCK_M, HALF_BLOCK_N)
        else:
            scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(BLOCK_M, SCALE_BLOCK_N,
                                                                       HALF_QUANT_BLOCK_SIZE).reshape(
                                                                           BLOCK_M, HALF_BLOCK_N)

        x0, x1 = tl.split(x.reshape(BLOCK_M, HALF_BLOCK_N, 2))
        x_packed = x0.to(tl.uint8, bitcast=True).to(tl.uint16) | (x1.to(tl.uint8, bitcast=True).to(tl.uint16) << 8)

        if output_dtype == tl.float32:
            if FP8_FORMAT == 0:
                y_packed = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_f32_fp8 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_packed, scales_fp32],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
            else:
                y_packed = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_f32_bf8 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_packed, scales_fp32],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
            y1 = (y_packed >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
            y0 = (y_packed & 0x00000000FFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        else:
            if FP8_FORMAT == 0:
                y_packed = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_packed, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            else:
                y_packed = tl.inline_asm_elementwise(
                    asm="""
                    v_cvt_scalef32_pk_bf16_bf8 $0, $1, $2 op_sel:[0,0];
                    """,
                    constraints="=&v,v,v",
                    args=[x_packed, scales_fp32],
                    dtype=tl.uint32,
                    is_pure=True,
                    pack=1,
                )
            y1 = (y_packed >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            y0 = (y_packed & 0x0000FFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)

        y = tl.join(y0, y1).reshape(BLOCK_M, BLOCK_N)
    else:
        if IS_2D_BLOCK:
            SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
            scales_fp32 = scales_fp32.expand_dims(axis=(1, 3)).broadcast_to(SCALE_BLOCK_M, QUANT_BLOCK_SIZE,
                                                                            SCALE_BLOCK_N,
                                                                            QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)
        else:
            scales_fp32 = scales_fp32.expand_dims(axis=2).broadcast_to(BLOCK_M, SCALE_BLOCK_N,
                                                                       QUANT_BLOCK_SIZE).reshape(BLOCK_M, BLOCK_N)

        y = x.to(tl.float32) * scales_fp32

        if output_dtype != tl.float32:
            y = y.to(output_dtype)

    return y


@triton.jit
def _convert_to_mxfp8_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    target_max_pow2: tl.constexpr,
    mbits: tl.constexpr,
    FP8_FORMAT: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr,
    USE_SR: tl.constexpr,
):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr: Pointer to the input tensor.
        y_ptr: Pointer to the output tensor where quantized values will be stored.
        s_ptr: Pointer to the output tensor where scaling factors will be stored.
        stride_xm, stride_xn: Strides for the input tensor.
        stride_ym, stride_yn: Strides for the output tensor.
        stride_sm, stride_sn: Strides for the scale tensor.
        BLOCK_M (tl.constexpr): Block size along the M dimension.
        BLOCK_N (tl.constexpr): Block size along the N dimension.
        QUANT_BLOCK_SIZE (tl.constexpr): The quantization block size (e.g., 32).
        target_max_pow2 (tl.constexpr): Maximum power-of-2 exponent for the target format.
        mbits (tl.constexpr): Number of mantissa bits for the target format.
        FP8_FORMAT (tl.constexpr): Target FP8 format (0=e4m3, 1=e5m2).
        IS_2D_BLOCK (tl.constexpr): Whether to use 2D block quantization.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_xn < N
    mask_sn = offs_sn < (N // QUANT_BLOCK_SIZE)

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
        mask_sm = offs_sm < (M // QUANT_BLOCK_SIZE)
    else:
        offs_sm = offs_m
        mask_sm = mask_m

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_xn[None, :] * stride_yn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    scales = _calculate_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        target_max_pow2=target_max_pow2,
        mbits=mbits,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    y = _quantize_fp8(
        x,
        scales,
        philox_seed,
        philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        FP8_FORMAT=FP8_FORMAT,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_ASM=USE_ASM,
        USE_SR=USE_SR,
    )

    tl.store(y_ptr + offs_y, y, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(s_ptr + offs_s, scales, mask=mask_sm[:, None] & mask_sn[None, :])


@triton_op("alto::convert_to_mxfp8", mutates_args={})
def convert_to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    mxfp_format: str = "e4m3",
    axis: int = -1,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: Optional[bool] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert high-precision tensor (FP32/BF16) to MXFP8 with block scales.

    Args:
        data_hp: Input tensor (FP32 or BF16).
        block_size: Quantization block size, default 32.
        mxfp_format: Target FP8 format: "e4m3" or "e5m2".
        axis: Axis to quantize along.
        is_2d_block: If True, use 2D block quantization.
        use_sr: If True, use stochastic rounding (requires use_asm=True).
        use_asm: If True, use CDNA4 ASM instructions. None=auto-detect.
        philox_seed: Random seed for stochastic rounding. None=auto-generate.
        philox_offset: Random offset for stochastic rounding. None=auto-generate.

    Returns:
        Tuple[data_lp, scales]: Quantized FP8 tensor and uint8 scale factors.
    """
    torch._check(data_hp.shape[axis] % block_size == 0,
                 f"tensor shape ({data_hp.shape}) at axis={axis} is not divisible by {block_size}")

    assert mxfp_format in SUPPORTED_FORMATS, \
        f"Unsupported format: {mxfp_format}. Supported: {SUPPORTED_FORMATS}"

    if mxfp_format == "e4m3":
        fp8_dtype = torch.float8_e4m3fn
        fp8_format_id = 0
    else:
        fp8_dtype = torch.float8_e5m2
        fp8_format_id = 1
    target_max_pow2 = FORMAT_TO_TARGET_MAX[mxfp_format]
    mbits = FORMAT_TO_MBITS[mxfp_format]

    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    if use_asm is None:
        use_asm = is_cdna4()
    assert not use_sr or use_asm, "use_sr=True requires use_asm=True"

    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape

    if is_2d_block:
        torch._check(ori_shape[-2] % block_size == 0,
                     f"tensor shape at axis={axis}-1 ({ori_shape[-2]}) is not divisible by {block_size} for 2D block")

    data_hp = data_hp.reshape(-1, ori_shape[-1])

    data_lp = torch.empty(ori_shape, dtype=fp8_dtype, device=data_hp.device).reshape(-1, ori_shape[-1])

    if is_2d_block:
        scales_shape = (*ori_shape[:-2], ori_shape[-2] // block_size, ori_shape[-1] // block_size)
    else:
        scales_shape = (*ori_shape[:-1], ori_shape[-1] // block_size)
    scales = torch.empty(scales_shape, dtype=torch.uint8, device=data_hp.device).reshape(-1, scales_shape[-1])

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

    wrap_triton(_convert_to_mxfp8_kernel)[grid](
        data_hp,
        data_lp,
        scales,
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        target_max_pow2=target_max_pow2,
        mbits=mbits,
        FP8_FORMAT=fp8_format_id,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
        USE_SR=use_sr,
    )

    return data_lp.reshape(ori_shape).transpose(axis, -1), scales.reshape(scales_shape).transpose(axis, -1)


@triton_op("alto::convert_from_mxfp8", mutates_args={})
def convert_from_mxfp8(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    """
    Convert MXFP8 tensor with block scales back to high-precision (FP32/BF16).

    Args:
        data_lp: FP8 quantized tensor.
        scales: uint8 scale factors from quantization.
        output_dtype: Output dtype (FP32 or BF16).
        block_size: Quantization block size, must match what was used during quantization.
        axis: Axis that was quantized along.
        is_2d_block: If True, use 2D block quantization.
        use_asm: If True, use CDNA4 ASM instructions. None=auto-detect.

    Returns:
        Dequantized high-precision tensor.
    """
    assert output_dtype in [torch.float32, torch.bfloat16], \
        f"output_dtype must be float32 or bfloat16, got {output_dtype}"

    if use_asm is None:
        use_asm = is_cdna4()

    if data_lp.dtype == torch.float8_e4m3fn:
        fp8_format_id = 0
    elif data_lp.dtype == torch.float8_e5m2:
        fp8_format_id = 1
    else:
        raise ValueError(f"Unsupported FP8 dtype: {data_lp.dtype}")

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    ori_shape = data_lp.shape
    data_lp = data_lp.reshape(-1, ori_shape[-1])
    scales = scales.reshape(-1, ori_shape[-1] // block_size)

    data_hp = torch.empty(ori_shape, dtype=output_dtype, device=data_lp.device).reshape(-1, ori_shape[-1])

    stride_xm, stride_xn = data_lp.stride()
    stride_ym, stride_yn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()
    M, N = data_lp.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N

    wrap_triton(_convert_from_mxfp8_kernel)[grid](
        data_lp,
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        FP8_FORMAT=fp8_format_id,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )

    return data_hp.reshape(ori_shape).transpose(axis, -1)


@triton.jit
def _convert_from_mxfp8_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
    USE_ASM: tl.constexpr,
):
    """Dequantize MXFP8 tensor back to FP32/BF16."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_sn = offs_sn < (N // QUANT_BLOCK_SIZE)

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
        mask_sm = offs_sm < (M // QUANT_BLOCK_SIZE)
    else:
        offs_sm = offs_m
        mask_sm = mask_m

    offs_x = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    offs_y = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn

    x = tl.load(x_ptr + offs_x, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    scales = tl.load(s_ptr + offs_s, mask=mask_sm[:, None] & mask_sn[None, :], other=1)

    y = _dequantize_fp8(
        x,
        scales,
        output_dtype=y_ptr.type.element_ty,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        FP8_FORMAT=FP8_FORMAT,
        IS_2D_BLOCK=IS_2D_BLOCK,
        USE_ASM=USE_ASM,
    )

    tl.store(y_ptr + offs_y, y, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _calculate_scales_kernel(
    x_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    stride_sm,
    stride_sn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    target_max_pow2: tl.constexpr,
    mbits: tl.constexpr,
    IS_2D_BLOCK: tl.constexpr,
):
    """Kernel for standalone scale calculation."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_xn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_xn < N

    offs_x = offs_m[:, None] * stride_xm + offs_xn[None, :] * stride_xn
    x = tl.load(x_ptr + offs_x, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    scales = _calculate_scales(
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
        target_max_pow2=target_max_pow2,
        mbits=mbits,
        IS_2D_BLOCK=IS_2D_BLOCK,
    )

    if IS_2D_BLOCK:
        SCALE_BLOCK_M: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        offs_sm = pid_m * SCALE_BLOCK_M + tl.arange(0, SCALE_BLOCK_M)
        mask_sm = offs_sm < (M // QUANT_BLOCK_SIZE)
    else:
        offs_sm = offs_m
        mask_sm = mask_m
    offs_sn = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)
    mask_sn = offs_sn < (N // QUANT_BLOCK_SIZE)
    offs_s = offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn
    tl.store(s_ptr + offs_s, scales, mask=mask_sm[:, None] & mask_sn[None, :])


@triton_op("alto::calculate_mxfp8_scales", mutates_args={})
def calculate_mxfp8_scales(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    mxfp_format: str = "e4m3",
    axis: int = -1,
    is_2d_block: bool = False,
) -> torch.Tensor:
    """
    Calculate block scales for MXFP8 quantization.

    Args:
        data_hp: Input tensor (FP32 or BF16).
        block_size: Quantization block size, default 32.
        mxfp_format: Target FP8 format: "e4m3" or "e5m2".
        axis: Axis to quantize along.
        is_2d_block: If True, use 2D block quantization.

    Returns:
        scales: uint8 scale factors.
    """
    assert mxfp_format in SUPPORTED_FORMATS, \
        f"Unsupported format: {mxfp_format}. Supported: {SUPPORTED_FORMATS}"
    assert data_hp.dtype in [torch.float32, torch.bfloat16]

    target_max_pow2 = FORMAT_TO_TARGET_MAX[mxfp_format]
    mbits = FORMAT_TO_MBITS[mxfp_format]

    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape

    torch._check(ori_shape[-1] % block_size == 0,
                 f"tensor shape at axis={axis} ({ori_shape[-1]}) is not divisible by {block_size}")
    if is_2d_block:
        torch._check(ori_shape[-2] % block_size == 0,
                     f"tensor shape at axis={axis}-1 ({ori_shape[-2]}) is not divisible by {block_size} for 2D block")

    data_hp = data_hp.reshape(-1, ori_shape[-1])

    M, N = data_hp.shape
    if is_2d_block:
        scales_shape = (M // block_size, N // block_size)
    else:
        scales_shape = (M, N // block_size)
    scales = torch.empty(scales_shape, dtype=torch.uint8, device=data_hp.device)

    stride_xm, stride_xn = data_hp.stride()
    stride_sm, stride_sn = scales.stride()

    BLOCK_M = 64 if M >= 64 else M
    BLOCK_N = 64 if N >= 64 else N

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

    wrap_triton(_calculate_scales_kernel)[grid](
        data_hp,
        scales,
        stride_xm,
        stride_xn,
        stride_sm,
        stride_sn,
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        QUANT_BLOCK_SIZE=block_size,
        target_max_pow2=target_max_pow2,
        mbits=mbits,
        IS_2D_BLOCK=is_2d_block,
    )

    # Restore original shape
    if is_2d_block:
        out_shape = (*ori_shape[:-2], ori_shape[-2] // block_size, ori_shape[-1] // block_size)
    else:
        out_shape = (*ori_shape[:-1], ori_shape[-1] // block_size)
    return scales.reshape(out_shape).transpose(axis, -1)
