# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import triton
import triton.language as tl


def make_generate_philox_randval_2x():
    @triton.jit
    def _generate_philox_randval_2x(m, n, philox_seed, philox_offset):
        """Generate two Philox random streams for stochastic rounding."""
        ms = tl.arange(0, m)
        ns = tl.arange(0, n)
        rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
        r1, r2, _, _ = tl.randint4x(philox_seed, rng_offsets)
        return r1, r2

    return _generate_philox_randval_2x


def make_quantize_e2m1():
    @triton.jit
    def _quantize_e2m1(x, scales_fp32, randval, USE_SR: tl.constexpr):
        """Quantize values into E2M1 FP4 using float32 scales."""
        EXP_BIAS_FP32: tl.constexpr = 127
        EXP_BIAS_FP4: tl.constexpr = 1
        EBITS_F32: tl.constexpr = 8
        EBITS_FP4: tl.constexpr = 2
        MBITS_F32: tl.constexpr = 23
        MBITS_FP4: tl.constexpr = 1

        max_normal: tl.constexpr = 6
        min_normal: tl.constexpr = 1

        qx = x.to(tl.float32) / scales_fp32
        qx = qx.to(tl.uint32, bitcast=True)

        s = qx & 0x80000000
        qx = qx ^ s

        qx_fp32 = qx.to(tl.float32, bitcast=True)
        saturate_mask = qx_fp32 >= max_normal
        denormal_mask = (~saturate_mask) & (qx_fp32 < min_normal)
        normal_mask = ~(saturate_mask | denormal_mask)

        if USE_SR:
            denorm_mask_low = denormal_mask & (qx_fp32 < 0.5)
            denorm_mask_high = denormal_mask & (~denorm_mask_low)
            randval_uint = randval.to(tl.uint32, bitcast=True)
            denormal_x = (qx * 0).to(tl.uint8)

            threshold_low = (qx_fp32 * (2**33 - 2)).to(tl.uint32)
            denormal_x = tl.where(randval_uint <= threshold_low, 1, denormal_x)

            threshold_high = ((qx_fp32 * 2 - 1) * (2**32 - 1)).to(tl.uint32)
            mask_high = randval_uint <= threshold_high
            denormal_x = tl.where(denorm_mask_high & mask_high, 2, denormal_x)
            denormal_x = tl.where(denorm_mask_high & (~mask_high), 1, denormal_x)
        else:
            denorm_exp: tl.constexpr = (
                (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
            )
            denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
            denorm_mask_float: tl.constexpr = tl.cast(
                denorm_mask_int, tl.float32, bitcast=True
            )

            denormal_x = qx_fp32 + denorm_mask_float
            denormal_x = denormal_x.to(tl.uint32, bitcast=True)
            denormal_x -= denorm_mask_int
            denormal_x = denormal_x.to(tl.uint8)

        normal_x = qx
        mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
        val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32)
        if USE_SR:
            val_to_add += randval & ((1 << (MBITS_F32 - MBITS_FP4)) - 1)
        else:
            val_to_add += (1 << (MBITS_F32 - MBITS_FP4 - 1)) - 1 + mant_odd
        normal_x += val_to_add
        normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
        normal_x = normal_x.to(tl.uint8)

        e2m1_value = ((qx * 0) + 0x7).to(tl.uint8)
        e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
        e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
        sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
        sign_lp = sign_lp.to(tl.uint8)
        e2m1_value = e2m1_value | sign_lp
        return e2m1_value

    return _quantize_e2m1


def make_dequantize_e2m1():
    @triton.jit
    def _dequantize_e2m1(qx, scales_fp32):
        """Dequantize E2M1 FP4 values with float32 scales."""
        EXP_BIAS_FP32: tl.constexpr = 127
        EXP_BIAS_FP4: tl.constexpr = 1
        EBITS_F32: tl.constexpr = 8
        EBITS_FP4: tl.constexpr = 2
        MBITS_F32: tl.constexpr = 23
        MBITS_FP4: tl.constexpr = 1

        tl.static_assert(qx.dtype == tl.uint8)

        s = qx & 0x8
        qx = qx ^ s

        zero_mask = qx == 0x0
        denormal_mask = qx == 0x1

        exp_biased_lp = qx >> MBITS_FP4
        exp_biased_f32 = exp_biased_lp - EXP_BIAS_FP4 + EXP_BIAS_FP32
        exp_biased_f32 = exp_biased_f32.to(tl.uint32) << MBITS_F32

        mantissa_lp_int32 = (qx & 0x1).to(tl.int32)
        mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - MBITS_FP4)
        x_int = exp_biased_f32 | mantissa_f32

        x_int = tl.where(zero_mask, 0, x_int)
        x_int = tl.where(denormal_mask, 0x3F000000, x_int)

        sign_lp = s.to(tl.uint32) << (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
        x_int = x_int | sign_lp

        x_fp = x_int.to(tl.float32, bitcast=True)
        return x_fp * scales_fp32

    return _dequantize_e2m1


generate_philox_randval_2x = make_generate_philox_randval_2x()
quantize_e2m1 = make_quantize_e2m1()
dequantize_e2m1 = make_dequantize_e2m1()
