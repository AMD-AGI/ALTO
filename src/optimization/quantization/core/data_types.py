import torch
from typing import Optional


class DtypeArgs:
    exponent: int
    mantissa: int
    bits: int
    exponent_bias: int
    max: float
    min: float
    min_normal: float
    max_subnorm: float
    min_subnorm: float
    dtype: Optional[torch.dtype] = None


class FP8_E4M3(DtypeArgs):
    exponent = 4
    mantissa = 3
    bits = 8
    exponent_bias = 7
    max = torch.finfo(torch.float8_e4m3fn).max
    min = torch.finfo(torch.float8_e4m3fn).min
    min_normal = torch.finfo(torch.float8_e4m3fn).smallest_normal
    max_subnormal = 2 ** (-6) * 0.875
    min_subnormal = 2 ** (-9)
    dtype = torch.float8_e4m3fn

    @staticmethod
    def cast_data_to_format(x, *args, **kwargs):
        ori_dtype = x.dtype
        return x.to(FP8_E4M3.dtype).to(ori_dtype)
    
    @staticmethod
    def cast_scale_to_format(x, *args, **kwargs):
        ori_dtype = x.dtype
        return x.clamp(min=FP8_E4M3.min_normal, max=FP8_E4M3.max).to(FP8_E4M3.dtype).to(ori_dtype)


class FP8_E5M2(DtypeArgs):
    exponent = 5
    mantissa = 2
    bits = 8
    exponent_bias = 15
    max = torch.finfo(torch.float8_e5m2).max
    min = torch.finfo(torch.float8_e5m2).min
    min_normal = torch.finfo(torch.float8_e5m2).smallest_normal
    max_subnormal = 2 ** (-14) * 0.75
    min_subnormal = 2 ** (-16)
    dtype = torch.float8_e5m2

    @staticmethod
    def cast_to_format(x, *args, **kwargs):
        ori_dtype = x.dtype
        return x.to(FP8_E5M2.dtype).to(ori_dtype)

    @staticmethod
    def cast_scale_to_format(x, *args, **kwargs):
        ori_dtype = x.dtype
        return x.clamp(min=FP8_E5M2.min_normal, max=FP8_E5M2.max).to(FP8_E5M2.dtype).to(ori_dtype)


class FP6_E3M2(DtypeArgs):
    exponent = 3
    mantissa = 2
    bits = 6
    exponent_bias = 3
    max = 28.0
    min = -28.0
    min_normal = 0.25
    max_subnormal = 0.1875
    min_subnormal = 0.0625
    
    @staticmethod
    def cast_to_format(x, *args, **kwargs):
        return round_to_float(x, e_bits=FP6_E3M2.exponent, m_bits=FP6_E3M2.mantissa, *args, **kwargs)

    @staticmethod
    def cast_scale_to_format(x, *args, **kwargs):
        x = x.clamp(min=FP6_E3M2.min_normal, max=FP6_E3M2.max)
        return round_to_float(x, e_bits=FP6_E3M2.exponent, m_bits=FP6_E3M2.mantissa, *args, **kwargs)
    

class FP6_E2M3(DtypeArgs):
    exponent = 2
    mantissa = 3
    bits = 6
    exponent_bias = 1
    max = 7.5
    min = -7.5
    min_normal = 1.0
    max_subnormal = 0.875
    min_subnormal = 0.125
    
    @staticmethod
    def cast_to_format(x, *args, **kwargs):
        return round_to_float(x, e_bits=FP6_E2M3.exponent, m_bits=FP6_E2M3.mantissa, *args, **kwargs)

    @staticmethod
    def cast_scale_to_format(x, *args, **kwargs):
        x = x.clamp(min=FP6_E2M3.min_normal, max=FP6_E2M3.max)
        return round_to_float(x, e_bits=FP6_E2M3.exponent, m_bits=FP6_E2M3.mantissa, *args, **kwargs)
    

class FP4_E2M1(DtypeArgs):
    exponent = 2
    mantissa = 1
    bits = 4
    exponent_bias = 1
    max = 6.0
    min = -6.0
    min_normal = 1.0
    max_subnormal = 0.5
    min_subnormal = 0.5

    @staticmethod
    @torch.compile
    def cast_to_format_fast(x):
        sign = torch.sign(x)
        x = torch.abs(x)
        x[(x >= 0.0) & (x <= 0.25)] = 0.0
        x[(x > 0.25) & (x < 0.75)] = 0.5
        x[(x >= 0.75) & (x <= 1.25)] = 1.0
        x[(x > 1.25) & (x < 1.75)] = 1.5
        x[(x >= 1.75) & (x <= 2.5)] = 2.0
        x[(x > 2.5) & (x < 3.5)] = 3.0
        x[(x >= 3.5) & (x <= 5.0)] = 4.0
        x[x > 5.0] = 6.0
        return x * sign
    
    @staticmethod
    def cast_to_format(x, *args, **kwargs):
        return FP4_E2M1.cast_to_format_fast(x)

    @staticmethod
    def cast_scale_to_format(x, *args, **kwargs):
        x = x.clamp(min=FP4_E2M1.min_normal, max=FP4_E2M1.max)
        return FP4_E2M1.cast_to_format_fast(x)


class POT_E32M0(DtypeArgs):
    exponent = 32
    mantissa = 0
    bits = 32
    max = 2.0 ** (bits - 1)
    min = -max

    @staticmethod
    def cast_data_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E32M0.exponent, emax=emax, *args, **kwargs)

    @staticmethod
    def cast_scale_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E32M0.exponent, emax=emax, *args, **kwargs)
    

class POT_E16M0(DtypeArgs):
    exponent = 16
    mantissa = 0
    bits = 16
    max = 2.0 ** (bits - 1)
    min = -max

    @staticmethod
    def cast_data_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E16M0.exponent, emax=emax, *args, **kwargs)

    @staticmethod
    def cast_scale_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E16M0.exponent, emax=emax, *args, **kwargs)
    

class POT_E8M0(DtypeArgs):
    exponent = 8
    mantissa = 0
    bits = 8
    max = 2.0 ** (bits - 1)
    min = -max

    @staticmethod
    def cast_data_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E8M0.exponent, emax=emax, *args, **kwargs)

    @staticmethod
    def cast_scale_to_format(x, scaling_mode, emax, *args, **kwargs):
        return cast_to_eBm0(x, scaling_mode, ebits=POT_E8M0.exponent, emax=emax, *args, **kwargs)


DTYPE = {
    (5, 2):  FP8_E5M2,
    (4, 3):  FP8_E4M3,
    (3, 2):  FP6_E3M2,
    (2, 3):  FP6_E2M3,
    (2, 1):  FP4_E2M1,
    (8, 0):  POT_E8M0,
    (16, 0): POT_E16M0,
    (32, 0): POT_E32M0,
}


def generate_fp_table(device, e_bits, m_bits):
    exp_bias = (2 ** (e_bits - 1)) - 1
    max_exp = (2 ** e_bits) - 1
    min_exp = 1
    exponents = torch.arange(min_exp, max_exp + 1, device=device).float()
    mantissas = torch.arange(0, 2 ** m_bits, device=device).float()
    e, m = torch.meshgrid(exponents, mantissas, indexing='ij')
    normal_vals = (1.0 + m / (2 ** m_bits)) * (2.0 ** (e - exp_bias))
    normal_vals = normal_vals.flatten()
    sub_mantissas = torch.arange(1, 2 ** m_bits, device=device).float()
    subnormal_vals = (sub_mantissas / (2 ** m_bits)) * (2.0 ** (1 - exp_bias))
    pos_vals = torch.cat([subnormal_vals, normal_vals])
    fp_vals = torch.cat([-pos_vals, torch.tensor([0.0], device=device), pos_vals])
    fp_vals = torch.sort(fp_vals)[0]
    return fp_vals


def round_to_float(x: torch.Tensor, e_bits: int, m_bits: int, device=None, *args, **kwargs):
    if device is None:
        device = x.device
    fp_vals = generate_fp_table(device, e_bits, m_bits)
    sign = torch.sign(x)
    x_abs = torch.abs(x).flatten()
    indices = torch.bucketize(x_abs, fp_vals)
    indices = torch.clamp(indices, 0, len(fp_vals) - 1)
    left = fp_vals[torch.clamp(indices - 1, 0)]
    right = fp_vals[indices]
    choose_right = (x_abs - left) > (right - x_abs)
    nearest = torch.where(choose_right, right, left)
    quantized = nearest.view(x.shape)
    return quantized * sign


def cast_to_eBm0(
    scale: torch.Tensor,
    scaling_mode: str = 'floor',
    ebits: int = 8,
    emax: int = 1,
) -> torch.Tensor:
    scale_f32 = scale.to(torch.float32)
    scale_f32 = torch.clamp(scale_f32, min=2**(-126))
    qmin = -(2 ** (ebits - 1) - 1)
    qmax = +(2 ** (ebits - 1) - 1)
    if scaling_mode == 'floor':
        exp_unbiased = torch.floor(torch.log2(scale_f32))
    elif scaling_mode == 'ceil':
        exp_unbiased = torch.ceil(torch.log2(scale_f32))
    elif scaling_mode == 'even':
        exp_unbiased = torch.round(torch.log2(scale_f32))
    else:
        raise ValueError(f"Unsupported scaling mode: {scaling_mode}")
    exp_biased = torch.clamp(exp_unbiased, qmin, qmax) + emax
    return 2 ** (exp_biased)