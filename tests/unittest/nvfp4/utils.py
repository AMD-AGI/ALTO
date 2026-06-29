# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_SCALE_FORMATS,
    _SCALE_FORMAT_TABLE,
    OUTER_BLOCK_SIZE_DEFAULT,
    OUTER_SCALE_EPS,
)
from alto.kernels.fp4.fp4_common import quantize_to_ue5m3

# Re-exported so existing ``from .utils import calc_snr, calc_cossim``
# call-sites keep working; the single source of truth lives in
# ``alto.kernels.fp4.testing_utils``.
from alto.kernels.fp4.testing_utils import calc_snr, calc_cossim  # noqa: F401


F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


def _quantize_inner_scale(inner_scale_raw: Tensor, scale_format: str) -> Tensor:
    """Snap per-block inner scales to the selected inner grid (FP32 out).

    NaN / +-Inf are sanitised to ``[fmt_eps, fmt_max]`` before the cast so a
    NaN never propagates a 0xFF (UE5M3 NaN / E4M3 NaN) code into downstream
    GEMM scales -- mirroring the TransformerEngine / vLLM / TRT-LLM pattern.
    """
    if scale_format not in _SCALE_FORMAT_TABLE:
        raise ValueError(
            f"scale_format={scale_format!r} not supported; "
            f"expected one of {SUPPORTED_SCALE_FORMATS}"
        )
    fmt_eps, fmt_max = _SCALE_FORMAT_TABLE[scale_format]
    inner_scale_raw = torch.nan_to_num(
        inner_scale_raw, nan=fmt_eps, posinf=fmt_max, neginf=fmt_eps,
    )
    clamped = inner_scale_raw.clamp(min=fmt_eps, max=fmt_max)
    if scale_format == "ue5m3":
        return quantize_to_ue5m3(clamped.float())
    return clamped.float().to(torch.float8_e4m3fn).to(torch.float32)


def prepare_data(tensor_shape, data_type, pattern="random", seed: int = 1234,
                 outlier_scale: float = 1000.0):
    """Prepare test data with specified pattern.

    Args:
        tensor_shape: Shape of the output tensor.
        data_type: Data type (torch.float32 or torch.bfloat16).
        pattern: Data pattern -
            "random"   : Gaussian with sparse outliers (default).
            "zeros"    : All zeros — exercises the E4M3 lower clamp on the
                         stored block scale.
            "large"    : All 5000.0 — exceeds F8E4M3_MAX * F4_E2M1_MAX (2688),
                         tests FP4 saturation and scale clamp to F8E4M3_MAX.
            "lognormal_channel_outlier" : Log-normal background with a small
                         fraction of channels (along the last axis) scaled by
                         ``outlier_scale``.  Approximates LLM activation
                         distributions (Sun et al. 2024 "massive activations"),
                         which are the worst case for tensor-wise scaling and
                         the best case for outer-block scaling.  See
                         NVFP4_Outer_Block_Review.md §3.3 T1.
        seed: Per-call seed override.  Defaults to 1234 for backward
              compatibility; tests that mix multiple patterns / shapes in
              a single file should pass distinct seeds to avoid sharing the
              same RNG state across parametrised cases.
        outlier_scale: Multiplier applied to the promoted "massive" channels of
              the ``lognormal_channel_outlier`` pattern (default 1e3).  Larger
              values (e.g. 1e5) drive per-tensor scaling into E4M3 inner-scale
              underflow, the regime where outer-block scaling is decisively
              better.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")

    if pattern == "random":
        x = torch.randn(tensor_shape, dtype=data_type, device=device)
        p_mask = torch.bernoulli(torch.ones_like(x) * 0.005)
        x += 100 * torch.randn_like(x) * p_mask
    elif pattern == "zeros":
        x = torch.zeros(tensor_shape, dtype=data_type, device=device)
    elif pattern == "large":
        x = torch.ones(tensor_shape, dtype=data_type, device=device) * 5000.0
    elif pattern == "hot_channel":
        x = torch.randn(tensor_shape, dtype=data_type, device=device)
        col = tensor_shape[-1] // 2
        x[..., col] = 300.0
    elif pattern == "lognormal":
        x = torch.exp(torch.randn(tensor_shape, dtype=data_type, device=device) * 1.5)
    elif pattern == "near_overflow":
        x = torch.ones(tensor_shape, dtype=data_type, device=device) * 1.0e4
    elif pattern == "near_underflow":
        x = torch.ones(tensor_shape, dtype=data_type, device=device) * 1.0e-4
    elif pattern == "single_spike":
        x = torch.zeros(tensor_shape, dtype=data_type, device=device)
        flat = x.reshape(-1)
        flat[flat.numel() // 2] = 5000.0
    elif pattern == "lognormal_channel_outlier":
        # Log-normal-style background (heavy-tailed, mean ≈ 0).  We take
        # ``randn ** 2 * 0.1 * sign(randn)`` so the marginal is approximately
        # ``±X^2 / 10`` with X ~ N(0, 1), a heavy-tailed proxy.
        base = torch.randn(tensor_shape, dtype=torch.float32, device=device)
        x = (base.abs() ** 2) * 0.1 * base.sign()
        if len(tensor_shape) >= 1 and tensor_shape[-1] >= 8:
            # Promote ~1% of the last-axis channels to massive activations.
            num_channels = tensor_shape[-1]
            num_outlier = max(1, num_channels // 100)
            generator = torch.Generator(device="cpu").manual_seed(seed + 1)
            outlier_idx = torch.randperm(num_channels, generator=generator)[:num_outlier]
            x[..., outlier_idx] = x[..., outlier_idx] * outlier_scale
        x = x.to(data_type)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return x


# ---------------------------------------------------------------------------
# Low-level FP4 <-> FP32 bit-manipulation helpers (shared with mxfp4)
# ---------------------------------------------------------------------------

def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1


def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/questions/8981913/how-to-perform-round-to-even-with-floating-point-numbers  # noqa: E501
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    max_normal = 2**(_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) /
                                                   (2**mbits))
    min_normal = 2**(1 - exp_bias)

    denorm_exp = (
        (F32_EXP_BIAS - exp_bias)
        + (MBITS_F32 - mbits)
        + 1)
    denorm_mask_int = denorm_exp << MBITS_F32

    denorm_mask_float = torch.tensor(denorm_mask_int,
                                     dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000

    x = x ^ sign

    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x
                                      < min_normal)
    normal_mask = torch.logical_not(
        torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _floatx_unpacked_to_f32(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    sign_lp = x & sign_mask

    x_pos = x ^ sign_lp

    zero_mask = x_pos == 0

    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32
    else:
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp -
                                (1 << i)) << (left_shift + MBITS_F32 - mbits)
                exp_biased_f32 = (denormal_exp_biased -
                                  left_shift) << MBITS_F32
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32)
        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    sign_f32 = sign_lp.to(
        torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def f32_to_f4_unpacked(x):
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def f4_unpacked_to_f32(x: torch.Tensor):
    return _floatx_unpacked_to_f32(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data) -> torch.Tensor:
    assert uint8_data.is_contiguous()
    shape = uint8_data.shape
    second_elements = (uint8_data >> 4).to(torch.uint8)
    first_elements = (uint8_data & 0b1111).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements],
                           dim=-1).view(up_size(shape))
    return unpacked


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# ---------------------------------------------------------------------------
# NVFP4 PyTorch reference implementations
# ---------------------------------------------------------------------------

def convert_to_nvfp4_pytorch(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: torch.Tensor = None,
    scale_format: str = "e4m3",
):
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    fmt_eps, fmt_max = _SCALE_FORMAT_TABLE[scale_format]

    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape

    if is_2d_block:
        M_blocks = ori_shape[-2] // block_size
        N_blocks = ori_shape[-1] // block_size
        grouped = data_hp.reshape(
            *ori_shape[:-2], M_blocks, block_size,
            N_blocks, block_size).float()
        max_abs = grouped.abs().amax(dim=-1).amax(dim=-2)
    else:
        data_hp_2d = data_hp.reshape(-1, ori_shape[-1])
        M, N = data_hp_2d.shape
        num_blocks = N // block_size
        grouped = data_hp_2d.reshape(M, num_blocks, block_size).float()
        max_abs = grouped.abs().amax(dim=-1)

    # NVFP4 spec order: outer_scale-normalise first, then derive the inner
    # block scale, with clamp + E4M3 round applied exactly once on the
    # final stored value.
    if outer_scale is not None:
        outer_scale = outer_scale.float().to(data_hp.device)
        inner_scale_raw = (max_abs / outer_scale / F4_E2M1_MAX).clamp(min=fmt_eps, max=fmt_max)
        inner_scale = _quantize_inner_scale(inner_scale_raw, scale_format)
        quant_scale = inner_scale * outer_scale
    else:
        inner_scale_raw = (max_abs / F4_E2M1_MAX).clamp(min=fmt_eps, max=fmt_max)
        inner_scale = _quantize_inner_scale(inner_scale_raw, scale_format)
        quant_scale = inner_scale

    if is_2d_block:
        quant_scale_expanded = quant_scale.unsqueeze(-2).unsqueeze(-1).expand_as(grouped)
        scaled = (grouped / quant_scale_expanded).reshape(ori_shape)
    else:
        quant_scale_expanded = quant_scale.unsqueeze(-1).expand_as(grouped)
        scaled = (grouped / quant_scale_expanded).reshape(M, N).reshape(ori_shape)
        inner_scale = inner_scale.reshape(*ori_shape[:-1], num_blocks)

    data_lp_unpacked = f32_to_f4_unpacked(scaled)
    data_lp = pack_uint4(data_lp_unpacked)

    return data_lp.transpose(axis, -1), inner_scale.transpose(axis, -1)


def convert_from_nvfp4_pytorch(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: torch.Tensor = None,
    scale_format: str = "e4m3",
):
    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1).float()

    orig_lp_shape = data_lp.shape
    f4_unpacked = unpack_uint4(data_lp)
    f32 = f4_unpacked_to_f32(f4_unpacked)
    orig_hp_shape = (*orig_lp_shape[:-1], orig_lp_shape[-1] * 2)

    if outer_scale is not None:
        scales = scales * outer_scale.float().to(scales.device)

    if is_2d_block:
        new_shape = (*orig_hp_shape[:-2],
                     orig_hp_shape[-2] // block_size, block_size,
                     orig_hp_shape[-1] // block_size, block_size)
        scale_shape = (*orig_hp_shape[:-2],
                       orig_hp_shape[-2] // block_size, 1,
                       orig_hp_shape[-1] // block_size, 1)
    else:
        new_shape = (*orig_hp_shape[:-1], orig_hp_shape[-1] // block_size, block_size)
        scale_shape = (*orig_hp_shape[:-1], orig_hp_shape[-1] // block_size, 1)

    f32 = f32.reshape(new_shape)
    s_expanded = scales.reshape(scale_shape)
    result = (f32 * s_expanded).to(output_dtype).reshape(orig_hp_shape)
    return result.transpose(axis, -1)


# ---------------------------------------------------------------------------
# NVFP4 outer-block PyTorch reference
#
# These mirror the structure of ``convert_to_nvfp4_outer_block`` /
# ``convert_from_nvfp4_outer_block`` in the production kernel, but go through
# pure-tensor amax + ``convert_to_nvfp4_pytorch`` so they can be used as a
# trustworthy bit-level reference for tests.
# ---------------------------------------------------------------------------


def _outer_scale_axis_last_pytorch(
    data_axis_last: torch.Tensor,
    *,
    is_2d_outer: bool,
    inner_block_size: int,
    outer_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference outer scale + expanded scale on an axis-last tensor.

    Mirrors ``_outer_scale_and_expanded_axis_last`` in the production kernel,
    including the FP32-floor clamp.  Returns ``(outer_scale, outer_expanded)``
    both in axis-last layout.
    """
    del inner_block_size  # outer scale doesn't depend on inner block size
    shape = tuple(data_axis_last.shape)
    m, n = shape[-2], shape[-1]
    prefix = shape[:-2]
    x = data_axis_last.float()
    if is_2d_outer:
        grouped = x.reshape(
            *prefix,
            m // outer_block_size, outer_block_size,
            n // outer_block_size, outer_block_size,
        )
        outer_amax = grouped.abs().amax(dim=-1).amax(dim=-2)
        outer_scale = (outer_amax / (F8E4M3_MAX * F4_E2M1_MAX)).clamp(min=OUTER_SCALE_EPS)
        expanded = outer_scale.unsqueeze(-2).unsqueeze(-1).expand_as(grouped).reshape(shape)
    else:
        grouped = x.reshape(*prefix, m, n // outer_block_size, outer_block_size)
        outer_amax = grouped.abs().amax(dim=-1)
        outer_scale = (outer_amax / (F8E4M3_MAX * F4_E2M1_MAX)).clamp(min=OUTER_SCALE_EPS)
        expanded = outer_scale.unsqueeze(-1).expand_as(grouped).reshape(shape)
    return outer_scale.to(torch.float32), expanded.to(torch.float32)


def convert_to_nvfp4_outer_block_pytorch(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    outer_block_size: int = OUTER_BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_inner: bool = False,
    is_2d_outer: bool = False,
    scale_format: str = "e4m3",
):
    """Pure-PyTorch reference for ``convert_to_nvfp4_outer_block``.

    Forward path: outer amax → FP32 outer scale → normalize → inner NVFP4 QDQ
    via :func:`convert_to_nvfp4_pytorch` with ``outer_scale=None``.
    Returns ``(data_lp, inner_scales, outer_scale)`` all in **original axis
    layout** (matching the production kernel's return convention).

    Note: this reference does **not** support stochastic rounding (the
    production kernel exposes ``use_sr`` only when called from
    ``convert_to_nvfp4``).  Tests that need bit-equality must therefore run
    with deterministic rounding (``use_sr=False``).
    """
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    data_axis_last = data_hp.transpose(axis, -1).contiguous()
    outer_scale_axis_last, outer_expanded = _outer_scale_axis_last_pytorch(
        data_axis_last,
        is_2d_outer=is_2d_outer,
        inner_block_size=block_size,
        outer_block_size=outer_block_size,
    )
    # Keep the per-outer-block normalization in FP32 through the inner QDQ to
    # match the production kernel (see ``convert_to_nvfp4_outer_block``): casting
    # back to bf16 here would add a redundant rounding step on top of the inner
    # NVFP4 quantization and break bit-exactness with the kernel.
    normalized = data_axis_last.float() / outer_expanded
    data_lp_axis_last, inner_scales_axis_last = convert_to_nvfp4_pytorch(
        normalized,
        block_size=block_size,
        axis=-1,
        is_2d_block=is_2d_inner,
        outer_scale=None,
        scale_format=scale_format,
    )
    # Ensure the inner-scale shape matches the production kernel's contract.
    return (
        data_lp_axis_last.transpose(axis, -1).contiguous(),
        inner_scales_axis_last.transpose(axis, -1).contiguous(),
        outer_scale_axis_last.transpose(axis, -1).contiguous(),
    )


def convert_from_nvfp4_outer_block_pytorch(
    data_lp: torch.Tensor,
    inner_scales: torch.Tensor,
    outer_scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    outer_block_size: int = OUTER_BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_inner: bool = False,
    is_2d_outer: bool = False,
    scale_format: str = "e4m3",
):
    """Pure-PyTorch reference for ``convert_from_nvfp4_outer_block``."""
    inner_dq = convert_from_nvfp4_pytorch(
        data_lp,
        inner_scales,
        output_dtype=output_dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_inner,
        outer_scale=None,
        scale_format=scale_format,
    )
    inner_dq_axis_last = inner_dq.transpose(axis, -1).contiguous()
    outer_axis_last = outer_scales.transpose(axis, -1).to(
        device=inner_dq.device, dtype=torch.float32,
    ).contiguous()
    shape = tuple(inner_dq_axis_last.shape)
    m, n = shape[-2], shape[-1]
    prefix = shape[:-2]
    if is_2d_outer:
        grouped_shape = (
            *prefix,
            m // outer_block_size, outer_block_size,
            n // outer_block_size, outer_block_size,
        )
        outer_expanded = outer_axis_last.unsqueeze(-2).unsqueeze(-1).expand(
            grouped_shape).reshape(shape)
    else:
        grouped_shape = (*prefix, m, n // outer_block_size, outer_block_size)
        outer_expanded = outer_axis_last.unsqueeze(-1).expand(
            grouped_shape).reshape(shape)
    result = (inner_dq_axis_last.float() * outer_expanded).to(output_dtype)
    return result.transpose(axis, -1).contiguous()
