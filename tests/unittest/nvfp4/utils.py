# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from torch import Tensor
from alto.kernels.fp4.nvfp4.nvfp_quantization import BLOCK_SIZE_DEFAULT

# Re-exported so existing ``from .utils import calc_snr, calc_cossim``
# call-sites keep working; the single source of truth lives in
# ``alto.kernels.fp4.testing_utils``.
from alto.kernels.fp4.testing_utils import calc_snr, calc_cossim  # noqa: F401


F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


def prepare_data(tensor_shape, data_type, pattern="random"):
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
    """
    torch.manual_seed(1234)
    device = torch.device("cuda")

    if pattern == "random":
        x = torch.randn(tensor_shape, dtype=data_type, device=device)
        p_mask = torch.bernoulli(torch.ones_like(x) * 0.005)
        x += 100 * torch.randn_like(x) * p_mask
    elif pattern == "zeros":
        x = torch.zeros(tensor_shape, dtype=data_type, device=device)
    elif pattern == "large":
        x = torch.ones(tensor_shape, dtype=data_type, device=device) * 5000.0
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
    assert scale_format == "e4m3", f"scale_format={scale_format!r} not yet supported"

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
        inner_scale_raw = (max_abs / outer_scale / F4_E2M1_MAX).clamp(min=E4M3_EPS, max=F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(torch.float8_e4m3fn).to(torch.float32)
        quant_scale = inner_scale * outer_scale
    else:
        inner_scale_raw = (max_abs / F4_E2M1_MAX).clamp(min=E4M3_EPS, max=F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(torch.float8_e4m3fn).to(torch.float32)
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
# Four Over Six (4/6) NVFP4 reference implementation (P1 eager reference)
#
# FP4 E2M1 has no representable values in the open interval (4, 6), so under
# the standard NVFP4 rule (scale each block's max to 6) the near-maximal
# values that land around 5 incur large rounding error -- empirically the
# dominant source of NVFP4 quantization error (Cook et al., "Four Over Six",
# arXiv:2512.02010).  4/6 instead lets each block scale its max to *either* 4
# or 6: it quantizes the block both ways, dequantizes, and keeps whichever has
# the lower per-block error (MAE for pre-training, MSE for PTQ).
#
# Because 4/6 only changes how the stored ``(fp4_value, inner_scale)`` pair is
# *chosen*, the dequantization path is unchanged: ``convert_from_nvfp4_pytorch``
# reconstructs a 4/6-encoded tensor with no modification.
#
# The global/outer scale uses an E4M3 ceiling of 256 rather than 448 so that
# the block holding the tensor amax can still pick M=4 without its block scale
# (448 * 6 / 4 = 672) overflowing E4M3's 448 maximum; with 256 the worst case
# is 256 * 6 / 4 = 384, which is exactly representable in E4M3.
# ---------------------------------------------------------------------------

F8E4M3_4O6_CEIL = 256.0


def compute_dynamic_outer_scale_4o6_pytorch(
    data_hp: torch.Tensor,
    scale_ceiling: float = F8E4M3_4O6_CEIL,
) -> torch.Tensor:
    """4/6 outer (global) FP32 scale: ``amax / (scale_ceiling * F4_E2M1_MAX)``.

    Uses ``scale_ceiling=256`` instead of the standard 448 so the amax block
    can select M=4 without overflowing the E4M3 block scale.  Mirrors the
    div-by-zero floor used by the production ``compute_dynamic_outer_scale``.
    """
    amax = data_hp.float().abs().max()
    outer_scale = (amax / (scale_ceiling * F4_E2M1_MAX)).clamp(min=1.0e-30)
    return outer_scale.to(dtype=torch.float32).reshape(1)


def _nvfp4_quant_candidate(grouped, max_abs, outer_scale, m_fp4, is_2d_block):
    """Quantize every block with a single FP4 ceiling ``m_fp4`` (4 or 6).

    Returns ``(fp4_unpacked, inner_scale_e4m3, dequantized)`` so the caller can
    compare per-block reconstruction error across the two ceilings.
    """
    if outer_scale is not None:
        outer_scale = outer_scale.float().to(grouped.device)
        inner_scale_raw = (max_abs / outer_scale / m_fp4).clamp(min=E4M3_EPS, max=F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(torch.float8_e4m3fn).to(torch.float32)
        quant_scale = inner_scale * outer_scale
    else:
        inner_scale_raw = (max_abs / m_fp4).clamp(min=E4M3_EPS, max=F8E4M3_MAX)
        inner_scale = inner_scale_raw.to(torch.float8_e4m3fn).to(torch.float32)
        quant_scale = inner_scale

    if is_2d_block:
        quant_scale_exp = quant_scale.unsqueeze(-2).unsqueeze(-1)
    else:
        quant_scale_exp = quant_scale.unsqueeze(-1)

    scaled = grouped / quant_scale_exp
    fp4_unpacked = f32_to_f4_unpacked(scaled)
    dequant = f4_unpacked_to_f32(fp4_unpacked) * quant_scale_exp
    return fp4_unpacked, inner_scale, dequant


def convert_to_nvfp4_4o6_pytorch(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: torch.Tensor = None,
    select_metric: str = "mae",
    return_choice: bool = False,
):
    """PyTorch reference for NVFP4 quantization with Four Over Six block scaling.

    Drop-in analogue of ``convert_to_nvfp4_pytorch`` whose output ``(data_lp,
    inner_scale)`` is consumed by the *unchanged* ``convert_from_nvfp4_pytorch``.

    Args:
        select_metric: per-block selection rule, ``"mae"`` (best for
            pre-training per the paper) or ``"mse"`` (best for PTQ).
        return_choice: also return the boolean mask of blocks that selected
            M=4 (useful for selection-correctness tests).
    """
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    assert select_metric in ("mae", "mse"), f"unknown select_metric={select_metric!r}"

    data_hp = data_hp.transpose(axis, -1)
    ori_shape = data_hp.shape

    if is_2d_block:
        m_blocks = ori_shape[-2] // block_size
        n_blocks = ori_shape[-1] // block_size
        grouped = data_hp.reshape(
            *ori_shape[:-2], m_blocks, block_size, n_blocks, block_size).float()
        max_abs = grouped.abs().amax(dim=-1).amax(dim=-2)
    else:
        data_hp_2d = data_hp.reshape(-1, ori_shape[-1])
        n_rows, n_cols = data_hp_2d.shape
        num_blocks = n_cols // block_size
        grouped = data_hp_2d.reshape(n_rows, num_blocks, block_size).float()
        max_abs = grouped.abs().amax(dim=-1)

    fp4_6, inner_6, dequant_6 = _nvfp4_quant_candidate(
        grouped, max_abs, outer_scale, F4_E2M1_MAX, is_2d_block)
    fp4_4, inner_4, dequant_4 = _nvfp4_quant_candidate(
        grouped, max_abs, outer_scale, 4.0, is_2d_block)

    def _per_block_err(dequant):
        diff = dequant - grouped
        diff = diff.abs() if select_metric == "mae" else diff * diff
        if is_2d_block:
            # reduce the two block_size dims, leaving [..., m_blocks, n_blocks]
            return diff.mean(dim=-1).mean(dim=-2)
        return diff.mean(dim=-1)

    choose_4 = _per_block_err(dequant_4) < _per_block_err(dequant_6)

    inner_scale = torch.where(choose_4, inner_4, inner_6)
    if is_2d_block:
        choose_4_q = choose_4.unsqueeze(-2).unsqueeze(-1).expand_as(fp4_6)
    else:
        choose_4_q = choose_4.unsqueeze(-1).expand_as(fp4_6)
    fp4_unpacked = torch.where(choose_4_q, fp4_4, fp4_6)

    if is_2d_block:
        fp4_unpacked = fp4_unpacked.reshape(ori_shape)
    else:
        fp4_unpacked = fp4_unpacked.reshape(n_rows, n_cols).reshape(ori_shape)
        inner_scale = inner_scale.reshape(*ori_shape[:-1], num_blocks)

    data_lp = pack_uint4(fp4_unpacked)

    data_lp = data_lp.transpose(axis, -1)
    inner_scale = inner_scale.transpose(axis, -1)
    if return_choice:
        return data_lp, inner_scale, choose_4
    return data_lp, inner_scale
