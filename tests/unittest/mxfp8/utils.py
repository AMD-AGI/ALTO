# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Tuple
import torch
from alto.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    SUPPORTED_FORMATS,
    FORMAT_TO_TARGET_MAX,
    FORMAT_TO_MBITS,
)
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M

# Re-exported so test call-sites can ``from .utils import calc_snr, calc_cossim``
# (mirrors mxfp4/nvfp4 utils); the single source of truth lives in
# ``alto.kernels.fp4.testing_utils``.
from alto.kernels.fp4.testing_utils import calc_snr, calc_cossim  # noqa: F401


def make_indices(num_groups, num_experts, device):
    """Contiguous routing: every ALIGN_SIZE_M-token group shares one expert.

    Matches the mxfp4/nvfp4 convention (random per-group expert id). Calls to
    ``prepare_data`` reset the global seed, so the ``randint`` draws that follow
    them are deterministic across runs without an explicit seed here.
    """
    indices = torch.zeros(num_groups * ALIGN_SIZE_M, dtype=torch.int32, device=device)
    for g in range(num_groups):
        e = torch.randint(0, num_experts, (1,), device=device, dtype=torch.int32).item()
        indices[g * ALIGN_SIZE_M:(g + 1) * ALIGN_SIZE_M] = e
    return indices


def prepare_data(tensor_shape, data_type, pattern="random"):
    """
    Prepare test data with specified pattern.
    
    Args:
        tensor_shape: Shape of the output tensor.
        data_type: Data type (torch.float32 or torch.bfloat16).
        pattern: Data pattern - "random", "zeros", or "large_e5m2".
    """
    torch.manual_seed(1234)
    device = torch.device("cuda")

    if pattern == "random":
        x = torch.randn(tensor_shape, dtype=data_type, device=device)
        p_mask = torch.bernoulli(torch.ones_like(x) * 0.005)
        x += 100 * torch.randn_like(x) * p_mask
    elif pattern == "zeros":
        x = torch.zeros(tensor_shape, dtype=data_type, device=device)
    elif pattern == "large_e5m2":
        x = torch.ones(tensor_shape, dtype=data_type, device=device) * 50000.0
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return x


def convert_to_mxfp8_pytorch(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    mxfp_format: str = "e4m3",
    axis: int = -1,
    is_2d_block: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation for MXFP8 quantization (ground truth for tests)."""
    assert data_hp.dtype in [torch.float32, torch.bfloat16]
    assert mxfp_format in SUPPORTED_FORMATS, \
        f"Unsupported format: {mxfp_format}. Supported: {SUPPORTED_FORMATS}"

    # Map binary distribution parameters of original data type
    if data_hp.dtype == torch.float32:
        hp_int_dtype = torch.int32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = torch.int16
        hp_mbits = 7
        hp_ebits = 8

    INPUT_SIGN_BIT = 1

    # Configure parameters according to the target format
    if mxfp_format == "e4m3":
        fp8_dtype = torch.float8_e4m3fn
    else:
        fp8_dtype = torch.float8_e5m2
    target_max_pow2 = FORMAT_TO_TARGET_MAX[mxfp_format]
    mbits = FORMAT_TO_MBITS[mxfp_format]

    # Dimension adjustment and block division logic
    data_hp = data_hp.transpose(axis, -1)
    orig_shape = data_hp.shape

    if is_2d_block:
        # 2D block: reshape to [..., M/block, block, N/block, block]
        new_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, block_size, orig_shape[-1] // block_size,
                     block_size)
        data_hp_blocked = data_hp.reshape(new_shape)
        # max over the two block dimensions (axis -1 and -3)
        max_abs = torch.amax(torch.abs(data_hp_blocked), dim=(-1, -3))
    else:
        # 1D block: reshape to [..., N/block, block]
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
        data_hp_blocked = data_hp.reshape(new_shape)
        max_abs = torch.amax(torch.abs(data_hp_blocked), dim=-1)

    # Extract exponent via bit manipulation (rounding based on target format's mantissa bits)
    max_abs = max_abs.view(hp_int_dtype)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + INPUT_SIGN_BIT)) - 1) << hp_mbits
    max_abs = ((max_abs + val_to_add) & mask) >> hp_mbits
    scales = max_abs - target_max_pow2

    scales = torch.clamp(scales, min=1).to(torch.uint8)

    # Broadcast scales and apply to original data
    # Use FP32 for division to match Triton implementation precision
    scales_fp32 = (scales.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    if is_2d_block:
        scales_fp32 = scales_fp32.unsqueeze(-3)
    scaled_data = (data_hp_blocked.to(torch.float32) / scales_fp32).reshape(orig_shape)

    # Clamp to FP8 range before conversion to match Triton's saturating behavior.
    # Power-of-2 scale rounding may leave values with mantissa in [1.75, carry_threshold)
    # above fp8_max. PyTorch's .to(fp8) behavior for out-of-range values is undefined;
    # Triton saturates natively. Explicit clamp aligns the two.
    fp8_max = torch.finfo(fp8_dtype).max
    scaled_data = torch.clamp(scaled_data, min=-fp8_max, max=fp8_max)
    data_lp = scaled_data.to(fp8_dtype)

    # Return with proper scale shape
    return data_lp.transpose(axis, -1), scales.transpose(axis, -1)


def convert_from_mxfp8_pytorch(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
) -> torch.Tensor:
    """PyTorch reference implementation for MXFP8 dequantization (ground truth for tests)."""
    assert output_dtype in [torch.float32, torch.bfloat16]

    data_lp = data_lp.transpose(axis, -1)
    scales = scales.transpose(axis, -1)
    orig_shape = data_lp.shape

    if is_2d_block:
        # Reshape to [..., M/block, block, N/block, block]
        new_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, block_size, orig_shape[-1] // block_size,
                     block_size)
        # Scales: [..., M/block, N/block] -> [..., M/block, 1, N/block, 1]
        scale_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, 1, orig_shape[-1] // block_size, 1)
    else:
        # Reshape to [*, N/block_size, block_size]
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
        # Scales: [..., N/block] -> [..., N/block, 1]
        scale_shape = (*orig_shape[:-1], orig_shape[-1] // block_size, 1)

    data_lp = data_lp.reshape(new_shape)

    # Convert scales (uint8 exponent) to fp32 scale factor: 2^scales
    scales_fp32 = (scales.to(torch.int32) << 23).view(torch.float32).reshape(scale_shape)

    # Dequantize: y = x * 2^scales
    data_hp = (data_lp.to(torch.float32) * scales_fp32).reshape(orig_shape)

    if output_dtype == torch.bfloat16:
        data_hp = data_hp.to(torch.bfloat16)

    return data_hp.transpose(axis, -1)


def _mxfp8_qdq(x: torch.Tensor, axis: int, is_2d_block: bool, block_size: int = BLOCK_SIZE_DEFAULT) -> torch.Tensor:
    """Round-trip a tensor through pure-PyTorch MXFP8 e4m3 quantize+dequantize.

    Returns the value an mxfp8 kernel operand would actually see.
    """
    lp, s = convert_to_mxfp8_pytorch(x, block_size=block_size, mxfp_format="e4m3", axis=axis, is_2d_block=is_2d_block)
    return convert_from_mxfp8_pytorch(lp, s, output_dtype=x.dtype, block_size=block_size, axis=axis,
                                      is_2d_block=is_2d_block)


def mxfp8_attention_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool,
    block_n: int = 64,
    block_size: int = BLOCK_SIZE_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch golden reference for the MXFP8 (e4m3) flash-attention forward.

    Faithfully replicates what ``triton_flash_attention_mxfp8.attn_fwd`` computes,
    so a kernel-vs-this-reference gap isolates Triton port bugs (masking /
    online-softmax / LSE) from mxfp8 quantization error itself:

      * Q/K/V are quantized 2D-block along head_dim (``is_2d_block=True``,
        ``axis=-1``), matching the autograd forward.
      * The softmax probabilities ``P`` are quantized **per key block** exactly
        like the kernel: within the online-softmax loop, using the *running* row
        max (not the global max), 1D-block along the key axis with
        ``block_size`` groups.
      * Contains no ``tl.dot_scaled``: every matmul is a plain fp32 matmul on the
        dequantized operands, so this runs on CPU / any device (no CDNA4).

    Args:
        q, k, v: ``[batch, nheads, seqlen, head_dim]`` (bhsd), fp32/bf16. GQA is
            supported (``nheads_k`` may be < ``nheads_q``).
        sm_scale: softmax scale applied to ``Q @ K^T``.
        causal: top-left causal mask ``key_j <= query_i`` (matches
            ``F.sdpa(is_causal=True)`` on bhsd tensors in PyTorch 2.x).
        block_n: key-block width, must match the kernel ``BLOCK_N`` (default 64).
        block_size: MXFP8 quant block, must match ``QUANT_BLOCK_SIZE`` (default 32).

    Returns:
        (o, softmax_lse): output ``[batch, nheads_q, seqlen_q, head_dim_v]`` and
        log-sum-exp ``[batch, nheads_q, seqlen_q]``.
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    b, hq, sq, dqk = q.shape
    _, hk, sk, _ = k.shape
    dv = v.shape[-1]
    assert hq % hk == 0, f"nheads_q ({hq}) must be a multiple of nheads_k ({hk})"
    n_rep = hq // hk

    # Quantize Q/K/V (2D block along head_dim) then dequantize -> operands the kernel sees.
    q_dq = _mxfp8_qdq(q, axis=-1, is_2d_block=True).to(torch.float32)
    k_dq = _mxfp8_qdq(k, axis=-1, is_2d_block=True).to(torch.float32)
    v_dq = _mxfp8_qdq(v, axis=-1, is_2d_block=True).to(torch.float32)

    # Expand K/V heads for GQA (head_q h -> head_k h // n_rep, matching off_h_k).
    if n_rep > 1:
        k_dq = k_dq.repeat_interleave(n_rep, dim=1)
        v_dq = v_dq.repeat_interleave(n_rep, dim=1)

    device = q.device
    # Real -inf (not finfo.min): masked scores must dequantize to exp(-inf)=0.
    # A finite sentinel would make a fully-masked block compute exp(0)=1.
    neg_inf = float("-inf")

    m_i = torch.full((b, hq, sq), float("-inf"), dtype=torch.float32, device=device)
    l_i = torch.zeros((b, hq, sq), dtype=torch.float32, device=device)
    acc = torch.zeros((b, hq, sq, dv), dtype=torch.float32, device=device)

    q_pos = torch.arange(sq, device=device)

    for j0 in range(0, sk, block_n):
        j1 = min(j0 + block_n, sk)
        k_blk = k_dq[:, :, j0:j1, :]  # [b, hq, bn, d]
        v_blk = v_dq[:, :, j0:j1, :]

        s = torch.matmul(q_dq, k_blk.transpose(-1, -2)) * sm_scale  # [b, hq, sq, bn]

        if causal:
            key_pos = torch.arange(j0, j1, device=device)
            allowed = key_pos[None, :] <= q_pos[:, None]  # [sq, bn]
            s = torch.where(allowed[None, None, :, :], s, torch.full_like(s, neg_inf))

        m_ij = torch.maximum(m_i, s.max(dim=-1).values)  # [b, hq, sq]
        p = torch.exp(s - m_ij[..., None])  # unnormalized, running max
        # Rows fully masked in this block yield exp(neg_inf - (-inf)) issues; zero them.
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        l_ij = p.sum(dim=-1)  # [b, hq, sq]

        # Quantize P per key block exactly like the kernel (1D block along key axis).
        p_dq = _mxfp8_qdq(p.to(torch.float32), axis=-1, is_2d_block=False, block_size=block_size)

        alpha = torch.exp(m_i - m_ij)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        acc = acc * alpha[..., None] + torch.matmul(p_dq, v_blk)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    l_safe = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
    o = acc / l_safe[..., None]
    softmax_lse = m_i + torch.log(l_safe)
    # Fully-masked query rows (can happen with causal when sk < sq): zero output.
    fully_masked = ~torch.isfinite(m_i) | (l_i <= 0)
    o = torch.where(fully_masked[..., None], torch.zeros_like(o), o)
    softmax_lse = torch.where(fully_masked, torch.zeros_like(softmax_lse), softmax_lse)

    return o.to(q.dtype), softmax_lse
