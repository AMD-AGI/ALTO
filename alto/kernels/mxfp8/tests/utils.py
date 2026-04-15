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
        new_shape = (*orig_shape[:-2],
                     orig_shape[-2] // block_size, block_size,
                     orig_shape[-1] // block_size, block_size)
        data_hp_blocked = data_hp.reshape(new_shape)
        # max over the two block dimensions (axis -1 and -3)
        max_abs = torch.amax(torch.abs(data_hp_blocked), dim=(-1, -3))
    else:
        # 1D block: reshape to [..., N/block, block]
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size,
                     block_size)
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
    scales_fp32 = (scales.to(torch.int32) << 23).view(
        torch.float32).unsqueeze(-1)
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
        new_shape = (*orig_shape[:-2],
                     orig_shape[-2] // block_size, block_size,
                     orig_shape[-1] // block_size, block_size)
        # Scales: [..., M/block, N/block] -> [..., M/block, 1, N/block, 1]
        scale_shape = (*orig_shape[:-2], orig_shape[-2] // block_size, 1,
                       orig_shape[-1] // block_size, 1)
    else:
        # Reshape to [*, N/block_size, block_size]
        new_shape = (*orig_shape[:-1], orig_shape[-1] // block_size,
                     block_size)
        # Scales: [..., N/block] -> [..., N/block, 1]
        scale_shape = (*orig_shape[:-1], orig_shape[-1] // block_size, 1)

    data_lp = data_lp.reshape(new_shape)

    # Convert scales (uint8 exponent) to fp32 scale factor: 2^scales
    scales_fp32 = (scales.to(torch.int32) << 23).view(
        torch.float32).reshape(scale_shape)

    # Dequantize: y = x * 2^scales
    data_hp = (data_lp.to(torch.float32) * scales_fp32).reshape(orig_shape)

    if output_dtype == torch.bfloat16:
        data_hp = data_hp.to(torch.bfloat16)
    
    return data_hp.transpose(axis, -1)
