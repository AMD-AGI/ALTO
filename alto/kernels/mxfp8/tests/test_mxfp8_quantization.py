# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from alto.kernels.mxfp8.mxfp8_quantization import convert_to_mxfp8, convert_from_mxfp8, calculate_mxfp8_scales, is_cdna4
from .utils import prepare_data, convert_to_mxfp8_pytorch, convert_from_mxfp8_pytorch


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("pattern", ["random", "zeros", "large_e5m2"])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mxfp8_quantization(tensor_shape, axis, data_type, mxfp_format, pattern, is_2d_block, use_sr, use_asm,
                            block_size):
    if use_asm and not is_cdna4():
        pytest.skip("ASM mode is only available on CDNA4 hardwares.")
    if use_sr and not use_asm:
        pytest.skip("use_sr=True requires use_asm=True.")

    x = prepare_data(tensor_shape, data_type, pattern=pattern)

    # 1. PyTorch Reference
    data_lp_ref, scales_ref = convert_to_mxfp8_pytorch(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    # 2. Triton Implementation
    data_lp, scales = convert_to_mxfp8(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
        use_sr=use_sr,
        use_asm=use_asm,
    )

    # 3. Verify scales (SR does not affect scale computation)
    assert torch.all(scales_ref == scales).item(), \
        f"Scale mismatch for format {mxfp_format}, shape {tensor_shape}, pattern {pattern}"

    # 4. Verify quantized data
    if not use_sr:
        assert torch.all(data_lp_ref == data_lp).item(), \
            f"Data mismatch for format {mxfp_format}, shape {tensor_shape}, pattern {pattern}"
    else:
        ref_u8 = data_lp_ref.view(torch.uint8).to(torch.int16)
        out_u8 = data_lp.view(torch.uint8).to(torch.int16)
        assert torch.all(torch.abs(ref_u8 - out_u8) <= 1).item(), \
            f"SR data differs from RTE by more than 1 ULP for format {mxfp_format}, shape {tensor_shape}, pattern {pattern}"


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("pattern", ["random", "zeros", "large_e5m2"])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mxfp8_dequantization(tensor_shape, axis, data_type, mxfp_format, pattern, is_2d_block, use_asm, block_size):
    x = prepare_data(tensor_shape, data_type, pattern=pattern)

    # 1. Quantize first
    data_lp, scales = convert_to_mxfp8(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
        use_asm=use_asm,
    )

    # 2. PyTorch Reference Dequantization
    data_hp_ref = convert_from_mxfp8_pytorch(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    # 3. Triton Dequantization
    data_hp = convert_from_mxfp8(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        use_asm=use_asm,
    )

    # 4. Verify dequantized data matches reference
    assert data_hp.dtype == data_hp_ref.dtype, \
        f"Dtype mismatch: expected {data_hp_ref.dtype}, got {data_hp.dtype}"
    assert data_hp.shape == data_hp_ref.shape, \
        f"Shape mismatch: expected {data_hp_ref.shape}, got {data_hp.shape}"
    assert torch.all(data_hp == data_hp_ref).item(), \
        f"Dequant mismatch for format {mxfp_format}, shape {tensor_shape}, pattern {pattern}"


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("use_asm", [False, True])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mxfp8_e2e(tensor_shape, axis, data_type, mxfp_format, is_2d_block, use_sr, use_asm, block_size):
    """E2E test: independently run quant+dequant on both PyTorch and Triton, compare final outputs."""
    if use_asm and not is_cdna4():
        pytest.skip("ASM mode is only available on CDNA4 hardwares.")
    if use_sr and not use_asm:
        pytest.skip("use_sr=True requires use_asm=True.")

    x = prepare_data(tensor_shape, data_type, pattern="random")

    # PyTorch reference: quant -> dequant
    data_lp_ref, scales_ref = convert_to_mxfp8_pytorch(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
    )
    x_dq_ref = convert_from_mxfp8_pytorch(
        data_lp_ref,
        scales_ref,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    # Triton: quant -> dequant
    data_lp, scales = convert_to_mxfp8(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
        use_sr=use_sr,
        use_asm=use_asm,
    )
    x_dq = convert_from_mxfp8(
        data_lp,
        scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        use_asm=use_asm,
    )

    assert torch.all(scales_ref == scales).item(), \
        f"Scale mismatch for format {mxfp_format}, shape {tensor_shape}"

    if not use_sr:
        assert torch.all(data_lp.view(torch.uint8) == data_lp_ref.view(torch.uint8)).item(), \
            f"Quantized data mismatch for format {mxfp_format}, shape {tensor_shape}"
        assert torch.all(x_dq == x_dq_ref).item(), \
            f"E2E dequant mismatch for format {mxfp_format}, shape {tensor_shape}"
    else:
        dq_mae = (x_dq_ref - x_dq).abs().mean().item()
        threshold = 1.0 if is_2d_block else 0.5
        assert dq_mae < threshold, \
            f"E2E SR dequant MAE too large ({dq_mae}) for format {mxfp_format}, shape {tensor_shape}"


@pytest.mark.parametrize("tensor_shape", [(128, 64), (2048, 2048), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("pattern", ["random", "zeros", "large_e5m2"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_mxfp8_scale(tensor_shape, axis, data_type, mxfp_format, is_2d_block, pattern, block_size):
    """Test scale calculation for 1D and 2D block modes."""
    x = prepare_data(tensor_shape, data_type, pattern=pattern)

    _, scales_ref = convert_to_mxfp8_pytorch(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    scales = calculate_mxfp8_scales(
        x,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=axis,
        is_2d_block=is_2d_block,
    )

    assert torch.all(scales_ref == scales).item(), \
        f"Scale mismatch for format {mxfp_format}, shape {tensor_shape}, axis {axis}, pattern {pattern}"
