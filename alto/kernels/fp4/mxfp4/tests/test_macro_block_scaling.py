import pytest
import torch

from alto.kernels.fp4.mxfp4.macro_block_scaling import (
    macro_block_scaling,
    macro_block_descaling,
    MACRO_BLOCK_SIZE,
)


@pytest.mark.parametrize("tensor_shape", [(1024, 2048)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
def test_macro_block_scaling(tensor_shape, axis, is_2d_block, data_type):
    x = torch.randn(tensor_shape).to(data_type)
    y, scale = macro_block_scaling(x, axis=axis, use_2d_block=is_2d_block)
    assert y.shape == x.shape
    transposed_shape = x.transpose(axis, -1).shape
    if is_2d_block:
        assert scale.shape == (transposed_shape[0] // MACRO_BLOCK_SIZE, transposed_shape[1] // MACRO_BLOCK_SIZE, 1)
    else:
        assert scale.shape == (transposed_shape[0], transposed_shape[1] // MACRO_BLOCK_SIZE, 1)

    scale_fp32 = ((scale.to(torch.int32) << 15) + (127 << 23)).view(torch.float32)
    assert torch.all(scale_fp32 >= 1) and torch.all(scale_fp32 <= 2)

    x_reconstructed = macro_block_descaling(y, scale, axis=axis, use_2d_block=is_2d_block)
    assert x_reconstructed.shape == x.shape
    assert x_reconstructed.dtype == x.dtype
    if data_type == torch.bfloat16:
        assert torch.allclose(x_reconstructed, x, rtol=1e-2, atol=1e-3)
    else:
        assert torch.allclose(x_reconstructed, x)
