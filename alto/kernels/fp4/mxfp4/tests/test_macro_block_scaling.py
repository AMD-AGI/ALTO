import pytest
import torch

from alto.kernels.fp4.mxfp4.macro_block_scaling import (
    MACRO_BLOCK_SIZE,
    _macro_block_descaling_torch_impl,
    _macro_block_descaling_triton_impl,
    _macro_block_scaling_torch_impl,
    _macro_block_scaling_triton_impl,
    macro_block_descaling,
    macro_block_scaling,
)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expected_scale_shape(x_shape: tuple[int, ...], axis: int, use_2d_block: bool) -> tuple[int, ...]:
    t_shape = list(x_shape)
    axis = axis % len(t_shape)
    t_shape[axis], t_shape[-1] = t_shape[-1], t_shape[axis]
    m, n = t_shape[-2], t_shape[-1]
    if use_2d_block:
        return tuple(t_shape[:-2] + [_ceil_div(m, MACRO_BLOCK_SIZE), _ceil_div(n, MACRO_BLOCK_SIZE)])
    return tuple(t_shape[:-2] + [m, _ceil_div(n, MACRO_BLOCK_SIZE)])


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("tensor_shape", [(1024, 2048), (513, 385), (2, 513, 385)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_macro_block_scaling_scale_shape_and_roundtrip(tensor_shape, axis, is_2d_block, data_type):
    x = torch.randn(tensor_shape, device="cuda", dtype=data_type)
    y, scale = macro_block_scaling(x, axis=axis, use_2d_block=is_2d_block)

    assert y.shape == x.shape
    assert scale.shape == _expected_scale_shape(tensor_shape, axis, is_2d_block)

    scale_fp32 = ((scale.to(torch.int32) << 15) + (127 << 23)).view(torch.float32)
    assert torch.all(scale_fp32 >= 1) and torch.all(scale_fp32 <= 2)

    x_reconstructed = macro_block_descaling(y, scale, axis=axis, use_2d_block=is_2d_block)
    assert x_reconstructed.shape == x.shape
    assert x_reconstructed.dtype == x.dtype
    if data_type == torch.bfloat16:
        assert torch.allclose(x_reconstructed, x, rtol=1e-2, atol=1e-3)
    else:
        assert torch.allclose(x_reconstructed, x, rtol=1e-6, atol=1e-6)


@CUDA_ONLY
@pytest.mark.parametrize("tensor_shape", [(512, 384), (513, 385), (2, 512, 384), (2, 513, 385)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_macro_block_scaling_triton_matches_torch(tensor_shape, axis, is_2d_block, data_type):
    x = torch.randn(tensor_shape, device="cuda", dtype=data_type)

    y_torch, scale_torch = _macro_block_scaling_torch_impl(x, axis=axis, use_2d_block=is_2d_block)
    y_triton, scale_triton = _macro_block_scaling_triton_impl(x, axis=axis, use_2d_block=is_2d_block)

    assert y_triton.shape == y_torch.shape
    assert scale_triton.shape == scale_torch.shape
    assert scale_triton.dtype == scale_torch.dtype == torch.uint8
    assert torch.equal(scale_triton, scale_torch)
    if data_type == torch.bfloat16:
        assert torch.allclose(y_triton, y_torch, rtol=1e-2, atol=1e-3)
    else:
        assert torch.allclose(y_triton, y_torch, rtol=1e-6, atol=1e-6)


@CUDA_ONLY
@pytest.mark.parametrize("tensor_shape", [(512, 384), (513, 385), (2, 512, 384), (2, 513, 385)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_macro_block_descaling_triton_matches_torch(tensor_shape, axis, is_2d_block, data_type):
    x = torch.randn(tensor_shape, device="cuda", dtype=data_type)
    y, scale = _macro_block_scaling_torch_impl(x, axis=axis, use_2d_block=is_2d_block)

    x_torch = _macro_block_descaling_torch_impl(y, scale, axis=axis, use_2d_block=is_2d_block)
    x_triton = _macro_block_descaling_triton_impl(y, scale, axis=axis, use_2d_block=is_2d_block)

    assert x_triton.shape == x_torch.shape
    if data_type == torch.bfloat16:
        assert torch.allclose(x_triton, x_torch, rtol=1e-2, atol=1e-3)
    else:
        assert torch.allclose(x_triton, x_torch, rtol=1e-6, atol=1e-6)
