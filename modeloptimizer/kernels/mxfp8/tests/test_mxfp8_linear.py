import pytest
import torch

from modeloptimizer.kernels.mxfp8.mxfp8_linear import blockwise_mxfp8_gemm
from modeloptimizer.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    convert_from_mxfp8,
    convert_to_mxfp8,
    is_cdna4,
)
from .utils import prepare_data


@pytest.mark.parametrize("shape", [(64, 64, 64), (256, 384, 128), (512, 256, 512)])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_mxfp8_linear_kernel(
    shape,
    trans_a,
    trans_b,
    contiguous,
    mxfp_format,
    data_type,
    output_dtype,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required.")
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = shape
    block_size = BLOCK_SIZE_DEFAULT

    if trans_a:
        if contiguous:
            a = prepare_data((k, m), data_type)
        else:
            a = prepare_data((m, k), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -2
    else:
        if contiguous:
            a = prepare_data((m, k), data_type)
        else:
            a = prepare_data((k, m), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -1

    if trans_b:
        if contiguous:
            b = prepare_data((n, k), data_type)
        else:
            b = prepare_data((k, n), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -1
    else:
        if contiguous:
            b = prepare_data((k, n), data_type)
        else:
            b = prepare_data((n, k), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -2

    a_lp, a_scales = convert_to_mxfp8(
        a,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=a_axis,
        is_2d_block=False,
    )
    b_lp, b_scales = convert_to_mxfp8(
        b,
        block_size=block_size,
        mxfp_format=mxfp_format,
        axis=b_axis,
        is_2d_block=False,
    )

    a_dq = convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=a_axis,
        is_2d_block=False,
    )
    b_dq = convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=data_type,
        block_size=block_size,
        axis=b_axis,
        is_2d_block=False,
    )

    c = blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        block_size=block_size,
        output_dtype=output_dtype,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)
    c_ref = c_ref.to(output_dtype)

    if output_dtype == torch.float32:
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 5e-2, 5e-2

    torch.testing.assert_close(c, c_ref, atol=atol, rtol=rtol)
