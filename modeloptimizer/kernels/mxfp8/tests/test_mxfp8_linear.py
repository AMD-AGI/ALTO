import pytest
import torch

import modeloptimizer.kernels.mxfp8.mxfp8_linear  # noqa: F401
from modeloptimizer.kernels.mxfp8.mxfp8_quantization import (
    is_cdna4,
)
from .utils import prepare_data


@pytest.mark.parametrize("shape", [(32, 32, 32), (256, 384, 128), (512, 1024, 256)])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp8_linear_kernel(
    shape,
    trans_a,
    trans_b,
    contiguous,
    mxfp_format,
    data_type,
):
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = shape

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

    a_lp, a_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        a,
        mxfp_format=mxfp_format,
        axis=a_axis,
        is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        b,
        mxfp_format=mxfp_format,
        axis=b_axis,
        is_2d_block=False,
    )

    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=data_type,
        axis=a_axis,
        is_2d_block=False,
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=data_type,
        axis=b_axis,
        is_2d_block=False,
    )

    c = torch.ops.modeloptimizer.blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        output_dtype=data_type,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)
    
    assert torch.allclose(c_ref, c)
