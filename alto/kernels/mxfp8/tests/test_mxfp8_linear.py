# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from alto.kernels.mxfp8.mxfp8_linear import (
    blockwise_mxfp8_gemm_dequant_debug,
    MXFP8LinearFunction,
)
from alto.kernels.fp4.mxfp4.tests.utils import calc_cossim, calc_snr
from alto.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    is_cdna4,
)
from .utils import prepare_data


SMALL_DIVISIBLE_SHAPES = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
]

LINEAR_KERNEL_SHAPES = [
    (32, 32, 32),
    (256, 384, 128),
    (512, 1024, 256),
]

# With BLOCK_SIZE_K tied to the 32-wide quant block, 1e-4/1e-4 is enough for
# the larger kernel shapes below while still catching the regressed K=64
# variant, which needs much looser tolerances to pass.
LINEAR_KERNEL_ATOL = 1e-4
LINEAR_KERNEL_RTOL = 1e-4

# The dequantize + tl.dot debug path still uses the simpler K>=64 launch
# heuristic, so it needs a looser relative tolerance on the larger shapes even
# though it stays much closer to the reference than the regressed dot_scaled
# K=64 variant.
DEQUANT_DOT_KERNEL_ATOL = 1e-4
DEQUANT_DOT_KERNEL_RTOL = 5e-3


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

    a_lp, a_scales = torch.ops.alto.convert_to_mxfp8(
        a,
        mxfp_format=mxfp_format,
        axis=a_axis,
        is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.alto.convert_to_mxfp8(
        b,
        mxfp_format=mxfp_format,
        axis=b_axis,
        is_2d_block=False,
    )

    a_dq = torch.ops.alto.convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=data_type,
        axis=a_axis,
        is_2d_block=False,
    )
    b_dq = torch.ops.alto.convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=data_type,
        axis=b_axis,
        is_2d_block=False,
    )

    c = torch.ops.alto.blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        output_dtype=data_type,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)

    torch.testing.assert_close(
        c,
        c_ref,
        atol=1e-4,
        rtol=1e-4,
    )

# =============================================================================
# DEBUG TEST CASES
# =============================================================================

@pytest.mark.parametrize("shape", SMALL_DIVISIBLE_SHAPES)
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp8_linear_dequant_dot_kernel(
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

    a_lp, a_scales = torch.ops.alto.convert_to_mxfp8(
        a,
        mxfp_format=mxfp_format,
        axis=a_axis,
        is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.alto.convert_to_mxfp8(
        b,
        mxfp_format=mxfp_format,
        axis=b_axis,
        is_2d_block=False,
    )

    a_dq = torch.ops.alto.convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=data_type,
        axis=a_axis,
        is_2d_block=False,
    )
    b_dq = torch.ops.alto.convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=data_type,
        axis=b_axis,
        is_2d_block=False,
    )

    c = blockwise_mxfp8_gemm_dequant_debug(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        output_dtype=data_type,
        debug_print=False,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)

    assert torch.allclose(c_ref, c)



@pytest.mark.parametrize("shape", [(1, 32, 32, 32), (1, 512, 384, 128), (4, 1024, 1024, 2048)])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp8_linear_autograd_function(
    shape,
    mxfp_format,
    use_sr_grad,
    data_type,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required.")

    batch, m_dim, n_dim, k_dim = shape
    inputs = prepare_data((batch, m_dim, k_dim), data_type).requires_grad_(True)
    weights = prepare_data((n_dim, k_dim), data_type).requires_grad_(True)
    target = prepare_data((batch, m_dim, n_dim), data_type)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.detach().clone()
    grad_weights_ref = weights.grad.detach().clone()

    inputs.grad.zero_()
    weights.grad.zero_()

    outputs = MXFP8LinearFunction.apply(
        inputs,
        weights,
        mxfp_format,
        use_sr_grad,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(weights.grad, grad_weights_ref)
    dw_sim = calc_cossim(weights.grad, grad_weights_ref)

    print(
        f"output_snr={output_snr:.4f}, output_cossim={output_sim:.6f}, "
        f"dx_snr={dx_snr:.4f}, dx_cossim={dx_sim:.6f}, "
        f"dw_snr={dw_snr:.4f}, dw_cossim={dw_sim:.6f}"
    )

    if mxfp_format == "e4m3":
        min_output_snr = 24
        min_grad_snr = 15
    else:
        min_output_snr = 19
        min_grad_snr = 12

    assert output_snr > min_output_snr
    assert dx_snr > min_grad_snr
    assert dw_snr > min_grad_snr
