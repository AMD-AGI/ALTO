# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from alto.nn import DecomposedLinear
from alto.kernels.fp4.mxfp4.mxfp_quantization import convert_to_mxfp4, convert_from_mxfp4
from alto.kernels.dispatch import TrainingOpConfig, swap_params

from utils import prepare_data, calc_cossim, calc_snr


@pytest.mark.parametrize("in_features", [128, 256, 512])
@pytest.mark.parametrize("out_features", [128, 256, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [16, 32, 64])
def test_decomposed_linear(in_features, out_features, bias, lora_rank):
    STD = 0.1
    linear = nn.Linear(in_features, out_features, bias=bias)
    linear.weight.data.normal_(mean=0, std=STD)
    if bias:
        linear.bias.data.normal_(mean=0, std=STD)
    x = torch.randn(1, in_features)

    decomposed_linear = DecomposedLinear.from_linear(linear, lora_rank=lora_rank)
    decomposed_linear.u.data.normal_(mean=0, std=STD)
    decomposed_linear.v.data.normal_(mean=0, std=STD)
    decomposed_linear.sigma.data.normal_(mean=0, std=STD)
    y_decomposed = decomposed_linear(x)

    combined_weight = decomposed_linear.weight + decomposed_linear.u.T @ torch.diag(
        decomposed_linear.sigma) @ decomposed_linear.v.T
    y_ref = F.linear(x, combined_weight, bias=decomposed_linear.bias)

    assert torch.allclose(y_ref, y_decomposed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("in_features", [128, 256, 512])
@pytest.mark.parametrize("out_features", [128, 256, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [32, 64])
@pytest.mark.parametrize("precision", ["mxfp4"])
def test_decomposed_linear_quantization(in_features, out_features, bias, lora_rank, precision):
    STD = 0.1
    decomposed_linear = DecomposedLinear(
        DecomposedLinear.Config(in_features=in_features, out_features=out_features, bias=bias,
                                lora_rank=lora_rank)).to("cuda")
    decomposed_linear.weight.data.normal_(mean=0, std=STD)
    if bias:
        decomposed_linear.bias.data.normal_(mean=0, std=STD)
    decomposed_linear.u.data.normal_(mean=0, std=STD)
    decomposed_linear.v.data.normal_(mean=0, std=STD)
    decomposed_linear.sigma.data.normal_(mean=0, std=STD)

    x = torch.randn(1, in_features, device="cuda")
    y_ref = decomposed_linear(x)

    quant_op_config = TrainingOpConfig(
        precision=precision,
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=False,
        use_dge=False,
    )
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="weight")
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="u")
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="v")
    y = decomposed_linear(x)

    max_diff = torch.max(torch.abs(y_ref - y))
    mean_diff = torch.mean(torch.abs(y_ref - y))
    snr = 20 * torch.log10(torch.norm(y_ref) / torch.norm(y_ref - y))
    cossim = torch.nn.functional.cosine_similarity(y_ref.flatten(), y.flatten(), dim=0)
    print(f"snr={snr}, cossim={cossim}, max_diff={max_diff}, mean_diff={mean_diff}")
    assert snr > 10, f"SNR too low: {snr}"


@pytest.mark.parametrize("shape", [(4, 1024, 1024, 2048)])
@pytest.mark.parametrize("data_type", [torch.float32])
def test_decomposed_linear_with_svd(shape, data_type):
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)

    # Create a target for gradient computation
    target = prepare_data((B, M, N), data_type)

    # PyTorch reference implementation
    linear = torch.nn.Linear(K, N, bias=False)
    linear.weight = torch.nn.Parameter(weights)
    outputs_ref = linear(inputs)

    # Compute loss and gradients with PyTorch
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)

    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = linear.weight.grad.clone()

    # Reset gradients
    inputs.grad.zero_()
    linear.weight.grad.zero_()

    decomposed_linear = DecomposedLinear.from_linear(linear, lora_rank=32)

    with torch.no_grad():
        w_fp4, w_scales = convert_to_mxfp4(weights, axis=-1, is_2d_block=True)
        w_qdq = convert_from_mxfp4(
            w_fp4,
            w_scales,
            output_dtype=data_type,
            axis=-1,
            is_2d_block=True,
        )
        decomposed_linear.weight.data.copy_(w_qdq)

        delta = weights - w_qdq
        u, sigma, v = torch.svd_lowrank(delta, q=32)

        decomposed_linear.u.data.copy_(u.T)
        decomposed_linear.sigma.data.copy_(sigma)
        decomposed_linear.v.data.copy_(v)

    # decomposed_linear.u.data.normal_(mean=0, std=STD)
    # decomposed_linear.v.data.normal_(mean=0, std=STD)
    # decomposed_linear.sigma.data.normal_(mean=0, std=STD)

    quant_op_config = TrainingOpConfig(
        precision="mxfp4",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=False,
        use_dge=False,
    )
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="weight")
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="u")
    swap_params(decomposed_linear, config=quant_op_config, target_parameter_name="v")
    outputs = decomposed_linear(inputs)

    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(linear.weight.grad, grad_weights_ref)
    dw_sim = calc_cossim(linear.weight.grad, grad_weights_ref)

    print()
    print(
        tabulate([
            ["O", output_snr, output_sim],
            ["dX", dx_snr, dx_sim],
            ["dW", dw_snr, dw_sim],
        ],
                 headers=["Tensor", "SNR", "Cosine Sim"],
                 tablefmt="github"))
