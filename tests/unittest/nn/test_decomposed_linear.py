# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from alto.nn import DecomposedLinear
from alto.kernels.dispatch import TrainingOpConfig, swap_params


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

    combined_weight = decomposed_linear.weight + decomposed_linear.u.T @ torch.diag(decomposed_linear.sigma) @ decomposed_linear.v.T
    y_ref = F.linear(x, combined_weight, bias=decomposed_linear.bias)

    assert torch.allclose(y_ref, y_decomposed, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("in_features", [128, 256, 512])
@pytest.mark.parametrize("out_features", [128, 256, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [32, 64])
@pytest.mark.parametrize("precision", ["mxfp4"])
def test_decomposed_linear_quantization(in_features, out_features, bias, lora_rank, precision):
    STD = 0.1
    decomposed_linear = DecomposedLinear(in_features, out_features, bias, lora_rank).to("cuda")
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
