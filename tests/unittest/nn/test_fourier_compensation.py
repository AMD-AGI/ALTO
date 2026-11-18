# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.config import TrainingOpConfig
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.lpt import _fourier_index, _fourier_weight, lpt_linear, lpt_module
from torchtitan.models.common.linear import GroupedLinear, Linear

LPTLinear = lpt_module(Linear)
LPTGroupedLinear = lpt_module(GroupedLinear)


def _config(*, fourier_n: int = 8) -> TrainingOpConfig:
    return TrainingOpConfig(
        precision="mxfp4",
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_hadamard=False,
        use_fwd_hadamard=False,
        use_dx_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
        use_uos=True,
        fourier_n=fourier_n,
    )


def _module() -> torch.nn.Module:
    module = LPTLinear(
        LPTLinear.Config(
            in_features=64,
            out_features=64,
            bias=True,
            training_op_config=_config(),
            inplace_wgrad_accum=False,
        )
    )
    return module.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_zero_coefficients_match_the_quantized_linear():
    torch.manual_seed(0)
    module = _module()
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    output = module(x)
    reference = lpt_linear(x, module.weight, module.bias, module.training_op_config)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_residual_is_the_inverse_fft():
    torch.manual_seed(1)
    module = _module()
    with torch.no_grad():
        module.fourier_coef.copy_(torch.randn_like(module.fourier_coef) * 0.1)
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    output = module(x)
    config = module.training_op_config
    main = lpt_linear(x, module.weight, module.bias, config)
    delta = _fourier_weight(
        module.fourier_coef, module.fourier_index, module.out_features, module.in_features
    )
    reference = main + torch.nn.functional.linear(x, delta.to(dtype=main.dtype))
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    assert not torch.equal(output, main)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_gradients_reach_the_coefficients_only():
    torch.manual_seed(2)
    module = _module()
    with torch.no_grad():
        module.fourier_coef.normal_(0, 0.05)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    module(x).float().square().sum().backward()
    assert module.fourier_coef.grad is not None
    assert module.fourier_coef.grad.abs().sum() > 0
    assert module.fourier_index.grad is None
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_locations_are_frozen_and_survive_init_states():
    first = _fourier_index(32, 48, 6, "cpu")
    second = _fourier_index(32, 48, 6, "cpu")
    assert torch.equal(first, second)
    assert first.unique().numel() == 6
    param_init = {
        "weight": torch.nn.init.xavier_uniform_,
        "bias": torch.nn.init.zeros_,
    }
    config = LPTLinear.Config(
        in_features=32,
        out_features=16,
        bias=True,
        training_op_config=_config(fourier_n=5),
        inplace_wgrad_accum=False,
        param_init=param_init,
    )
    with torch.device("meta"):
        module = config.build()
    module.to_empty(device="cpu")
    module.init_states()
    assert set(param_init) == {"weight", "bias"}
    assert module.fourier_coef.dtype == torch.float32
    assert torch.count_nonzero(module.fourier_coef) == 0
    expected = _fourier_index(16, 32, 5, module.fourier_index.device)
    assert torch.equal(module.fourier_index, expected)


def test_fourier_rejects_a_table_larger_than_the_spectrum():
    with pytest.raises(ValueError, match="exceeds"):
        LPTLinear(
            LPTLinear.Config(
                in_features=4,
                out_features=4,
                training_op_config=_config(fourier_n=64),
                inplace_wgrad_accum=False,
            )
        )
    with pytest.raises(NotImplementedError, match="Linear only"):
        LPTGroupedLinear(
            LPTGroupedLinear.Config(
                group_size=2,
                in_features=8,
                out_features=8,
                training_op_config=_config(),
            )
        )
    with pytest.raises(ValueError, match="non-negative"):
        LowPrecisionTrainingModifier(scheme="mxfp4", fourier_n=-1)
