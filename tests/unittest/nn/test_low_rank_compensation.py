# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.config import TrainingOpConfig
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.lpt import lpt_linear, lpt_module
from torchtitan.models.common.linear import GroupedLinear, Linear

LPTLinear = lpt_module(Linear)
LPTGroupedLinear = lpt_module(GroupedLinear)


def _config(*, lora_rank: int = 32) -> TrainingOpConfig:
    return TrainingOpConfig(
        precision="mxfp4",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=False,
        use_fwd_hadamard=False,
        use_dx_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
        use_uos=True,
        lora_rank=lora_rank,
    )


def _module(*, bias: bool = True) -> torch.nn.Module:
    module = LPTLinear(
        LPTLinear.Config(
            in_features=64,
            out_features=128,
            bias=bias,
            training_op_config=_config(),
            inplace_wgrad_accum=False,
        )
    )
    return module.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_zero_v_matches_the_quantized_linear():
    torch.manual_seed(0)
    module = _module()
    with torch.no_grad():
        module.v.zero_()
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    output = module(x)
    reference = lpt_linear(x, module.weight, module.bias, module.training_op_config)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_branch_adds_the_quantized_low_rank_term():
    torch.manual_seed(1)
    module = _module()
    with torch.no_grad():
        module.u.normal_(0, 0.02)
        module.v.normal_(0, 0.02)
        module.sigma.copy_(torch.linspace(0.25, 1.75, 32, device="cuda"))
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    output = module(x)
    config = module.training_op_config
    main = lpt_linear(x, module.weight, module.bias, config)
    hidden = lpt_linear(x, module.v, None, config)
    hidden = (hidden.float() * module.sigma.float()).to(dtype=hidden.dtype)
    reference = main + lpt_linear(hidden, module.u, None, config)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    assert not torch.equal(output, main)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP4 GEMM needs a GPU")
def test_gradients_reach_the_low_rank_factors():
    torch.manual_seed(2)
    module = _module(bias=False)
    with torch.no_grad():
        module.u.normal_(0, 0.05)
        module.v.normal_(0, 0.05)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    module(x).float().square().sum().backward()
    assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert module.u.grad is not None and module.u.grad.abs().sum() > 0
    assert module.v.grad is not None and module.v.grad.abs().sum() > 0
    assert module.sigma.grad is not None and module.sigma.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_init_states_keeps_the_parent_initializer_and_zeros_v():
    param_init = {
        "weight": torch.nn.init.xavier_uniform_,
        "bias": torch.nn.init.zeros_,
    }
    config = LPTLinear.Config(
        in_features=64,
        out_features=64,
        bias=True,
        training_op_config=_config(),
        inplace_wgrad_accum=False,
        param_init=param_init,
    )
    with torch.device("meta"):
        module = config.build()
    assert set(param_init) == {"weight", "bias"}
    module.to_empty(device="cpu")
    module.init_states()
    assert set(param_init) == {"weight", "bias"}
    assert torch.count_nonzero(module.v) == 0
    assert module.sigma.dtype == torch.float32
    assert torch.equal(module.sigma, torch.ones_like(module.sigma))
    assert module.u.abs().sum() > 0
    assert module.weight.abs().sum() > 0
    assert torch.count_nonzero(module.bias) == 0


def test_rank_zero_adds_no_factors_and_rejects_a_bad_shape():
    module = LPTLinear(
        LPTLinear.Config(
            in_features=4,
            out_features=8,
            training_op_config=_config(lora_rank=0),
            inplace_wgrad_accum=False,
        )
    )
    assert not hasattr(module, "u")
    with pytest.raises(ValueError, match="multiples"):
        LPTLinear(
            LPTLinear.Config(
                in_features=4,
                out_features=32,
                training_op_config=_config(),
                inplace_wgrad_accum=False,
            )
        )
    with pytest.raises(NotImplementedError, match="Linear only"):
        LPTGroupedLinear(
            LPTGroupedLinear.Config(
                group_size=2,
                in_features=64,
                out_features=64,
                training_op_config=_config(),
            )
        )


def test_recipe_rejects_a_rank_that_is_not_a_block_multiple():
    with pytest.raises(ValueError, match="multiple of the MX block size"):
        LowPrecisionTrainingModifier(scheme="mxfp4", lora_rank=16)
