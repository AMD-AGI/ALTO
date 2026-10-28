# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.config import TrainingOpConfig
from alto.models.gpt_oss.config_registry import gpt_oss_debugmodel_lpt
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.lpt import LPTModule, lpt_module, require_training_op_config
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.linear import Linear

LPTLinear = lpt_module(Linear)


def _mxfp4_config() -> TrainingOpConfig:
    return TrainingOpConfig(
        precision="mxfp4",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=True,
        use_dge=False,
    )


def test_lpt_linear_rejects_unset_training_op_config():
    config = LPTLinear.Config(in_features=4, out_features=4)
    assert isinstance(config, LPTModule.Config)
    assert config.training_op_config is None
    with pytest.raises(RuntimeError, match="not fully initialized"):
        LPTLinear(config)
    with pytest.raises(RuntimeError, match="not fully initialized"):
        require_training_op_config(LPTLinear.Config(in_features=4, out_features=4))


def test_lpt_linear_exposes_weight_for_deoscillation():
    config = _mxfp4_config()
    module = LPTLinear(
        LPTLinear.Config(
            in_features=4,
            out_features=8,
            training_op_config=config,
            inplace_wgrad_accum=False,
        )
    )
    parameter, op_config, axis = next(module.iter_lpt_weights())
    assert parameter is module.weight
    assert op_config is config
    assert axis == -1
    assert type(parameter.data) is torch.Tensor
    assert "training_op_config=TrainingOpConfig(precision='mxfp4'" in repr(module)


def test_lpt_sdpa_has_no_weights():
    config = _mxfp4_config()
    module_cls = lpt_module(ScaledDotProductInnerAttention)
    module = module_cls(module_cls.Config(training_op_config=config))
    assert isinstance(module, LPTModule)
    assert module.training_op_config is config
    assert list(module.iter_lpt_weights()) == []
    assert "training_op_config=TrainingOpConfig(precision='mxfp4'" in repr(module)
    nvfp4 = TrainingOpConfig(
        precision="nvfp4",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
    )
    with pytest.raises(ValueError, match="Unsupported SDPA precision"):
        module_cls(module_cls.Config(training_op_config=nvfp4))


def test_gpt_oss_recipe_installs_lpt_modules():
    modifier = LowPrecisionTrainingModifier(
        scheme="mxfp4",
        targets=["Linear", "GptOssGroupedLinear"],
        ignore=["lm_head", "re:.*\\.router\\.gate"],
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=True,
        use_dge=False,
    )
    model_config = gpt_oss_debugmodel_lpt().model
    modifier.on_convert_config(model_config)
    layer = model_config.layers[0]

    assert type(layer.attention.qkv_linear.wqkv).__qualname__ == (
        "LPTColumnParallelLinear.Config"
    )
    assert type(layer.attention.wo).__qualname__ == "LPTRowParallelLinear.Config"
    assert type(layer.feed_forward.routed_experts.w13).__qualname__ == (
        "LPTGptOssGroupedLinear.Config"
    )
    assert type(layer.attention.inner_attention).__qualname__ == (
        "SlidingWindowVarlenInnerAttention.Config"
    )
    assert type(model_config.layers[1].attention.inner_attention).__qualname__ == (
        "VarlenInnerAttention.Config"
    )
    assert type(layer.feed_forward.router.gate).__qualname__ == "HiMidLoLinear.Config"
    assert type(model_config.lm_head).__qualname__ == "Linear.Config"
    assert layer.attention.qkv_linear.wqkv.inplace_wgrad_accum is False
    assert (
        layer.feed_forward.routed_experts.w13.training_op_config.precision == "mxfp4"
    )

    with torch.device("meta"):
        model = model_config.build()
    expert = model.layers["0"].feed_forward.routed_experts.w13
    parameter, op_config, axis = next(expert.iter_lpt_weights())
    assert parameter is expert.weight
    assert op_config.precision == "mxfp4"
    assert axis == -1
    assert "training_op_config=TrainingOpConfig(precision='mxfp4'" in repr(expert)
    linear = model.layers["0"].attention.wo
    assert "in_features=" in repr(linear)
    assert "training_op_config=TrainingOpConfig(precision='mxfp4'" in repr(linear)
