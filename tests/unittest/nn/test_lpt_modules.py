# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.config import TrainingOpConfig
from alto.models.gpt_oss.config_registry import gpt_oss_debugmodel_lpt
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.lpt import LPTModule, lpt_module, require_training_op_config
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    ScaledDotProductInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.gpt_oss import build_model_config
from torchtitan.models.gpt_oss.model import apply_attention_sink_rescale

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


def _gpt_oss_lpt_modifier() -> LowPrecisionTrainingModifier:
    return LowPrecisionTrainingModifier(
        scheme="mxfp4",
        targets=["Linear", "GptOssGroupedLinear", "VarlenInnerAttention"],
        ignore=["lm_head", "re:.*\\.router\\.gate"],
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=True,
        use_dge=False,
    )


def _assert_gpt_oss_mxfp4_attention(model_config) -> None:
    sliding = model_config.layers[0].attention.inner_attention
    full = model_config.layers[1].attention.inner_attention
    assert type(sliding).__qualname__ == "LPTSlidingWindowVarlenInnerAttention.Config"
    assert sliding.sliding_window_size == 128
    assert sliding.is_full_attention is False
    assert sliding.training_op_config.precision == "mxfp4"
    assert type(full).__qualname__ == "LPTVarlenInnerAttention.Config"
    assert full.is_full_attention is True
    assert full.training_op_config.precision == "mxfp4"


def test_gpt_oss_recipe_installs_lpt_modules():
    modifier = _gpt_oss_lpt_modifier()
    model_config = gpt_oss_debugmodel_lpt().model
    modifier.on_convert_config(model_config)
    layer = model_config.layers[0]
    _assert_gpt_oss_mxfp4_attention(model_config)

    megatron = build_model_config(
        "20b_megatron",
        seq_len=8192,
        parallelism=ParallelismConfig(expert_parallel_degree=1),
    )
    modifier.on_convert_config(megatron)
    _assert_gpt_oss_mxfp4_attention(megatron)

    assert type(layer.attention.qkv_linear.wqkv).__qualname__ == (
        "LPTColumnParallelLinear.Config"
    )
    assert type(layer.attention.wo).__qualname__ == "LPTRowParallelLinear.Config"
    assert type(layer.feed_forward.routed_experts.w13).__qualname__ == (
        "LPTGptOssGroupedLinear.Config"
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
    sliding = model.layers["0"].attention.inner_attention
    full = model.layers["1"].attention.inner_attention
    assert type(sliding).__name__ == "LPTSlidingWindowVarlenInnerAttention"
    assert type(full).__name__ == "LPTVarlenInnerAttention"
    assert list(sliding.iter_lpt_weights()) == []
    assert list(full.iter_lpt_weights()) == []


def test_lpt_varlen_keeps_documents_window_and_sink_lse():
    from alto.kernels.hadamard_transform.transform import HadamardFactory

    class _Identity:
        def __call__(self, value, inverse=False, left_mul=False):
            del inverse, left_mul
            return value

    raw = HadamardFactory.__dict__["create_transform"]
    HadamardFactory.create_transform = classmethod(lambda cls, device: _Identity())
    try:
        config = _mxfp4_config()
        module_cls = lpt_module(VarlenInnerAttention)
        module = module_cls(module_cls.Config(training_op_config=config)).cuda()
        sliding_cls = lpt_module(SlidingWindowVarlenInnerAttention)
        sliding_config = sliding_cls.Config(
            sliding_window_size=4, training_op_config=config
        )
        positions = torch.arange(32, device="cuda")
        metadata = sliding_config.build_attention_metadata(positions)
        assert metadata.window_size == (3, 0)

        torch.manual_seed(0)
        tokens, query_heads, kv_heads, dim = 64, 4, 2, 64
        q = torch.randn(
            tokens, query_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
        full_metadata = VarlenAttentionMetadata(cu, cu, 32, 32, window_size=(-1, 0))
        window_metadata = VarlenAttentionMetadata(cu, cu, 32, 32, window_size=(3, 0))
        sinks = torch.nn.Parameter(torch.zeros(query_heads, device="cuda"))
        seen = {}

        def transform(out, lse):
            seen["lse"] = lse
            return apply_attention_sink_rescale(out, lse, sinks)

        with torch.no_grad():
            raw = module(q, k, v, attention_metadata=full_metadata, scale=None)
            k_changed = k.detach().clone()
            k_changed[32:] = k_changed[32:] + 1
            isolated = module(
                q, k_changed, v, attention_metadata=full_metadata, scale=None
            )
            assert torch.equal(raw[:32], isolated[:32])
            assert not torch.equal(raw[32:], isolated[32:])
            windowed = module(q, k, v, attention_metadata=window_metadata, scale=None)
            assert not torch.equal(raw, windowed)

        sunk = module(
            q, k, v, attention_metadata=full_metadata, scale=None, out_transform=transform
        )
        assert seen["lse"].shape == (tokens, query_heads)
        assert torch.isfinite(seen["lse"]).all()
        sunk.sum().backward()
        assert torch.isfinite(q.grad).all()
        assert q.grad.shape == q.shape
        assert sinks.grad is not None and torch.isfinite(sinks.grad).all()
        assert sinks.grad.abs().sum() > 0
        nvfp4 = TrainingOpConfig(
            precision="nvfp4",
            use_2dblock_x=False,
            use_2dblock_w=True,
            use_hadamard=False,
            use_sr_grad=False,
            use_dge=False,
        )
        with pytest.raises(ValueError, match="Unsupported attention precision"):
            module_cls(module_cls.Config(training_op_config=nvfp4))
    finally:
        HadamardFactory.create_transform = raw
