# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    ScaledDotProductInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import decoder_vocab_size, get_attention_config
from torchtitan.models.common.rope import CosSinRoPE

from alto.models.build_record import install_build_recorder, rebuild_recorded_model
from alto.patches.patcher import ModelPatcher
from alto.train import apply_loaded_overrides


def _build(seq_len: int, parallelism: ParallelismConfig, attn_backend: str = "varlen"):
    install_build_recorder("llama3")
    from torchtitan.models.llama3 import build_model_config

    return build_model_config(
        "debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
        attn_backend=attn_backend,
    )


def _trainer(model, parallelism, seq_len: int):
    loss_fn = SimpleNamespace(global_vocab_size=-1)
    return SimpleNamespace(
        model=model,
        parallelism=parallelism,
        training=SimpleNamespace(
            max_context_length=seq_len,
            disable_accelerator_graphs=False,
        ),
        loss=SimpleNamespace(loss_fn=loss_fn),
        model_converters=SimpleNamespace(converters=None),
        _loss_fn=loss_fn,
    )


def test_rebuild_replays_recipe_edits_and_refreshes_sequence_length():
    built_with = ParallelismConfig()
    model = _build(2048, built_with, attn_backend="varlen")
    for _fqn, attention, _parent, _attr in model.traverse(
        type(model.layers[0].attention)
    ):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(model)
    model.layers[0].attention.rope.theta = 12345.0

    updated = ParallelismConfig(enable_sequence_parallel=False)
    config = _trainer(model, updated, seq_len=4096)
    rebuild_recorded_model(config)

    fresh = _build(4096, updated, attn_backend="varlen")
    ModelPatcher.patch_model_config(fresh)

    assert config.model.max_context_length == 4096
    assert config.model.layers[0].attention.rope.max_context_length == 4096
    assert type(config.model.layers[0].attention.rope) is CosSinRoPE.Config
    assert config.model.layers[0].attention.rope.theta == 12345.0
    assert config.model.layers[1].attention.rope.theta == fresh.layers[1].attention.rope.theta
    assert (
        config.model.layers[0].attention_norm.sharding_config
        == fresh.layers[0].attention_norm.sharding_config
    )
    assert (
        model.layers[0].attention_norm.sharding_config
        != fresh.layers[0].attention_norm.sharding_config
    )
    for layer in config.model.layers:
        assert type(layer.attention.inner_attention) is ScaledDotProductInnerAttention.Config
        assert layer.attention.inner_attention.sharding_config is None
    assert type(fresh.layers[0].attention.inner_attention) is VarlenInnerAttention.Config
    assert config._loss_fn.global_vocab_size == decoder_vocab_size(config.model)


def test_model_field_override_is_applied_to_rebuilt_config():
    parallelism = ParallelismConfig()
    model = _build(2048, parallelism, attn_backend="sdpa")
    for layer in model.layers:
        layer.attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(model)
    config = _trainer(model, parallelism, seq_len=2048)

    apply_loaded_overrides(
        config,
        [
            ("training.max_context_length", "8192"),
            ("model.max_context_length", "4096"),
            ("model_converters.converters", "[]"),
        ],
    )

    assert config.training.max_context_length == 8192
    assert config.model.max_context_length == 4096
    assert config.model.layers[0].attention.rope.max_context_length == 8192
    assert config.model_converters.converters == []
    for layer in config.model.layers:
        assert type(layer.attention.inner_attention) is ScaledDotProductInnerAttention.Config
        assert layer.attention.inner_attention.sharding_config is None
