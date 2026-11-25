# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import yaml

from alto.models.gpt_oss.config_registry import (
    LPT_RECIPE,
    MXFP4_GEMM_MXFP8_ATTN_RECIPE,
    gpt_oss_20b_pretrain,
    gpt_oss_20b_pretrain_flex_mxfp4_gemm_detached_lse,
    gpt_oss_20b_pretrain_flex_mxfp4_gemm_mxfp8_attn,
)
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.attention import (
    DetachedLseFlexInnerAttention,
    DetachedLseSlidingWindowFlexInnerAttention,
)
from alto.nn.lpt import LPTModule

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _modifier_args(recipe: str) -> dict:
    recipe_yaml = yaml.safe_load((_REPO_ROOT / recipe).read_text())
    return recipe_yaml["training_stage"]["lpt_modifiers"]["LowPrecisionTrainingModifier"]


def _lpt_precisions(config) -> dict[str, str]:
    (converter,) = config.model_converters.converters
    LowPrecisionTrainingModifier(**_modifier_args(converter.recipe)).on_convert_config(config.model)
    return {
        fqn: inner.training_op_config.precision
        for fqn, inner, _parent, _attr in config.model.traverse(LPTModule.Config)
    }


def test_mxfp4_gemm_pair_differs_only_in_attention_precision():
    bf16_attn = gpt_oss_20b_pretrain_flex_mxfp4_gemm_detached_lse()
    mxfp8_attn = gpt_oss_20b_pretrain_flex_mxfp4_gemm_mxfp8_attn()
    bf16_precisions = _lpt_precisions(bf16_attn)
    mxfp8_precisions = _lpt_precisions(mxfp8_attn)

    attention = {
        fqn: precision
        for fqn, precision in mxfp8_precisions.items()
        if fqn.endswith("attention.inner_attention")
    }
    assert len(attention) == len(mxfp8_attn.model.layers)
    assert set(attention.values()) == {"mxfp8_e4m3"}

    gemms = {fqn: p for fqn, p in mxfp8_precisions.items() if fqn not in attention}
    assert gemms and set(gemms.values()) == {"mxfp4"}
    assert not any(fqn.endswith(("router.gate", "lm_head")) for fqn in gemms)
    assert bf16_precisions == gemms

    for layer_id, layer in enumerate(bf16_attn.model.layers):
        expected = (
            DetachedLseSlidingWindowFlexInnerAttention
            if layer_id % 2 == 0
            else DetachedLseFlexInnerAttention
        )
        assert type(layer.attention.inner_attention) is expected.Config, layer_id


def test_combined_recipe_keeps_lpt_recipe_mxfp4_settings():
    mxfp4 = _modifier_args(LPT_RECIPE)
    combined = _modifier_args(MXFP4_GEMM_MXFP8_ATTN_RECIPE)
    assert combined["scheme"]["mxfp4"] == mxfp4["targets"]
    for key, value in mxfp4.items():
        if key not in ("scheme", "targets"):
            assert combined[key] == value, key


def test_mxfp4_gemm_pair_keeps_gpt_oss_20b_pretrain_settings():
    reference = gpt_oss_20b_pretrain()
    for config in (
        gpt_oss_20b_pretrain_flex_mxfp4_gemm_detached_lse(),
        gpt_oss_20b_pretrain_flex_mxfp4_gemm_mxfp8_attn(),
    ):
        for name in ("training", "optim", "parallelism", "loss", "checkpointer", "debug", "metrics"):
            assert getattr(config, name) == getattr(reference, name), name
        assert config.dataloader.dataset.dataset.processor.seq_align is None
        assert config.validator.dataloader.dataset.dataset.processor.seq_align is None
