# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear

from alto.models.gpt_oss.config_registry import (
    LPT_RECIPE,
    MXFP4_GEMM_MXFP8_ATTN_RECIPE,
    MXFP4_GEMM_RECIPE,
    gpt_oss_20b_lpt,
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

    dense = [fqn for fqn in gemms if fqn.endswith(("wqkv", "wo"))]
    grouped = [fqn for fqn in gemms if fqn.endswith(("w13", "w2"))]
    assert len(dense) == len(grouped) == 2 * len(mxfp8_attn.model.layers)
    assert len(gemms) == len(dense) + len(grouped)
    for layer in mxfp8_attn.model.layers:
        owner = layer.feed_forward.routed_experts.w13._owner
        assert issubclass(owner, GroupedLinear) and not issubclass(owner, GptOssGroupedLinear)

    for layer_id, layer in enumerate(bf16_attn.model.layers):
        expected = (
            DetachedLseSlidingWindowFlexInnerAttention
            if layer_id % 2 == 0
            else DetachedLseFlexInnerAttention
        )
        assert type(layer.attention.inner_attention) is expected.Config, layer_id


def test_mxfp4_gemm_recipes_are_lpt_recipe_plus_uos():
    lpt = _modifier_args(LPT_RECIPE)
    gemm_only = _modifier_args(MXFP4_GEMM_RECIPE)
    combined = _modifier_args(MXFP4_GEMM_MXFP8_ATTN_RECIPE)
    assert gemm_only["targets"] == combined["scheme"]["mxfp4"] == lpt["targets"]
    for recipe in (gemm_only, combined):
        assert recipe["use_uos"] is True
        for key, value in lpt.items():
            if key not in ("scheme", "targets"):
                assert recipe[key] == value, key


def test_uos_may_share_a_modifier_with_mxfp8_but_needs_mxfp4():
    LowPrecisionTrainingModifier(
        scheme={"mxfp4": ["Linear"], "mxfp8_e4m3": ["FlexInnerAttention"]}, use_uos=True
    )
    for scheme in ("mxfp8_e4m3", {"mxfp4": ["Linear"], "nvfp4": ["GroupedLinear"]}):
        with pytest.raises(ValueError, match="UOS"):
            LowPrecisionTrainingModifier(scheme=scheme, use_uos=True)


def _strip_filter_lambdas(loader):
    """``megatron_dataset_config`` builds a new filter lambda per call; drop it before comparing."""
    packing = loader.dataset
    return replace(loader, dataset=replace(packing, dataset=replace(packing.dataset, post_filters=())))


def test_mxfp4_gemm_pair_matches_gpt_oss_20b_lpt_settings():
    reference = gpt_oss_20b_lpt()
    for config in (
        gpt_oss_20b_pretrain_flex_mxfp4_gemm_detached_lse(),
        gpt_oss_20b_pretrain_flex_mxfp4_gemm_mxfp8_attn(),
    ):
        for name in ("training", "optim", "parallelism", "loss", "debug", "metrics"):
            assert getattr(config, name) == getattr(reference, name), name
        for ours, ref in (
            (config.dataloader, reference.dataloader),
            (config.validator.dataloader, reference.validator.dataloader),
        ):
            assert ours.dataset.dataset.processor.seq_align == 32
            assert _strip_filter_lambdas(ours) == _strip_filter_lambdas(ref)
        assert config.validator.freq == reference.validator.freq
        # MLPerf's 1024 eval samples.
        assert config.validator.steps == 2 * reference.validator.steps == 64
