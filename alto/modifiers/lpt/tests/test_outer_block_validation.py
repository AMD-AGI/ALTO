# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Validation tests for NVFP4 outer-block LPT recipe fields."""

import pytest

pytest.importorskip("torchtitan")

from alto.modifiers.lpt.base import LowPrecisionTrainingModifier


def test_outer_block_side_fields_require_outer_block_mode():
    with pytest.raises(ValueError, match='two_level_scaling="outer_block"'):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="tensorwise",
            use_outer_2dblock_x=True,
        )


def test_non_default_outer_block_size_requires_outer_block_mode():
    with pytest.raises(ValueError, match='two_level_scaling="outer_block"'):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="none",
            outer_block_size=256,
        )


def test_outer_block_mode_requires_nvfp4_scheme():
    with pytest.raises(ValueError, match="only defined for scheme='nvfp4'"):
        LowPrecisionTrainingModifier(
            scheme="mxfp4",
            two_level_scaling="outer_block",
        )


def test_outer_block_mode_rejects_mixed_scheme_dict():
    with pytest.raises(ValueError, match="only defined for scheme='nvfp4'"):
        LowPrecisionTrainingModifier(
            scheme={"nvfp4": ["Linear"], "mxfp4": ["OtherLinear"]},
            two_level_scaling="outer_block",
        )


def test_outer_block_mode_accepts_nvfp4_recipe():
    modifier = LowPrecisionTrainingModifier(
        scheme="nvfp4",
        two_level_scaling="outer_block",
        use_outer_2dblock_x=False,
        use_outer_2dblock_w=True,
        outer_block_size=128,
    )
    assert modifier.two_level_scaling == "outer_block"


# ---------------------------------------------------------------------------
# Outer-block scaling does not support MoE / GroupedExperts modules (the
# dispatch wrapper raises ``NotImplementedError`` on grouped_mm).  Reject
# such recipes at construction time so the failure points at the yaml.
# ---------------------------------------------------------------------------


def test_outer_block_rejects_targets_with_grouped_experts():
    with pytest.raises(ValueError, match="grouped-experts"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            targets=["Linear", "GptOssGroupedExperts"],
            two_level_scaling="outer_block",
        )


def test_outer_block_rejects_dict_scheme_with_moe_targets():
    with pytest.raises(ValueError, match="grouped-experts"):
        LowPrecisionTrainingModifier(
            scheme={"nvfp4": ["Linear", "DeepseekV3GroupedExperts"]},
            two_level_scaling="outer_block",
        )


def test_outer_block_rejects_regex_matching_grouped_experts():
    with pytest.raises(ValueError, match="grouped-experts"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            targets=["Linear", "re:.*GroupedExperts$"],
            two_level_scaling="outer_block",
        )


def test_outer_block_accepts_targets_without_moe():
    """Sanity: dense-only targets are still accepted under outer-block."""
    modifier = LowPrecisionTrainingModifier(
        scheme="nvfp4",
        targets=["Linear"],
        two_level_scaling="outer_block",
        use_outer_2dblock_w=True,
    )
    assert modifier.two_level_scaling == "outer_block"


def test_pts_recipe_is_unchanged_by_p0_4_validator():
    """tensorwise / none recipes must keep accepting MoE targets unchanged."""
    modifier = LowPrecisionTrainingModifier(
        scheme="nvfp4",
        targets=["Linear", "GptOssGroupedExperts"],
        two_level_scaling="tensorwise",
    )
    assert modifier.two_level_scaling == "tensorwise"
