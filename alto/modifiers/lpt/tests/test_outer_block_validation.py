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


# ---------------------------------------------------------------------------
# Paper-recipe knobs: align_x_forward_wgrad (P0-2) and use_dx_rht (P0-4).
# ---------------------------------------------------------------------------


def test_align_x_forward_wgrad_requires_outer_block_mode():
    with pytest.raises(ValueError, match="align_x_forward_wgrad=True requires"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="tensorwise",
            align_x_forward_wgrad=True,
        )


def test_align_x_forward_wgrad_requires_1d_inner_x():
    with pytest.raises(ValueError, match="use_2dblock_x=False"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="outer_block",
            use_2dblock_x=True,
            align_x_forward_wgrad=True,
        )


def test_align_x_forward_wgrad_incompatible_with_hadamard():
    with pytest.raises(ValueError, match="incompatible with use_hadamard"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="outer_block",
            use_hadamard=True,
            align_x_forward_wgrad=True,
        )


def test_align_x_forward_wgrad_incompatible_with_dge():
    with pytest.raises(ValueError, match="not supported with use_dge"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="outer_block",
            use_dge=True,
            align_x_forward_wgrad=True,
        )


def test_align_x_forward_wgrad_accepts_paper_recipe():
    modifier = LowPrecisionTrainingModifier(
        scheme="nvfp4",
        two_level_scaling="outer_block",
        use_2dblock_x=False,
        use_2dblock_w=False,
        align_x_forward_wgrad=True,
    )
    assert modifier.align_x_forward_wgrad is True
    assert modifier.use_dx_rht is False


def test_dx_rht_requires_outer_block_mode():
    with pytest.raises(ValueError, match="use_dx_rht=True requires"):
        LowPrecisionTrainingModifier(
            scheme="nvfp4",
            two_level_scaling="tensorwise",
            use_dx_rht=True,
        )


def test_dx_rht_combined_with_align_is_accepted():
    """Paper-recipe (C+rht) arm: align + dx_rht should compose cleanly."""
    modifier = LowPrecisionTrainingModifier(
        scheme="nvfp4",
        two_level_scaling="outer_block",
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_outer_2dblock_w=True,
        align_x_forward_wgrad=True,
        use_dx_rht=True,
    )
    assert modifier.use_dx_rht is True
    assert modifier.align_x_forward_wgrad is True
