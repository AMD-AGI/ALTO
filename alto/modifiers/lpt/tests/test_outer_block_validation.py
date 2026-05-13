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
