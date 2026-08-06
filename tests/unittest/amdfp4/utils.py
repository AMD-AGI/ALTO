# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Test helpers for the AMD-FP4 unit suite.

Reuses the NVFP4 PyTorch oracle (``tests/unittest/nvfp4/utils.py``)
because the recipe-level quant / dequant body is identical between
NVFP4 and AMD-FP4 modulo the inner-grid choice.  This module just
exposes thin wrappers that pin ``scale_format='ue5m3'``, so the AMD-FP4
test files never spell ``ue5m3`` themselves.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from alto.kernels.fp4.nvfp4.nvfp_quantization import BLOCK_SIZE_DEFAULT
from alto.kernels.fp4.testing_utils import calc_cossim, calc_snr  # noqa: F401

# Sibling import: the AMD-FP4 PyTorch oracle is the NVFP4 oracle pinned
# to ``scale_format='ue5m3'``; ``conftest.py`` arranges for the parent
# unittest dir on ``sys.path`` so this works.
from nvfp4.utils import (  # noqa: E402  (see amdfp4/conftest.py sys.path)
    convert_from_nvfp4_pytorch as _convert_from_nvfp4_pytorch,
    convert_to_nvfp4_pytorch as _convert_to_nvfp4_pytorch,
    prepare_data,
)

__all__ = (
    "BLOCK_SIZE_DEFAULT",
    "calc_cossim",
    "calc_snr",
    "convert_from_amdfp4_pytorch",
    "convert_to_amdfp4_pytorch",
    "prepare_data",
)


def convert_to_amdfp4_pytorch(
    data_hp: Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[Tensor] = None,
):
    """PyTorch oracle for the AMD-FP4 quant op (UE5M3 inner scale)."""
    return _convert_to_nvfp4_pytorch(
        data_hp,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        scale_format="ue5m3",
    )


def convert_from_amdfp4_pytorch(
    data_lp: Tensor,
    scales: Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[Tensor] = None,
):
    """PyTorch oracle for the AMD-FP4 dequant op (UE5M3 inner scale)."""
    return _convert_from_nvfp4_pytorch(
        data_lp,
        scales,
        output_dtype=output_dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        scale_format="ue5m3",
    )
