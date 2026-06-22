# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 (UE5M3 inner scale) public quantize / dequantize ops.

AMD-FP4 = NVFP4 spec + UE5M3 inner-block scale + FP32 per-tensor outer
scale (GFXIPARCH-2067 §19.10 / OCP E5M3 aligned).  The block-quant
implementation is identical to NVFP4 modulo the inner-grid choice, so
these ops are thin wrappers over the NVFP4 quant kernels with
``scale_format`` hard-pinned to ``"ue5m3"``.  This module only adds the
AMD-FP4 ATen op surface:

* ``alto::convert_to_amdfp4``    -- pinned ``scale_format='ue5m3'``
* ``alto::convert_from_amdfp4``  -- pinned ``scale_format='ue5m3'``

These ops are exposed alongside (not in place of) the NVFP4 ops.  They
exist so dispatch / tracing / benchmarking tools can identify AMD-FP4
calls by op id without inspecting ``scale_format`` kwargs.
"""

from typing import Optional, Tuple

import torch
from torch.library import triton_op

from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    _fake_convert_from_nvfp4,
    _fake_convert_to_nvfp4,
    convert_from_nvfp4,
    convert_to_nvfp4,
)


# ``outer_scale`` is mutated in place (``copy_``) when ``update_outer_scale``
# is True; declare it so Dynamo/functionalization invalidate stale caches.
@triton_op("alto::convert_to_amdfp4", mutates_args={"outer_scale"})
def convert_to_amdfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    update_outer_scale: bool = True,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a high-precision tensor to AMD-FP4 (E2M1 + UE5M3 inner scale).

    The signature mirrors :func:`alto.kernels.fp4.nvfp4.convert_to_nvfp4`
    minus the ``scale_format`` parameter, which is hard-pinned to ``"ue5m3"``
    for this op.
    """
    return convert_to_nvfp4(
        data_hp,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        update_outer_scale=update_outer_scale,
        scale_format="ue5m3",
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        use_asm=use_asm,
    )


@triton_op("alto::convert_from_amdfp4", mutates_args={})
def convert_from_amdfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    """Dequantize AMD-FP4 (E2M1 + UE5M3 inner scale) back to high precision.

    Like :func:`convert_to_amdfp4`, this op pins ``scale_format='ue5m3'``;
    the dequantization path itself only multiplies by the stored FP32
    scale, so the format pin is a contract / op-id signal rather than a
    numerical switch.
    """
    return convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=output_dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        scale_format="ue5m3",
        use_asm=use_asm,
    )


@convert_to_amdfp4.register_fake
def _fake_convert_to_amdfp4(
    data_hp: torch.Tensor,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    update_outer_scale: bool = True,
    use_sr: bool = False,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    use_asm: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _fake_convert_to_nvfp4(
        data_hp,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        update_outer_scale=update_outer_scale,
        scale_format="ue5m3",
        use_sr=use_sr,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        use_asm=use_asm,
    )


@convert_from_amdfp4.register_fake
def _fake_convert_from_amdfp4(
    data_lp: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    block_size: int = BLOCK_SIZE_DEFAULT,
    axis: int = -1,
    is_2d_block: bool = False,
    outer_scale: Optional[torch.Tensor] = None,
    use_asm: Optional[bool] = None,
) -> torch.Tensor:
    return _fake_convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=output_dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        outer_scale=outer_scale,
        scale_format="ue5m3",
        use_asm=use_asm,
    )
