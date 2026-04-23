# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public entry-points for NVFP4 Grouped GEMM.

Mirrors mxfp4/mxfp_grouped_gemm/functional.py so the two format families
can be swapped transparently at the dispatch layer.

Two public APIs:

  * ``nvfp4_grouped_gemm``: accepts ``expert_indices`` (per-token expert id),
    works with any token ordering.  Uses the Python-loop backend.

  * ``_quantize_then_nvfp4_scaled_grouped_mm``: accepts ``offs`` (cumulative offsets)
    from the dispatch layer where tokens are pre-sorted by expert.  Uses
    ``torch._grouped_mm`` when available, falls back to loop otherwise.
"""

from __future__ import annotations

import torch

from alto.kernels.hadamard_transform import HadamardFactory
from .autograd import NVFP4GroupedGEMM


def _build_hadamard_transform_if_needed(
    *,
    use_hadamard: bool,
    use_2dblock_x: bool,
    device: torch.device,
):
    """Create the grouped-path Hadamard transform if the recipe requests it."""
    if not use_hadamard:
        return None
    assert not use_2dblock_x, (
        "Hadamard transform can only be applied when use_2dblock_x=False."
    )
    with torch.no_grad():
        return HadamardFactory.create_transform(device=device)


def _nvfp4_grouped_gemm_impl(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    expert_indices: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = False,
    use_sr_grad: bool = True,
    use_per_tensor_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Normalized entrypoint shared by both public APIs.

    This is the NVFP4 analogue of the MXFP4 `mxfp4_grouped_gemm(...)` wrapper:
    it owns recipe normalization / transform construction and is the single
    place that calls into the grouped autograd Function.
    """
    if expert_indices is not None and expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)
    hadamard_transform = _build_hadamard_transform_if_needed(
        use_hadamard=use_hadamard,
        use_2dblock_x=use_2dblock_x,
        device=expert_weights.device,
    )
    return NVFP4GroupedGEMM.apply(
        inputs,
        expert_weights,
        expert_indices,
        offs,
        trans_weights,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_per_tensor_scale,
        hadamard_transform,
        use_dge,
    )


def nvfp4_grouped_gemm(
    inputs: torch.Tensor,          # [M_total, K]
    expert_weights: torch.Tensor,  # [E, N, K] if trans_weights else [E, K, N]
    expert_indices: torch.Tensor,  # [M_total]  int32
    *,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = False,
    use_sr_grad: bool = True,
    use_per_tensor_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """NVFP4 QDQ-emulated Grouped GEMM with full autograd support.

    All tokens routed to the same expert must occupy a contiguous block of
    ALIGN_SIZE_M rows in *inputs*.  Expert assignment order can be arbitrary.

    Args:
        inputs:            [M_total, K] activation tensor (BF16).
        expert_weights:    [E, N, K] or [E, K, N] expert weight tensors.
        expert_indices:    [M_total] int32, one entry per token giving its expert id.
        trans_weights:     If True, expert_weights is [E, N, K] and we compute x @ W.T.
        use_2dblock_x:     Use 16x16 2D scaling for activations.
        use_2dblock_w:     Use 16x16 2D scaling for weights.
        use_sr_grad:       Stochastic rounding for gradient quantisation.
        use_per_tensor_scale: Enable dynamic per-tensor scale on top of block scale.
        use_hadamard:      Apply Hadamard on the wgrad reduction axis (1D-x only).
        use_dge:           Apply DGE modulation to grouped weight gradients.

    Returns:
        [M_total, N] output tensor.
    """
    return _nvfp4_grouped_gemm_impl(
        inputs,
        expert_weights,
        expert_indices=expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_per_tensor_scale=use_per_tensor_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )


def _quantize_then_nvfp4_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_per_tensor_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Drop-in for the dispatch layer, mirroring mxfp4's
    ``_quantize_then_mxfp_scaled_grouped_mm``.

    Tokens in ``A`` are already sorted by expert.  ``offs`` is the cumulative
    per-expert token count tensor from the MoE routing layer.  This is the
    dispatch-entry helper that normalizes the grouped recipe and then forwards
    to the same grouped wrapper layer used by ``nvfp4_grouped_gemm(...)``.
    """
    return _nvfp4_grouped_gemm_impl(
        A,
        B,
        offs=offs,
        trans_weights=False,  # dispatch convention: B is [E, K, N]
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_per_tensor_scale=use_per_tensor_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
