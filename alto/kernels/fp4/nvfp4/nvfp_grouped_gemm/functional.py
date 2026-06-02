# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public entry-points for NVFP4 Grouped GEMM.

Mirrors ``mxfp4/mxfp_grouped_gemm/functional.py`` so the two format families
can be swapped transparently at the dispatch layer.

Two public APIs:

  * ``nvfp4_grouped_gemm``: accepts ``expert_indices`` (per-token expert id),
    works with any token ordering.

  * ``_quantize_then_nvfp4_scaled_grouped_mm``: accepts ``offs`` (cumulative
    offsets) from the dispatch layer where tokens are pre-sorted by expert.

Both paths funnel into :class:`~alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.autograd.NVFP4GroupedGEMM`
through the single private helper :func:`_nvfp4_grouped_gemm_impl`, which is
also the only place that materializes ``expert_weights`` into the canonical
``[E, N, K]`` internal layout.
"""

from __future__ import annotations

import torch

from alto.kernels.fp4.fp4_common import build_hadamard_transform_if_needed
from .autograd import NVFP4GroupedGEMM


def _nvfp4_grouped_gemm_impl(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,  # always [E, N, K] (canonical) when this is called
    *,
    expert_indices: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = False,
    use_sr_grad: bool = True,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Normalized entrypoint shared by both public APIs.

    This is the NVFP4 analogue of the MXFP4 ``mxfp4_grouped_gemm(...)`` wrapper:
    it owns recipe normalization / transform construction and is the single
    place that calls into the grouped autograd Function.  Its caller is
    responsible for having already transposed ``expert_weights`` into the
    canonical ``[E, N, K]`` layout.
    """
    if expert_indices is not None and expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)
    hadamard_transform = build_hadamard_transform_if_needed(
        use_hadamard=use_hadamard,
        use_2dblock_x=use_2dblock_x,
        device=expert_weights.device,
    )
    return NVFP4GroupedGEMM.apply(
        inputs,
        expert_weights,
        expert_indices,
        offs,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_outer_scale,
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
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """NVFP4 QDQ-emulated Grouped GEMM with full autograd support.

    All tokens routed to the same expert must occupy a contiguous block of
    ALIGN_SIZE_M rows in *inputs*.  Expert assignment order can be arbitrary.

    Args:
        inputs:            [M_total, K] activation tensor (BF16).
        expert_weights:    [E, N, K] if ``trans_weights=True`` (default), or
                           [E, K, N] if ``trans_weights=False``.  Internally
                           normalized to [E, N, K] once.
        expert_indices:    [M_total] int32, one entry per token giving its expert id.
        trans_weights:     Whether caller stores weights in ``[E, N, K]`` (True)
                           or ``[E, K, N]`` (False).
        use_2dblock_x:     Use 16x16 2D scaling for activations.
        use_2dblock_w:     Use 16x16 2D scaling for weights.
        use_sr_grad:       Stochastic rounding for gradient quantisation.
        use_outer_scale:   Enable the outer-level (NVFP4 ``s_global``) scale
                           on top of the per-block scale.
        use_hadamard:      Apply Hadamard on the wgrad reduction axis (1D-x only).
        use_dge:           Apply DGE modulation to grouped weight gradients.

    Returns:
        [M_total, N] output tensor.
    """
    if not trans_weights:
        # Downstream grouped kernels and _qdq both consume strided
        # expert_weights, so no .contiguous() needed.
        expert_weights = expert_weights.transpose(-2, -1)
    return _nvfp4_grouped_gemm_impl(
        inputs,
        expert_weights,
        expert_indices=expert_indices,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )


def _quantize_then_nvfp4_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,               # [E, K, N] per dispatch convention
    offs: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Drop-in for the dispatch layer, mirroring mxfp4's
    ``_quantize_then_mxfp4_scaled_grouped_mm``.

    Tokens in ``A`` are already sorted by expert.  ``offs`` is the cumulative
    per-expert token count tensor from the MoE routing layer.  ``B`` arrives
    in dispatch layout ``[E, K, N]`` and is transposed once to the canonical
    ``[E, N, K]`` layout shared by the autograd function and its primitives.
    """
    # Zero-copy stride view; see nvfp4_grouped_gemm for rationale.
    B_canonical = B.transpose(-2, -1)
    return _nvfp4_grouped_gemm_impl(
        A,
        B_canonical,
        offs=offs,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=use_outer_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
