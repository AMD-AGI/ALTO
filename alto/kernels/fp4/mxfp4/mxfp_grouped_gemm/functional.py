# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Public entry-points for MXFP4 Grouped GEMM.

Two public APIs:

  * ``mxfp4_grouped_gemm``: user-facing wrapper that takes ``expert_indices``
    directly (per-token expert id).
  * ``_quantize_then_mxfp4_scaled_grouped_mm``: drop-in for the dispatch
    layer, which materializes ``expert_indices`` from cumulative ``offs`` and
    assumes the ``[E, K, N]`` weight layout that MoE routing supplies.

Both paths funnel into :class:`~alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autograd.MXFP4GroupedGEMM`.
"""

import torch

from alto.kernels.fp4.fp4_common import build_hadamard_transform_if_needed
from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync

from .autograd import MXFP4GroupedGEMM


def mxfp4_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    *,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    use_sr_grad: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Contiguous grouped GEMM with full backward pass support.

    Args:
        inputs: Input tensor of shape [M_total, K].
        expert_weights: Expert weight tensor of shape [num_experts, N, K] if
            ``trans_weights`` else [num_experts, K, N].
        expert_indices: Indices tensor of shape [M_total] mapping each token to
            its expert.
        trans_weights: Whether expert weights are transposed.
        use_2dblock_x: Whether to use 2D block for activations.
        use_2dblock_w: Whether to use 2D block for weights.
        use_sr_grad: Whether to use stochastic rounding in gradients.
        use_hadamard: Whether to use Hadamard transform.
        use_dge: Whether to use DGE.

    Returns:
        Output tensor of shape [M_total, N].
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    hadamard_transform = build_hadamard_transform_if_needed(
        use_hadamard=use_hadamard,
        use_2dblock_x=use_2dblock_x,
        device=expert_weights.device,
    )

    return MXFP4GroupedGEMM.apply(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        hadamard_transform,
        use_dge,
    )


def _quantize_then_mxfp4_scaled_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_hadamard: bool,
    use_dge: bool,
) -> torch.Tensor:
    """Dispatch-layer entry point for MXFP4 grouped GEMM.

    Materializes per-token ``expert_indices`` from cumulative ``offs`` and
    forwards to :func:`mxfp4_grouped_gemm` with the dispatch-layer convention
    (``trans_weights=False``).  Parameter order matches the NVFP4 twin
    :func:`~alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.functional._quantize_then_nvfp4_scaled_grouped_mm`
    so both families can be swapped transparently.
    """
    m_indices = create_indices_from_offsets_nosync(offs)
    return mxfp4_grouped_gemm(
        A,
        B,
        m_indices,
        trans_weights=False,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
