# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Top-level user-facing entry point for mxfp8 grouped GEMM."""

import torch

from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_backward import MXFP8GroupedGEMM


def mxfp8_grouped_gemm(
    inputs: torch.Tensor,            # [M_total, K], bf16/fp32
    expert_weights: torch.Tensor,    # [num_experts, N, K] if trans_weights else [num_experts, K, N]
    expert_indices: torch.Tensor,    # [M_total], int (auto-cast to int32)
    *,
    trans_weights: bool = True,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    use_sr_grad: bool = False,
    fwd_format: str = "e4m3",
    bwd_grad_format: str = "e4m3",
) -> torch.Tensor:
    """MXFP8 contiguous grouped GEMM with full backward.

    V1 uses e4m3 for all operands across fwd/dgrad/wgrad. The mixed format
    (bwd grad_output in e5m2) is reserved for v2 — flip bwd_grad_format to
    "e5m2" once the e5m2 dot_scaled branch is enabled. See
    MXFP8_GROUPED_GEMM_PLAN.md §0 for the rationale.
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return MXFP8GroupedGEMM.apply(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        fwd_format,
        bwd_grad_format,
    )


def _quantize_then_mxfp8_scaled_grouped_mm(
    A: torch.Tensor,       # [M_bufferlen, K], bf16/fp32; may be padded (M_bufferlen >= M_total)
    B: torch.Tensor,       # [num_experts, K, N], dispatch convention (trans_weights=False)
    offs: torch.Tensor,    # [num_experts], int32 cumulative offsets; offs[-1] == M_total (routed)
    *,
    use_2dblock_x: bool = False,
    use_2dblock_w: bool = True,
    use_sr_grad: bool = False,
    fwd_format: str = "e4m3",
    bwd_grad_format: str = "e4m3",
) -> torch.Tensor:
    """Dispatch-layer entry for mxfp8 grouped GEMM (GPT-OSS-style MoE call).

    Mirrors the mxfp4 ``_quantize_then_mxfp_scaled_grouped_mm`` contract: ``offs``
    is the 1-D cumulative offset tensor from the MoE routing layer (e.g.
    [128, 256, 384, 512]), so ``offs[-1] == M_total`` (routed token count). ``A``
    may have ``shape[0] > offs[-1]`` (padded activation buffer); trailing rows
    produce zero output and zero gradient.

    ``B`` is kept in dispatch convention ``[E, K, N]`` and passed straight through
    with ``trans_weights=False`` (no transpose copy).
    """
    m_indices = create_indices_from_offsets_nosync(offs)
    return mxfp8_grouped_gemm(
        A,
        B,
        m_indices,
        trans_weights=False,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        fwd_format=fwd_format,
        bwd_grad_format=bwd_grad_format,
    )
