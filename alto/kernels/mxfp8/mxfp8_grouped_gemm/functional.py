# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Top-level user-facing entry point for mxfp8 grouped GEMM."""

import torch

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
