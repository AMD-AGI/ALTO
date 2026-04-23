# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_backward import (
    cg_grouped_gemm_backward_weights,
)
from alto.kernels.fp4.nvfp4.nvfp_quantization import _qdq

from .autotune import ALIGN_SIZE_M
from .utils import _check_grouped_loop_contract, _group_ids_from_expert_indices, _use_cdna4_grouped_backend


def _nvfp4_grouped_wgrad(
    grad_output: torch.Tensor,
    x_bwd: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    num_groups: int,
    *,
    use_sr_grad: bool,
    use_per_tensor_scale: bool,
    use_2dblock_x: bool,
    trans_weights: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize grad_output and compute grouped dW."""
    if use_2dblock_x:
        g_m_dq = _qdq(
            grad_output, axis=-1, is_2d_block=True,
            use_per_tensor_scale=use_per_tensor_scale, use_sr=use_sr_grad,
        )
    else:
        g_m_dq = _qdq(
            grad_output, axis=0, is_2d_block=False,
            use_per_tensor_scale=use_per_tensor_scale, use_sr=use_sr_grad,
        )

    if _use_cdna4_grouped_backend():
        dw = cg_grouped_gemm_backward_weights(
            g_m_dq.contiguous(),
            x_bwd.contiguous(),
            expert_indices.contiguous(),
            num_experts=num_experts,
        ).to(output_dtype)
        if not trans_weights:
            dw = dw.transpose(-2, -1).contiguous()
        return dw

    N = g_m_dq.shape[1]
    K = x_bwd.shape[1]
    _check_grouped_loop_contract(grad_output.shape[0], where="_nvfp4_grouped_wgrad")
    group_ids = _group_ids_from_expert_indices(expert_indices, num_groups)
    go_groups = g_m_dq.view(num_groups, ALIGN_SIZE_M, N)
    x_groups = x_bwd.view(num_groups, ALIGN_SIZE_M, K)
    dw = torch.zeros((num_experts, N, K), dtype=output_dtype, device=g_m_dq.device)
    for eid in range(num_experts):
        mask = group_ids == eid
        go_e = go_groups[mask].reshape(-1, N)
        x_e = x_groups[mask].reshape(-1, K)
        dw[eid] = go_e.T @ x_e
    if not trans_weights:
        dw = dw.transpose(-2, -1).contiguous()
    return dw
