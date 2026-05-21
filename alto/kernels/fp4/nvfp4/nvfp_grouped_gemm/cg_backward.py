# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_backward import (
    cg_grouped_gemm_backward_inputs,
    cg_grouped_gemm_backward_weights,
)
from alto.kernels.fp4.nvfp4.nvfp_quantization import _qdq

from alto.kernels.fp4.fp4_common import (
    check_grouped_loop_contract,
    group_ids_from_expert_indices,
    use_cdna4_grouped_backend,
)
from .autotune import ALIGN_SIZE_M


def _nvfp4_grouped_dgrad(
    grad_output: torch.Tensor,     # [M_total, N]  (already QDQ-ed)
    expert_weights: torch.Tensor,  # [E, N, K]     (already QDQ-ed on dgrad axis)
    *,
    expert_indices: torch.Tensor,
    num_groups: Optional[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped activation-gradient GEMM.

    Computes, per expert e, ``dX[m, :] = grad_output[m, :] @ expert_weights[e, :, :]``
    over the tokens routed to that expert.

    Shapes
        grad_output     : ``[M_total, N]``
        expert_weights  : ``[E, N, K]`` (canonical; matches the input layout
                          that ``cg_grouped_gemm_backward_inputs`` expects)
    Returns
        grad_inputs     : ``[M_total, K]`` in ``output_dtype``

    The CDNA4 backend calls ``cg_grouped_gemm_backward_inputs`` directly, so
    there is no transpose on the hot path.
    """
    if use_cdna4_grouped_backend():
        # The CDNA4 dgrad kernel computes pointer offsets from caller-supplied
        # strides, so non-contiguous ``grad_output`` / ``expert_weights`` are
        # consumed natively (see ``_kernel_cg_backward_dx`` in
        # ``alto/kernels/blockwise_fp8/grouped_gemm/cg_backward.py``).
        return cg_grouped_gemm_backward_inputs(
            grad_output,
            expert_weights,
            expert_indices,
        ).to(output_dtype)

    assert num_groups is not None, "_nvfp4_grouped_dgrad: loop fallback requires num_groups"
    # Trailing buffer rows in grad_output are not routed; produce zeros.
    M_bufferlen = grad_output.shape[0]
    M_total = num_groups * ALIGN_SIZE_M
    check_grouped_loop_contract(M_total, where="_nvfp4_grouped_dgrad", align_size_m=ALIGN_SIZE_M)
    N = grad_output.shape[1]
    K = expert_weights.shape[2]
    group_ids = group_ids_from_expert_indices(expert_indices, num_groups, align_size_m=ALIGN_SIZE_M)
    g_groups = grad_output[:M_total].view(num_groups, ALIGN_SIZE_M, N)
    dx_groups = torch.zeros((num_groups, ALIGN_SIZE_M, K), dtype=output_dtype, device=grad_output.device)
    for eid in range(expert_weights.shape[0]):
        mask = group_ids == eid
        # expert_weights[eid] is already [N, K]; the natural (non-transposed)
        # matmul produces dX=[M,K].
        dx_groups[mask] = g_groups[mask] @ expert_weights[eid]
    dx_routed = dx_groups.reshape(M_total, K)
    if M_bufferlen == M_total:
        return dx_routed
    out = torch.zeros((M_bufferlen, K), dtype=output_dtype, device=grad_output.device)
    out[:M_total] = dx_routed
    return out


def _nvfp4_grouped_wgrad(
    grad_output: torch.Tensor,
    x_bwd: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    num_groups: Optional[int],
    *,
    use_sr_grad: bool,
    use_outer_scale: bool,
    use_2dblock_x: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize ``grad_output`` and compute the grouped weight gradient.

    Returns ``dW`` in the canonical ``[E, N, K]`` layout, matching both
    ``cg_grouped_gemm_backward_weights`` (CDNA4 native) and the autograd
    function's internal weight layout. No post-transpose needed.
    """
    if use_2dblock_x:
        g_m_dq = _qdq(
            grad_output, axis=-1, is_2d_block=True,
            use_outer_scale=use_outer_scale, use_sr=use_sr_grad,
        )
    else:
        g_m_dq = _qdq(
            grad_output, axis=0, is_2d_block=False,
            use_outer_scale=use_outer_scale, use_sr=use_sr_grad,
        )

    if use_cdna4_grouped_backend():
        # The CDNA4 wgrad kernel consumes non-contiguous tensors via
        # caller-supplied strides; no defensive ``.contiguous()`` needed.
        return cg_grouped_gemm_backward_weights(
            g_m_dq,
            x_bwd,
            expert_indices,
            num_experts=num_experts,
        ).to(output_dtype)

    assert num_groups is not None, "_nvfp4_grouped_wgrad: loop fallback requires num_groups"
    N = g_m_dq.shape[1]
    K = x_bwd.shape[1]
    # Trailing buffer rows are not routed and do not contribute to dW.
    M_total = num_groups * ALIGN_SIZE_M
    check_grouped_loop_contract(M_total, where="_nvfp4_grouped_wgrad", align_size_m=ALIGN_SIZE_M)
    group_ids = group_ids_from_expert_indices(expert_indices, num_groups, align_size_m=ALIGN_SIZE_M)
    go_groups = g_m_dq[:M_total].view(num_groups, ALIGN_SIZE_M, N)
    x_groups = x_bwd[:M_total].view(num_groups, ALIGN_SIZE_M, K)
    dw = torch.zeros((num_experts, N, K), dtype=output_dtype, device=g_m_dq.device)
    for eid in range(num_experts):
        mask = group_ids == eid
        go_e = go_groups[mask].reshape(-1, N)
        x_e = x_groups[mask].reshape(-1, K)
        dw[eid] = go_e.T @ x_e
    return dw
