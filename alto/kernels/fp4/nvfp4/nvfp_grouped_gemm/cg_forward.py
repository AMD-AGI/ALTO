# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_forward import cg_grouped_gemm_forward

from alto.kernels.fp4.fp4_common import (
    check_grouped_loop_contract,
    group_ids_from_expert_indices,
    use_cdna4_grouped_backend,
)
from .autotune import ALIGN_SIZE_M


def _nvfp4_grouped_fprop(
    inputs: torch.Tensor,          # [M_total, K]
    expert_weights: torch.Tensor,  # [E, N, K]  (canonical internal layout)
    *,
    expert_indices: torch.Tensor,
    num_groups: Optional[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped forward GEMM for 2D activations × 3D expert weights.

    Computes, per expert e, ``Y[m, :] = inputs[m, :] @ expert_weights[e, :, :].T``
    over the tokens routed to that expert.

    Shapes
        inputs          : ``[M_total, K]`` (already QDQ-ed on axis -1)
        expert_weights  : ``[E, N, K]`` (already QDQ-ed on axis -1; this is the
                          CDNA4-native layout shared with
                          ``cg_grouped_gemm_forward``)
        expert_indices  : ``[M_total]`` int32, contiguous
    Returns
        output          : ``[M_total, N]`` in ``output_dtype``

    On CDNA4 the Triton grouped backend is used directly (no per-step transpose);
    otherwise a Python-loop fallback computes one ALIGN_SIZE_M-sized expert
    group at a time without per-group host syncs.
    """
    if use_cdna4_grouped_backend():
        # The CDNA4 grouped GEMM kernels now compute pointer offsets from
        # caller-supplied strides (``stride_am``/``stride_be``/...), so they
        # consume non-contiguous ``inputs`` / ``expert_weights`` natively.
        # ``expert_indices`` is the only tensor still required to be
        # contiguous; that is enforced upstream by ``resolve_expert_indices``.
        return cg_grouped_gemm_forward(
            inputs,
            expert_weights,
            expert_indices,
        ).to(output_dtype)

    assert num_groups is not None, "_nvfp4_grouped_fprop: loop fallback requires num_groups"
    M_total = inputs.shape[0]
    check_grouped_loop_contract(M_total, where="_nvfp4_grouped_fprop", align_size_m=ALIGN_SIZE_M)
    K = inputs.shape[1]
    N = expert_weights.shape[1]
    group_ids = group_ids_from_expert_indices(expert_indices, num_groups, align_size_m=ALIGN_SIZE_M)
    x_groups = inputs.view(num_groups, ALIGN_SIZE_M, K)
    y_groups = torch.zeros((num_groups, ALIGN_SIZE_M, N), dtype=output_dtype, device=inputs.device)
    for eid in range(expert_weights.shape[0]):
        mask = group_ids == eid
        # expert_weights[eid] is [N, K]; .T is a zero-copy strided view that
        # PyTorch's bf16 matmul handles natively, so no contiguous() needed.
        y_groups[mask] = x_groups[mask] @ expert_weights[eid].T
    return y_groups.reshape(M_total, N)
