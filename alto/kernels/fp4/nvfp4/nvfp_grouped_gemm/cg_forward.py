# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.blockwise_fp8.grouped_gemm.cg_forward import cg_grouped_gemm_forward

from .autotune import ALIGN_SIZE_M
from .utils import _check_grouped_loop_contract, _group_ids_from_expert_indices, _use_cdna4_grouped_backend


def _nvfp4_grouped_mm_2d_3d(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    expert_indices: torch.Tensor,
    num_groups: Optional[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped GEMM for 2D activations × 3D expert weights.

    `A` is `[M_total, X]`, `B` is `[E, X, Y]`. On CDNA4 the Triton grouped
    backend is used; otherwise a loop fallback computes one ALIGN_SIZE_M-sized
    expert group at a time without per-group host syncs.
    """
    if _use_cdna4_grouped_backend():
        return cg_grouped_gemm_forward(
            A.contiguous(),
            B.transpose(-2, -1).contiguous(),
            expert_indices.contiguous(),
        ).to(output_dtype)

    assert num_groups is not None, "_nvfp4_grouped_mm_2d_3d: loop fallback requires num_groups"
    M_total = A.shape[0]
    _check_grouped_loop_contract(M_total, where="_nvfp4_grouped_mm_2d_3d")
    X = A.shape[1]
    Y = B.shape[2]
    group_ids = _group_ids_from_expert_indices(expert_indices, num_groups)
    A_groups = A.view(num_groups, ALIGN_SIZE_M, X)
    y_groups = torch.zeros((num_groups, ALIGN_SIZE_M, Y), dtype=output_dtype, device=A.device)
    for eid in range(B.shape[0]):
        mask = group_ids == eid
        y_groups[mask] = A_groups[mask] @ B[eid]
    return y_groups.reshape(M_total, Y)
