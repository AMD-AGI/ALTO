# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal helpers for NVFP4 grouped GEMM.

Notably includes ``_use_cdna4_grouped_backend()``, which decides whether the
grouped GEMM routes to the CDNA4-only Triton kernels reused from
``alto.kernels.blockwise_fp8.grouped_gemm`` (``cg_grouped_gemm_forward`` /
``cg_grouped_gemm_backward_inputs`` / ``cg_grouped_gemm_backward_weights``).
Despite the old name this check has nothing to do with ``torch._grouped_mm``;
it purely selects the CDNA4 Triton grouped backend over the generic
Python-loop fallback.
"""

from __future__ import annotations

from typing import Optional

import torch

from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync
from alto.kernels.fp4.nvfp4.nvfp_quantization import is_cdna4

from .autotune import ALIGN_SIZE_M, QDQ_AXIS0_BLOCK_SIZE

_CDNA4_GROUPED_BACKEND_AVAILABLE: Optional[bool] = None


def _use_cdna4_grouped_backend() -> bool:
    """True when the CDNA4 Triton grouped backend is available.

    The result is cached on first call because the underlying
    ``is_cdna4()`` probe hits the Triton driver and is not free.
    """
    global _CDNA4_GROUPED_BACKEND_AVAILABLE
    if _CDNA4_GROUPED_BACKEND_AVAILABLE is not None:
        return _CDNA4_GROUPED_BACKEND_AVAILABLE
    _CDNA4_GROUPED_BACKEND_AVAILABLE = bool(torch.cuda.is_available() and is_cdna4())
    return _CDNA4_GROUPED_BACKEND_AVAILABLE


def _reset_cdna4_grouped_backend_cache() -> None:
    """Invalidate the cached backend decision.

    Test-only helper: re-probe the CDNA4 backend after ``is_cdna4`` (or a
    surrounding ``torch.cuda.is_available()``) has been monkey-patched.
    """
    global _CDNA4_GROUPED_BACKEND_AVAILABLE
    _CDNA4_GROUPED_BACKEND_AVAILABLE = None


def _check_grouped_loop_contract(M_total: int, *, where: str) -> None:
    torch._check(
        M_total > 0 and M_total % ALIGN_SIZE_M == 0,
        lambda: (
            f"{where} (loop backend) requires M_total ({M_total}) to be a "
            f"positive multiple of ALIGN_SIZE_M ({ALIGN_SIZE_M})."
        ),
    )


def _check_grouped_axis0_qdq_contract(M_total: int, *, where: str) -> None:
    torch._check(
        M_total > 0 and M_total % QDQ_AXIS0_BLOCK_SIZE == 0,
        lambda: (
            f"{where} requires M_total ({M_total}) to be a positive multiple "
            f"of QDQ_AXIS0_BLOCK_SIZE ({QDQ_AXIS0_BLOCK_SIZE}) so the axis=0 "
            "NVFP4 QDQ view is well-defined."
        ),
    )


def _resolve_expert_indices(
    *,
    expert_indices: Optional[torch.Tensor],
    offs: Optional[torch.Tensor],
    M_total: int,
) -> torch.Tensor:
    """Normalize routing information to a contiguous int32 expert-index tensor."""
    if expert_indices is None:
        torch._check(offs is not None, lambda: "Either expert_indices or offs must be provided.")
        expert_indices = create_indices_from_offsets_nosync(offs).contiguous()
    elif expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)
    if not expert_indices.is_contiguous():
        expert_indices = expert_indices.contiguous()
    torch._check(
        expert_indices.shape[0] == M_total,
        lambda: (
            f"expert_indices.shape[0] ({expert_indices.shape[0]}) must match "
            f"M_total ({M_total})."
        ),
    )
    return expert_indices


def _group_ids_from_expert_indices(expert_indices: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Return one expert id per ALIGN_SIZE_M-sized group without host sync."""
    return expert_indices.view(num_groups, ALIGN_SIZE_M)[:, 0].contiguous()
