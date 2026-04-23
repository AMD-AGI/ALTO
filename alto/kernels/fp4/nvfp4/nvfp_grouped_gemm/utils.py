# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4-specific grouped GEMM helpers.

Shared pieces (backend probe, ``ALIGN_SIZE_M`` loop contract, expert-index
normalization, group-id extraction, Hadamard construction) live in
``alto.kernels.fp4.fp4_common.grouped_utils`` and are re-exported here for
backward compatibility with legacy imports.

Only truly NVFP4-specific contracts (the axis-0 QDQ block-size alignment) stay
in this module.
"""

from __future__ import annotations

import torch

from alto.kernels.fp4.fp4_common import (  # noqa: F401 — re-exported for BC
    build_hadamard_transform_if_needed,
    check_grouped_loop_contract,
    group_ids_from_expert_indices,
    reset_cdna4_grouped_backend_cache,
    resolve_expert_indices,
    use_cdna4_grouped_backend,
)

from .autotune import QDQ_AXIS0_BLOCK_SIZE


def check_grouped_axis0_qdq_contract(M_total: int, *, where: str) -> None:
    """Require ``M_total`` to be a positive multiple of ``QDQ_AXIS0_BLOCK_SIZE``.

    NVFP4-specific: the axis=0 QDQ view used on the wgrad reduction axis is
    only well-defined when ``M_total`` is aligned to the block size.
    """
    torch._check(
        M_total > 0 and M_total % QDQ_AXIS0_BLOCK_SIZE == 0,
        lambda: (
            f"{where} requires M_total ({M_total}) to be a positive multiple "
            f"of QDQ_AXIS0_BLOCK_SIZE ({QDQ_AXIS0_BLOCK_SIZE}) so the axis=0 "
            "NVFP4 QDQ view is well-defined."
        ),
    )


# --- Legacy private aliases ------------------------------------------------
# The NVFP4 code inside this package historically used underscore-prefixed
# names; keep them as aliases so existing imports keep working.

_use_cdna4_grouped_backend = use_cdna4_grouped_backend
_reset_cdna4_grouped_backend_cache = reset_cdna4_grouped_backend_cache
_check_grouped_loop_contract = check_grouped_loop_contract
_check_grouped_axis0_qdq_contract = check_grouped_axis0_qdq_contract
_resolve_expert_indices = resolve_expert_indices
_group_ids_from_expert_indices = group_ids_from_expert_indices
