# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Grouped-GEMM helpers shared by ``mxfp_grouped_gemm`` and ``nvfp_grouped_gemm``.

These utilities were originally duplicated (or silently inlined) in the two
family-specific grouped GEMM packages.  Hosting them here avoids drift and
makes the shared pieces of the MoE routing / backend-selection / layout
contract explicit.

What lives here
---------------

* :data:`DEFAULT_ALIGN_SIZE_M` — canonical MoE tile/alignment size; individual
  grouped GEMMs may override via the ``align_size_m`` kwarg on the helpers.
* :func:`use_cdna4_grouped_backend` — cached decision that chooses between the
  CDNA4 Triton grouped backend (from ``alto.kernels.blockwise_fp8``) and the
  generic Python-loop fallback.  Despite the name, this has **nothing** to do
  with ``torch._grouped_mm``; the probe purely inspects the Triton driver
  target arch.
* :func:`reset_cdna4_grouped_backend_cache` — test helper to invalidate the
  cached decision when ``is_cdna4`` or surrounding availability checks are
  monkey-patched.
* :func:`check_grouped_loop_contract` — fails fast when ``M_total`` is not a
  positive multiple of the configured alignment.
* :func:`resolve_expert_indices` — normalizes routing info (``offs`` or
  ``expert_indices``) to a contiguous ``int32`` ``[M_total]`` tensor.
* :func:`group_ids_from_expert_indices` — one expert id per
  ``align_size_m``-sized group without a host sync.
* :func:`build_hadamard_transform_if_needed` — constructs a
  :class:`HadamardTransform` when the recipe requests one.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from alto.kernels.dsgemm_utils import create_indices_from_offsets_nosync

DEFAULT_ALIGN_SIZE_M: int = 128


# --- Backend selection ------------------------------------------------------

_CDNA4_GROUPED_BACKEND_AVAILABLE: Optional[bool] = None


def _probe_is_cdna4() -> bool:
    """Probe the active Triton driver target for CDNA4 (``gfx950``)."""
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def use_cdna4_grouped_backend() -> bool:
    """True when the CDNA4 Triton grouped backend is available.

    The result is cached on first call because the underlying driver probe is
    not free.  Use :func:`reset_cdna4_grouped_backend_cache` in tests that
    need to re-probe.
    """
    global _CDNA4_GROUPED_BACKEND_AVAILABLE
    if _CDNA4_GROUPED_BACKEND_AVAILABLE is not None:
        return _CDNA4_GROUPED_BACKEND_AVAILABLE
    _CDNA4_GROUPED_BACKEND_AVAILABLE = bool(torch.cuda.is_available() and _probe_is_cdna4())
    return _CDNA4_GROUPED_BACKEND_AVAILABLE


def reset_cdna4_grouped_backend_cache() -> None:
    """Invalidate the cached backend decision.  Intended for tests."""
    global _CDNA4_GROUPED_BACKEND_AVAILABLE
    _CDNA4_GROUPED_BACKEND_AVAILABLE = None


# --- Shape / routing contracts ---------------------------------------------


def check_grouped_loop_contract(
    M_total: int,
    *,
    where: str,
    align_size_m: int = DEFAULT_ALIGN_SIZE_M,
) -> None:
    """Fail fast if ``M_total`` is not a positive multiple of ``align_size_m``."""
    torch._check(
        M_total > 0 and M_total % align_size_m == 0,
        lambda: (
            f"{where} (loop backend) requires M_total ({M_total}) to be a "
            f"positive multiple of ALIGN_SIZE_M ({align_size_m})."
        ),
    )


def resolve_expert_indices(
    *,
    expert_indices: Optional[torch.Tensor],
    offs: Optional[torch.Tensor],
    M_total: int,
) -> torch.Tensor:
    """Normalize routing information to a contiguous int32 expert-index tensor.

    Accepts either ``expert_indices`` or cumulative ``offs``.  ``M_total`` is
    the activation buffer length (``inputs.shape[0]``); the returned tensor
    has length equal to the number of routed tokens, which may be smaller
    when callers pad the buffer (e.g. GPT-OSS ``indices_padding_wrapper``).
    Trailing un-routed rows are treated as zeros by the grouped GEMM kernels.
    """
    if expert_indices is None:
        torch._check(offs is not None, lambda: "Either expert_indices or offs must be provided.")
        expert_indices = create_indices_from_offsets_nosync(offs).contiguous()
    elif expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)
    if not expert_indices.is_contiguous():
        expert_indices = expert_indices.contiguous()
    torch._check(
        expert_indices.shape[0] <= M_total,
        lambda: (
            f"expert_indices.shape[0] ({expert_indices.shape[0]}) must not "
            f"exceed activation buffer length M_total ({M_total})."
        ),
    )
    return expert_indices


def group_ids_from_expert_indices(
    expert_indices: torch.Tensor,
    num_groups: int,
    *,
    align_size_m: int = DEFAULT_ALIGN_SIZE_M,
) -> torch.Tensor:
    """Return one expert id per ``align_size_m``-sized group, without host sync."""
    return expert_indices.view(num_groups, align_size_m)[:, 0].contiguous()


# --- Recipe construction ----------------------------------------------------


def build_hadamard_transform_if_needed(
    *,
    use_hadamard: bool,
    use_2dblock_x: bool,
    device: torch.device,
):
    """Construct the grouped-path Hadamard transform if the recipe requests it."""
    if not use_hadamard:
        return None
    assert not use_2dblock_x, (
        "Hadamard transform can only be applied when use_2dblock_x=False."
    )
    # Local import to avoid a module-level cycle between fp4_common and the
    # hadamard package (which itself may pull in fp4-specific utilities).
    from alto.kernels.hadamard_transform import HadamardFactory
    with torch.no_grad():
        return HadamardFactory.create_transform(device=device)
