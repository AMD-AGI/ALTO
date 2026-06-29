# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Dispatch-layer guard tests for ``precision='amdfp4'``.

These tests lock down the AMD-FP4 entry of
:class:`alto.kernels.dispatch.tensor.NVFP4TrainingWeightWrapperTensor` (the
NVFP4 / AMD-FP4 family share one wrapper class):

* ``precision`` is the single source of truth for the inner-scale grid:
  ``precision='amdfp4'`` implies the UE5M3 grid with no separate knob to set
  or mis-set (there is no ``inner_scale_format`` field).
* Linear / grouped_mm dispatch under ``precision='amdfp4'`` routes to the
  AMD-FP4 thin wrappers (``_to_amdfp4_then_scaled_mm`` /
  ``_quantize_then_amdfp4_scaled_grouped_mm``), not the NVFP4 ones.
* ``precision='amdfp4'`` end-to-end produces a finite output of the
  expected shape on a small smoke shape (no silent fallback).
"""

from __future__ import annotations

import pytest
import torch

import alto.kernels.dispatch.tensor as dispatch_tensor
from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.conversion import swap_params
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor
from alto.kernels.fp4.amdfp4 import ALIGN_SIZE_M


def _make_amdfp4_config(**overrides) -> TrainingOpConfig:
    defaults = dict(
        precision="amdfp4",
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
    )
    defaults.update(overrides)
    return TrainingOpConfig(**defaults)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("AMD-FP4 dispatch guards require a CUDA device")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Config-level invariants
# ---------------------------------------------------------------------------

def test_amdfp4_config_has_no_inner_scale_format_knob():
    """The inner-scale grid is implied by ``precision`` alone — there is no
    separate ``inner_scale_format`` field to set or mis-set.  This locks in
    the design decision (precision is the single source of truth)."""
    cfg = _make_amdfp4_config()
    assert cfg.precision == "amdfp4"
    assert not hasattr(cfg, "inner_scale_format")


def test_amdfp4_swap_params_wraps_linear_weight():
    """Regression for BLOCKER-1: ``swap_params`` must support
    ``precision='amdfp4'``.  Before the fix, ``_get_tensor_cls_for_config``
    had no amdfp4 arm and raised ``ValueError`` at model setup.

    amdfp4 reuses the NVFP4 wrapper (it re-dispatches on ``config.precision``
    internally), so the swapped parameter must become an
    ``NVFP4TrainingWeightWrapperTensor`` carrying the amdfp4 config.  This is
    a CPU-only setup-time check — no kernel launch — so it does not require a
    GPU.
    """
    cfg = _make_amdfp4_config()
    linear = torch.nn.Linear(8, 4, bias=False)

    swap_params(linear, config=cfg)

    assert isinstance(linear.weight.data, NVFP4TrainingWeightWrapperTensor), (
        "swap_params(precision='amdfp4') must wrap the weight in the "
        "NVFP4/AMD-FP4 family wrapper tensor"
    )
    assert linear.weight.data.config.precision == "amdfp4"


# ---------------------------------------------------------------------------
# linear dispatch routing
# ---------------------------------------------------------------------------

def test_amdfp4_linear_routes_to_amdfp4_thin_wrapper(monkeypatch, device):
    """Dispatch must route AMD-FP4 wrapped linear to ``_to_amdfp4_then_scaled_mm``,
    not ``_to_nvfp4_then_scaled_mm``."""
    calls = []

    def _mock_amdfp4(A, W, *, use_2dblock_x, use_2dblock_w,
                     use_sr_grad, use_outer_scale, use_hadamard, use_dge,
                     use_outer_block_scale=False, use_outer_2dblock_x=False,
                     use_outer_2dblock_w=False, outer_block_size=128):
        calls.append({
            "use_2dblock_x": use_2dblock_x,
            "use_2dblock_w": use_2dblock_w,
            "use_sr_grad": use_sr_grad,
            "use_outer_scale": use_outer_scale,
            "use_hadamard": use_hadamard,
            "use_dge": use_dge,
        })
        return A @ W

    def _forbidden_nvfp4(*args, **kwargs):
        raise AssertionError(
            "precision='amdfp4' must NOT call into _to_nvfp4_then_scaled_mm"
        )

    monkeypatch.setattr(dispatch_tensor, "_to_amdfp4_then_scaled_mm", _mock_amdfp4)
    monkeypatch.setattr(dispatch_tensor, "_to_nvfp4_then_scaled_mm", _forbidden_nvfp4)

    cfg = _make_amdfp4_config()
    K, N, M = 16, 16, 16
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)

    y = torch.nn.functional.linear(A, W_wrapped)
    assert y.shape == (M, N)
    # ``F.linear`` may dispatch through ``linear`` and additionally re-enter
    # via the ``mm.default`` / ``matmul`` decomposition path (we route both
    # in ``gemm_ops``).  We just need every routed call to land on the
    # AMD-FP4 thin wrapper.
    assert len(calls) >= 1, "F.linear with AMD-FP4 weight should hit the amdfp4 routing path"
    for call in calls:
        # AMD-FP4 wrapper has no ``scale_format`` parameter; it's pinned UE5M3.
        assert "scale_format" not in call


def test_amdfp4_grouped_mm_routes_to_amdfp4_thin_wrapper(monkeypatch, device):
    """Dispatch must route AMD-FP4 wrapped grouped_mm to
    ``_quantize_then_amdfp4_scaled_grouped_mm``."""
    calls = []

    def _mock_amdfp4_grouped(A, B, *, offs, use_2dblock_x, use_2dblock_w,
                             use_sr_grad, use_outer_scale, use_hadamard, use_dge,
                             use_outer_block_scale=False, use_outer_2dblock_w=False,
                             outer_block_size=128):
        calls.append({
            "offs_len": offs.numel(),
            "use_2dblock_x": use_2dblock_x,
            "use_2dblock_w": use_2dblock_w,
            "use_sr_grad": use_sr_grad,
            "use_outer_scale": use_outer_scale,
            "use_hadamard": use_hadamard,
            "use_dge": use_dge,
        })
        return torch.zeros(A.shape[0], B.shape[-1], dtype=A.dtype, device=A.device)

    def _forbidden_nvfp4_grouped(*args, **kwargs):
        raise AssertionError(
            "precision='amdfp4' grouped_mm must NOT call into _quantize_then_nvfp4_scaled_grouped_mm"
        )

    monkeypatch.setattr(
        dispatch_tensor, "_quantize_then_amdfp4_scaled_grouped_mm", _mock_amdfp4_grouped
    )
    monkeypatch.setattr(
        dispatch_tensor, "_quantize_then_nvfp4_scaled_grouped_mm", _forbidden_nvfp4_grouped
    )

    cfg = _make_amdfp4_config()
    num_experts, K, N, M = 2, 16, 16, ALIGN_SIZE_M
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device=device)

    _ = torch._grouped_mm(A, W_wrapped, offs=offs)
    assert len(calls) >= 1, "torch._grouped_mm with AMD-FP4 weight should hit the amdfp4 routing path"
    for call in calls:
        assert "scale_format" not in call


# ---------------------------------------------------------------------------
# end-to-end smoke
# ---------------------------------------------------------------------------

def test_amdfp4_linear_smoke(device):
    """End-to-end: precision='amdfp4' linear must produce a finite BF16 output
    of the expected shape (no silent fallback)."""
    cfg = _make_amdfp4_config()
    K, N, M = 32, 32, 32
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    y = torch.nn.functional.linear(A, W_wrapped)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
    assert y.abs().max().item() > 0, "AMD-FP4 linear smoke must exercise a real matmul"


def test_amdfp4_grouped_mm_smoke(device):
    """End-to-end: precision='amdfp4' grouped_mm must produce a finite output."""
    cfg = _make_amdfp4_config()
    num_experts, K, N, M = 2, 16, 16, ALIGN_SIZE_M
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device=device)
    y = torch._grouped_mm(A, W_wrapped, offs=offs)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
    assert y.abs().max().item() > 0, "AMD-FP4 grouped_mm smoke must exercise a real matmul"
