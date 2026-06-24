# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""De-oscillation QDQ support for ``precision='amdfp4'``.

The weight de-oscillation hook rebuilds the forward-path QDQ round trip via
:func:`alto.components.optimizer._make_qdq_fn_for`.  Before AMD-FP4 support
was wired in, that factory only knew ``mxfp4`` / ``nvfp4`` and raised
``ValueError`` for ``amdfp4`` — so enabling de-oscillation on an AMD-FP4 run
would have crashed (AMD-FP4 weights reuse the NVFP4 wrapper, which the hook
already peels).

These tests lock in:

* ``_make_qdq_fn_for`` accepts ``precision='amdfp4'`` (no longer raises);
* the produced closure routes to the UE5M3 AMD-FP4 convert ops (bit-identical
  to calling ``convert_from_amdfp4 ∘ convert_to_amdfp4`` directly);
* non-FP4 precisions are still rejected (the ``else`` guard is intact).
"""

from __future__ import annotations

import pytest
import torch

from alto.components.optimizer import (
    DeOscillationConfig,
    _DeOscillationHook,
    _make_qdq_fn_for,
)
from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor
from alto.kernels.fp4 import convert_from_amdfp4, convert_to_amdfp4


def _make_config(precision: str, **overrides) -> TrainingOpConfig:
    defaults = dict(
        precision=precision,
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
        two_level_scaling="tensorwise",
    )
    defaults.update(overrides)
    return TrainingOpConfig(**defaults)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("AMD-FP4 de-oscillation QDQ requires a CUDA / ROCm device")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Factory-level invariants (no kernel launch)
# ---------------------------------------------------------------------------

def test_make_qdq_fn_for_accepts_amdfp4():
    """Regression for H4: the factory must not raise for ``amdfp4``."""
    qdq = _make_qdq_fn_for(_make_config("amdfp4"))
    assert callable(qdq)


def test_make_qdq_fn_for_still_rejects_non_fp4():
    """Adversarial: a non-FP4 precision must still hit the ``else`` guard, so
    the amdfp4 arm didn't accidentally widen the accepted set."""
    with pytest.raises(ValueError, match="only supports FP4 wrappers"):
        _make_qdq_fn_for(_make_config("mxfp8_e4m3"))


# ---------------------------------------------------------------------------
# Numerical behaviour (GPU)
# ---------------------------------------------------------------------------

def test_amdfp4_qdq_roundtrip_finite(device):
    qdq = _make_qdq_fn_for(_make_config("amdfp4"))
    w = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    out = qdq(w, -1)
    assert out.shape == w.shape
    assert out.dtype == w.dtype
    assert torch.isfinite(out).all()
    # de-oscillation snaps to the FP4 bin center, so the round trip must move
    # the weight (a no-op would mean QDQ silently fell back to identity).
    assert (out.float() - w.float()).abs().max().item() > 0


def test_amdfp4_qdq_matches_direct_ue5m3_convert(device):
    """The closure must route through the UE5M3 AMD-FP4 convert ops, i.e. be
    bit-identical to calling them directly with the same (deterministic, SR-off)
    settings.  This proves precision='amdfp4' selects the UE5M3 grid."""
    cfg = _make_config("amdfp4")
    qdq = _make_qdq_fn_for(cfg)
    w = torch.randn(32, 32, dtype=torch.bfloat16, device=device)

    got = qdq(w, -1)

    outer_scale = torch.empty(1, dtype=torch.float32, device=device)
    data_lp, scales = convert_to_amdfp4(
        w, axis=-1, is_2d_block=True,
        outer_scale=outer_scale, update_outer_scale=True,
    )
    expected = convert_from_amdfp4(
        data_lp, scales, output_dtype=w.dtype,
        axis=-1, is_2d_block=True, outer_scale=outer_scale,
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Hook-level wiring (the eligibility gate, not just the QDQ factory)
# ---------------------------------------------------------------------------

def test_amdfp4_param_is_deosc_eligible():
    """The hook's eligibility gate must accept amdfp4-wrapped params.  Before
    this fix it only allowed mxfp4/nvfp4, so AMD-FP4 weights were silently
    skipped even though ``_make_qdq_fn_for`` supports them — i.e. the path was
    not actually open end to end."""
    hook = _DeOscillationHook(DeOscillationConfig(enable=True))
    w = torch.randn(32, 32, dtype=torch.bfloat16)
    param = torch.nn.Parameter(
        NVFP4TrainingWeightWrapperTensor(w, _make_config("amdfp4")),
        requires_grad=True,
    )
    wrapper = hook._wrapper_if_eligible(param)
    assert wrapper is not None, "amdfp4 param must be eligible for de-oscillation"
    assert isinstance(wrapper, NVFP4TrainingWeightWrapperTensor)
    assert wrapper.config.precision == "amdfp4"


def test_amdfp4_deosc_hook_tracks_param(device):
    """End-to-end through the hook: an amdfp4 wrapper is seeded then QDQ-tracked
    across a period without raising, proving the gate + factory + hook chain is
    wired for AMD-FP4."""
    hook = _DeOscillationHook(DeOscillationConfig(enable=True, period=2))
    w = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    wrapper = NVFP4TrainingWeightWrapperTensor(w, _make_config("amdfp4"))
    state: dict = {}

    # First call seeds the period (no before/after pair yet).
    assert hook._step_param(wrapper, state) == (0, 0, False)
    assert _DeOscillationHook.KEY_PREV in state

    # Move the underlying weight, then take a tracked step.
    wrapper._data.add_(torch.randn_like(wrapper._data) * 0.01)
    hook._step_param(wrapper, state)
    dist_qdq = state[_DeOscillationHook.KEY_DIST_QDQ]
    assert torch.isfinite(dist_qdq).all()
