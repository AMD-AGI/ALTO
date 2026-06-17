# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Unit tests for observer_hooks.py — loaded directly to avoid triton imports.

observer_hooks.py has no alto or triton imports so it can be exec'd in isolation
on a CPU-only box, following the same pattern as test_adahop_modifier_helpers.py.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

_HOOKS_PATH = (Path(__file__).resolve().parents[3] / "alto" / "modifiers" / "debug" / "observer_hooks.py")


def _load_hooks():
    name = "_debug_observer_hooks_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HOOKS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hooks():
    return _load_hooks()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_captures(fqn: str, active: bool = True) -> dict:
    return {fqn: {"active": active}}


def _step_ref(step: int = 0) -> list:
    return [step]


# ---------------------------------------------------------------------------
# nn.Linear forward pre-hook
# ---------------------------------------------------------------------------

class TestLinearFwdPreHook:

    def test_captures_input(self, hooks):
        fqn = "layers.0.linear"
        captures = _make_captures(fqn)
        step_ref = _step_ref(1)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=True, capture_weight=False
        )
        x = torch.randn(2, 4, requires_grad=True)
        hook(module, (x,))
        assert 1 in captures[fqn]
        assert "input" in captures[fqn][1]
        assert torch.allclose(captures[fqn][1]["input"], x.detach().cpu())

    def test_captures_weight(self, hooks):
        fqn = "layers.0.linear"
        captures = _make_captures(fqn)
        step_ref = _step_ref(2)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=False, capture_weight=True
        )
        hook(module, (torch.randn(2, 4),))
        assert "weight" in captures[fqn][2]
        assert captures[fqn][2]["weight"].shape == module.weight.shape

    def test_gate_prevents_capture(self, hooks):
        fqn = "layers.0.linear"
        captures = _make_captures(fqn, active=False)
        step_ref = _step_ref(0)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=True, capture_weight=True
        )
        hook(module, (torch.randn(2, 4),))
        assert 0 not in captures[fqn]

    def test_captured_tensor_not_requires_grad(self, hooks):
        fqn = "l"
        captures = _make_captures(fqn)
        step_ref = _step_ref(0)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=True, capture_weight=False
        )
        x = torch.randn(2, 4, requires_grad=True)
        hook(module, (x,))
        assert not captures[fqn][0]["input"].requires_grad


# ---------------------------------------------------------------------------
# nn.Linear backward hook
# ---------------------------------------------------------------------------

class TestLinearBwdHook:

    def test_captures_grad_output(self, hooks):
        fqn = "layers.0"
        captures = _make_captures(fqn)
        step_ref = _step_ref(3)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_bwd_hook(
            captures, fqn, step_ref, capture_grad_output=True, capture_grad_weight=False
        )
        go = torch.randn(2, 4)
        hook(module, (None,), (go,))
        assert "grad_output" in captures[fqn][3]
        assert torch.allclose(captures[fqn][3]["grad_output"], go.detach().cpu())

    def test_captures_grad_weight_from_grad_input(self, hooks):
        fqn = "layers.0"
        captures = _make_captures(fqn)
        step_ref = _step_ref(3)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_bwd_hook(
            captures, fqn, step_ref, capture_grad_output=False, capture_grad_weight=True
        )
        gw = torch.randn(4, 4)
        # grad_input = (grad_x, grad_weight) — mirrors what autograd.Function.backward returns
        hook(module, (torch.randn(2, 4), gw), (torch.randn(2, 4),))
        assert "grad_weight" in captures[fqn][3]
        assert torch.allclose(captures[fqn][3]["grad_weight"], gw.detach().cpu())

    def test_inactive_gate_bwd(self, hooks):
        fqn = "l"
        captures = _make_captures(fqn, active=False)
        step_ref = _step_ref(0)
        module = nn.Linear(4, 4, bias=False)
        hook = hooks.make_linear_bwd_hook(
            captures, fqn, step_ref, capture_grad_output=True, capture_grad_weight=True
        )
        hook(module, (None, torch.randn(4, 4)), (torch.randn(2, 4),))
        assert 0 not in captures[fqn]


# ---------------------------------------------------------------------------
# GptOssGroupedExperts hooks
# ---------------------------------------------------------------------------

class _FakeGroupedExperts(nn.Module):
    """Minimal stand-in for GptOssGroupedExperts (no triton ops)."""

    def __init__(self):
        super().__init__()
        self.mlp1_weight = nn.Parameter(torch.randn(2, 8, 4))
        self.mlp2_weight = nn.Parameter(torch.randn(2, 4, 8))

    def forward(self, x):
        return x


class TestGroupedExpertsHooks:

    def test_fwd_pre_captures_input(self, hooks):
        fqn = "moe.experts"
        captures = _make_captures(fqn)
        step_ref = _step_ref(1)
        module = _FakeGroupedExperts()
        hook = hooks.make_grouped_experts_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=True, capture_weight=False
        )
        x = torch.randn(4, 4)
        hook(module, (x,))
        assert "input" in captures[fqn][1]
        assert torch.allclose(captures[fqn][1]["input"], x.detach().cpu())

    def test_fwd_pre_captures_both_mlp_weights(self, hooks):
        fqn = "moe.experts"
        captures = _make_captures(fqn)
        step_ref = _step_ref(1)
        module = _FakeGroupedExperts()
        hook = hooks.make_grouped_experts_fwd_pre_hook(
            captures, fqn, step_ref, capture_input=False, capture_weight=True
        )
        hook(module, (torch.randn(4, 4),))
        assert "mlp1_weight" in captures[fqn][1]
        assert "mlp2_weight" in captures[fqn][1]

    def test_grad_weight_hooks_stored_under_separate_keys(self, hooks):
        fqn = "moe.experts"
        captures = _make_captures(fqn)
        step_ref = _step_ref(7)
        hook1 = hooks.make_grouped_experts_grad_weight_hook(
            captures, fqn, step_ref, key="grad_mlp1_weight", capture_grad_weight=True
        )
        hook2 = hooks.make_grouped_experts_grad_weight_hook(
            captures, fqn, step_ref, key="grad_mlp2_weight", capture_grad_weight=True
        )
        g1 = torch.randn(2, 8, 4)
        g2 = torch.randn(2, 4, 8)
        hook1(g1)
        hook2(g2)
        assert captures[fqn][7]["grad_mlp1_weight"].shape == g1.shape
        assert captures[fqn][7]["grad_mlp2_weight"].shape == g2.shape
