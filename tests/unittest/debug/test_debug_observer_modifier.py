# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""End-to-end tests for DebugObserverModifier.

These tests mock the alto package imports that trigger the triton driver so
they can run on a CPU-only box, following the pattern established in
tests/unittest/adahop/test_adahop_modifier_helpers.py.

The modifier's external dependencies (alto.modifiers.Modifier, HooksMixin,
match_named_modules, logger) are stubbed at the sys.modules level before the
module is exec'd.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub the heavy alto / torchtitan imports before loading the modifier module
# ---------------------------------------------------------------------------

def _install_stubs():
    """Install minimal stubs so debug_observer.py can be loaded without GPU."""
    if "_alto_stubs_installed" in sys.modules:
        return

    # --- torchtitan.tools.logging ---
    tt_tools = ModuleType("torchtitan.tools")
    tt_logging = ModuleType("torchtitan.tools.logging")
    tt_logging.logger = MagicMock()
    sys.modules.setdefault("torchtitan", ModuleType("torchtitan"))
    sys.modules.setdefault("torchtitan.tools", tt_tools)
    sys.modules["torchtitan.tools.logging"] = tt_logging

    # --- compressed_tensors.utils.match_named_modules ---
    def _match_named_modules(model, targets, ignore):
        """Simple FQN walker that matches by class name."""
        ignore_patterns = []
        for ig in ignore:
            ignore_patterns.append(ig.lstrip("re:"))

        for fqn, module in model.named_modules():
            if not fqn:
                continue
            cls_name = module.__class__.__name__
            if cls_name not in targets and "Linear" not in targets:
                continue
            matched = any(cls_name == t or cls_name.endswith(t) for t in targets)
            if not matched:
                continue
            if any(p in fqn for p in ignore_patterns):
                continue
            yield fqn, module

    ct_utils = ModuleType("compressed_tensors.utils")
    ct_utils.match_named_modules = _match_named_modules
    sys.modules.setdefault("compressed_tensors", ModuleType("compressed_tensors"))
    sys.modules["compressed_tensors.utils"] = ct_utils

    # --- alto.modifiers (Modifier base class) ---
    # We need a real Pydantic base class so PrivateAttr and Field work.
    from pydantic import BaseModel, PrivateAttr, Field
    from torch.utils.hooks import RemovableHandle

    class _FakeHooksMixin(BaseModel):
        model_config = {"extra": "forbid"}
        index: int | None = None
        group: str | None = None
        start: float | None = None
        end: float | None = None
        update: float | None = None
        initialized_: bool = False
        finalized_: bool = False
        started_: bool = False
        ended_: bool = False
        _hooks: set = PrivateAttr(default_factory=set)

        def register_hook(self, target, hook, hook_type, **kwargs):
            if hook_type in ("forward_pre", "forward", "full_backward"):
                handle = getattr(target, f"register_{hook_type}_hook")(hook, **kwargs)
            else:
                handle = getattr(target, f"register_{hook_type}_hook")(hook, **kwargs)
            self._hooks.add(handle)
            return handle

        def remove_hooks(self, handles=None):
            if handles is None:
                handles = set(self._hooks)
            for h in handles:
                h.remove()
            self._hooks -= handles

    class _FakeModifier(_FakeHooksMixin):

        @property
        def initialized(self):
            return self.initialized_

        @property
        def requires_training_mode(self):
            return False

        def initialize(self, model_parts, **kwargs):
            self.initialized_ = self.on_initialize(model_parts, **kwargs)

        def finalize(self, model_parts, **kwargs):
            self.finalized_ = self.on_finalize(model_parts, **kwargs)

        def pre_step(self, model_parts, **kwargs):
            self.started_ = self.on_pre_step(model_parts, **kwargs)

        def post_step(self, model_parts, **kwargs):
            self.ended_ = self.on_post_step(model_parts, **kwargs)

        def convert(self, model, **kwargs):
            return self.on_convert(model, **kwargs)

        def on_initialize(self, model_parts, **kwargs):
            raise NotImplementedError
        def on_finalize(self, model_parts, **kwargs):
            raise NotImplementedError
        def on_pre_step(self, model_parts, **kwargs):
            raise NotImplementedError
        def on_post_step(self, model_parts, **kwargs):
            raise NotImplementedError
        def on_convert(self, model, **kwargs):
            raise NotImplementedError

    alto_mod = ModuleType("alto")
    alto_modifiers = ModuleType("alto.modifiers")
    alto_modifiers.Modifier = _FakeModifier
    sys.modules["alto"] = alto_mod
    sys.modules["alto.modifiers"] = alto_modifiers

    sys.modules["_alto_stubs_installed"] = True


def _load_file_as_module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the debug_observer module in isolation
# ---------------------------------------------------------------------------

def _load_debug_observer():
    _install_stubs()

    debug_root = Path(__file__).resolve().parents[3] / "alto" / "modifiers" / "debug"

    # Register observer_hooks so that `from alto.modifiers.debug.observer_hooks import ...`
    # inside debug_observer.py resolves correctly.
    alto_debug = _load_file_as_module(
        "alto.modifiers.debug", debug_root / "__init__.py"
    )
    _load_file_as_module(
        "alto.modifiers.debug.observer_hooks", debug_root / "observer_hooks.py"
    )

    return _load_file_as_module(
        "_debug_observer_under_test", debug_root / "debug_observer.py"
    )


@pytest.fixture(scope="module")
def obs_module():
    return _load_debug_observer()


@pytest.fixture(scope="module")
def DebugObserverModifier(obs_module):
    return obs_module.DebugObserverModifier


# ---------------------------------------------------------------------------
# Tiny test model (plain nn.Linear, no triton)
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _run_n_steps(model, modifier, n: int):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    model_parts = [model]
    for _ in range(n):
        modifier.pre_step(model_parts)
        x = torch.randn(4, 8, requires_grad=True)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        modifier.post_step(model_parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDebugObserverModifierLifecycle:

    def _make_modifier(self, DebugObserverModifier, **kwargs):
        defaults = dict(
            targets=["Linear"],
            ignore=[],
            capture_every=1,
            max_captures=5,
            output_path="/tmp/_test_obs.pt",
        )
        defaults.update(kwargs)
        return DebugObserverModifier(**defaults)

    def test_initialize_registers_layers(self, DebugObserverModifier):
        model = _TinyMLP()
        mod = self._make_modifier(DebugObserverModifier)
        mod.initialize([model])
        assert len(mod._captures) == 2
        for fqn, data in mod._captures.items():
            assert "active" in data

    def test_captures_accumulate_over_steps(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(DebugObserverModifier, output_path=path, max_captures=3)
            mod.initialize([model])
            _run_n_steps(model, mod, n=3)
            assert mod._n_captured == 3

    def test_max_captures_hard_cap(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(DebugObserverModifier, output_path=path, max_captures=2)
            mod.initialize([model])
            _run_n_steps(model, mod, n=5)
            assert mod._n_captured <= 2
            assert mod._detached is True

    def test_finalize_writes_file(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(DebugObserverModifier, output_path=path, max_captures=2)
            mod.initialize([model])
            _run_n_steps(model, mod, n=2)
            mod.finalize([model])
            assert os.path.exists(path)

    def test_dump_has_expected_schema(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(DebugObserverModifier, output_path=path, max_captures=2)
            mod.initialize([model])
            _run_n_steps(model, mod, n=2)
            mod.finalize([model])
            blob = torch.load(path, map_location="cpu", weights_only=False)
            assert "_meta" in blob
            meta = blob["_meta"]
            assert "rank" in meta
            assert len(meta["iterations_captured"]) == 2
            for fqn, layer_data in blob.items():
                if fqn == "_meta":
                    continue
                for step, tensors in layer_data.items():
                    assert "input" in tensors
                    assert "grad_weight" in tensors

    def test_hooks_removed_after_finalize(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(DebugObserverModifier, output_path=path, max_captures=1)
            mod.initialize([model])
            _run_n_steps(model, mod, n=1)
            mod.finalize([model])
            assert len(mod._hooks) == 0
            assert len(mod._param_hook_handles) == 0

    def test_capture_every_skips_steps(self, DebugObserverModifier):
        model = _TinyMLP()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "obs.pt")
            mod = self._make_modifier(
                DebugObserverModifier, output_path=path, capture_every=2, max_captures=10
            )
            mod.initialize([model])
            _run_n_steps(model, mod, n=6)
            # capture_every=2, 6 steps → captures at steps 2, 4, 6
            assert mod._n_captured == 3
