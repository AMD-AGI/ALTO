# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for grad-clip registry, apply_clip helper, and GradientClippingModifier.

CPU-safe: tests that require GPU (actual MXFP4LinearFunction backward injection)
are skipped when no GPU is available.

Uses the same importlib stub pattern as test_debug_observer_modifier.py to avoid
triggering the triton driver on import.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub installation (mirrors test_debug_observer_modifier.py)
# ---------------------------------------------------------------------------

def _install_stubs():
    if "_alto_stubs_installed" in sys.modules:
        return

    tt_tools = ModuleType("torchtitan.tools")
    tt_logging = ModuleType("torchtitan.tools.logging")
    tt_logging.logger = MagicMock()
    sys.modules.setdefault("torchtitan", ModuleType("torchtitan"))
    sys.modules.setdefault("torchtitan.tools", tt_tools)
    sys.modules["torchtitan.tools.logging"] = tt_logging

    def _match_named_modules(model, targets, ignore):
        ignore_patterns = [ig.lstrip("re:") for ig in ignore]
        for fqn, module in model.named_modules():
            if not fqn:
                continue
            cls_name = module.__class__.__name__
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

    from pydantic import BaseModel, PrivateAttr, Field

    class _FakeModifier(BaseModel):
        model_config = {"extra": "forbid"}
        index: int | None = None
        group: str | None = None
        start: float | None = None
        end: float | None = None
        update: float | None = None
        initialized_: bool = False
        finalized_: bool = False

        def initialize(self, model_parts, **kwargs):
            self.initialized_ = self.on_initialize(model_parts, **kwargs)

        def finalize(self, model_parts, **kwargs):
            self.finalized_ = self.on_finalize(model_parts, **kwargs)

        def pre_step(self, model_parts, **kwargs):
            return self.on_pre_step(model_parts, **kwargs)

        def post_step(self, model_parts, **kwargs):
            return self.on_post_step(model_parts, **kwargs)

        def convert(self, model, **kwargs):
            return self.on_convert(model, **kwargs)

    alto_mod = ModuleType("alto")
    alto_modifiers = ModuleType("alto.modifiers")
    alto_modifiers.Modifier = _FakeModifier
    sys.modules["alto"] = alto_mod
    sys.modules["alto.modifiers"] = alto_modifiers

    sys.modules["_alto_stubs_installed"] = True


def _load_file(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_grad_clip_modules():
    _install_stubs()
    fp4_common = Path(__file__).resolve().parents[3] / "alto" / "kernels" / "fp4" / "fp4_common"
    cfg_mod = _load_file("alto.kernels.fp4.fp4_common.grad_clip_config", fp4_common / "grad_clip_config.py")
    reg_mod = _load_file("alto.kernels.fp4.fp4_common.grad_clip_registry", fp4_common / "grad_clip_registry.py")
    return cfg_mod, reg_mod


def _load_modifier_module():
    _install_stubs()
    cfg_mod, reg_mod = _load_grad_clip_modules()

    # Ensure the fp4_common package stub is in sys.modules so grad_clip.py can import from it.
    fp4_pkg = sys.modules.get("alto.kernels.fp4.fp4_common")
    if fp4_pkg is None:
        fp4_pkg = ModuleType("alto.kernels.fp4.fp4_common")
        sys.modules["alto.kernels.fp4.fp4_common"] = fp4_pkg
    fp4_pkg.grad_clip_registry = reg_mod

    alto_kernels = sys.modules.get("alto.kernels", ModuleType("alto.kernels"))
    alto_kernels_fp4 = sys.modules.get("alto.kernels.fp4", ModuleType("alto.kernels.fp4"))
    sys.modules.setdefault("alto.kernels", alto_kernels)
    sys.modules.setdefault("alto.kernels.fp4", alto_kernels_fp4)

    lpt_root = Path(__file__).resolve().parents[3] / "alto" / "modifiers" / "lpt"
    return _load_file("_grad_clip_modifier_under_test", lpt_root / "grad_clip.py")


@pytest.fixture(scope="module")
def modules():
    cfg_mod, reg_mod = _load_grad_clip_modules()
    return cfg_mod, reg_mod


@pytest.fixture(scope="module")
def modifier_module():
    return _load_modifier_module()


@pytest.fixture(scope="module")
def GradClipConfig(modules):
    cfg_mod, _ = modules
    return cfg_mod.GradClipConfig


@pytest.fixture(scope="module")
def registry(modules):
    _, reg_mod = modules
    return reg_mod


@pytest.fixture(scope="module")
def GradientClippingModifier(modifier_module):
    return modifier_module.GradientClippingModifier


# ---------------------------------------------------------------------------
# Tiny model for modifier tests
# ---------------------------------------------------------------------------

class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# apply_clip tests
# ---------------------------------------------------------------------------

class TestApplyClip:

    def test_apply_clip_norm(self, registry):
        t = torch.randn(32, 32)
        t = t * (10.0 / t.norm())  # set norm to 10
        clipped = registry.apply_clip(t, max_norm=0.5, clip_value=None)
        assert clipped.norm().item() <= 0.5 + 1e-5

    def test_apply_clip_value(self, registry):
        t = torch.full((4, 4), 5.0)
        clipped = registry.apply_clip(t, max_norm=None, clip_value=2.0)
        assert clipped.abs().max().item() <= 2.0 + 1e-6

    def test_apply_clip_both(self, registry):
        # Norm applied first, then value clamp.
        t = torch.full((4, 4), 5.0)  # norm = 5*4 = 20
        clipped = registry.apply_clip(t, max_norm=1.0, clip_value=0.1)
        # After norm clip: all elements ≈ 1/16; after value clamp still ≤ 0.1
        assert clipped.abs().max().item() <= 0.1 + 1e-6

    def test_apply_clip_noop(self, registry):
        t = torch.randn(4, 4)
        result = registry.apply_clip(t, max_norm=None, clip_value=None)
        assert result is t


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_register_and_get(self, registry, GradClipConfig):
        cfg = GradClipConfig(clip_grad_output=True, grad_output_max_norm=1.0)
        fake_id = 999999
        registry.register(fake_id, cfg)
        retrieved = registry.get(fake_id)
        assert retrieved is cfg
        registry.deregister(fake_id)
        assert registry.get(fake_id) is None

    def test_get_none_id(self, registry):
        assert registry.get(None) is None

    def test_deregister_missing_is_noop(self, registry):
        registry.deregister(0)  # should not raise


# ---------------------------------------------------------------------------
# GradientClippingModifier lifecycle tests
# ---------------------------------------------------------------------------

class TestGradientClippingModifierLifecycle:

    def _make_modifier(self, GradientClippingModifier, **kwargs):
        defaults = dict(
            targets=["Linear"],
            ignore=[],
            clip_grad_output=True,
            grad_output_max_norm=1.0,
            grad_output_clip_value=None,
            clip_grad_weight=True,
            grad_weight_max_norm=1.0,
            grad_weight_clip_value=None,
        )
        defaults.update(kwargs)
        return GradientClippingModifier(**defaults)

    def test_initialize_registers_modules(self, GradientClippingModifier, registry):
        model = _TinyMLP()
        mod = self._make_modifier(GradientClippingModifier)
        mod.initialize([model])
        for _fqn, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert registry.get(id(module)) is not None

    def test_forward_pre_hook_stamps_module_id(self, GradientClippingModifier):
        """The forward_pre_hook must stamp module_id on the weight each forward.

        At initialize time module_id is NOT set (FSDP drops instance attrs on
        each all-gather). Instead, a forward_pre_hook re-stamps it before
        __torch_function__ is called. Simulate that by running a forward pass
        with a wrapped weight that has module_id=None before the hook fires.
        """

        class _WrappedParam(nn.Parameter):
            module_id: int | None = None

        model = _TinyMLP()
        stub = _WrappedParam(model.fc1.weight.data)
        stub.module_id = None
        model.fc1.weight = stub

        mod = self._make_modifier(GradientClippingModifier)
        mod.initialize([model])

        # module_id is still None at this point — that's expected.
        assert stub.module_id is None

        # Trigger one forward pass; the pre-hook should stamp module_id.
        model(torch.randn(2, 8))
        assert stub.module_id == id(model.fc1)

    def test_finalize_deregisters_all(self, GradientClippingModifier, registry):
        model = _TinyMLP()
        mod = self._make_modifier(GradientClippingModifier)
        mod.initialize([model])
        ids_before = [id(m) for _, m in model.named_modules() if isinstance(m, nn.Linear)]
        mod.finalize([model])
        for mid in ids_before:
            assert registry.get(mid) is None
        assert len(mod._hook_handles) == 0


# ---------------------------------------------------------------------------
# GPU-only: actual backward injection test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
class TestClippingBoundsGrad:

    def test_clipping_bounds_grad_weight(self):
        """Verify that the clip actually fires in MXFP4LinearFunction.backward.

        This test imports the real kernel stack (triton required) and checks that
        grad_weight's abs max is bounded after a forward+backward with a tight clip.
        """
        import importlib
        mxfp_linear = importlib.import_module("alto.kernels.fp4.mxfp4.mxfp_linear")
        grad_clip_registry = importlib.import_module("alto.kernels.fp4.fp4_common.grad_clip_registry")
        GradClipConfig = importlib.import_module("alto.kernels.fp4.fp4_common.grad_clip_config").GradClipConfig

        M, N, K = 16, 16, 16
        x = torch.randn(M, K, device="cuda", requires_grad=True)
        w = torch.randn(N, K, device="cuda", requires_grad=True)

        clip_value = 1e-3
        fake_module_id = id(w)
        cfg = GradClipConfig(
            clip_grad_weight=True,
            grad_weight_clip_value=clip_value,
        )
        grad_clip_registry.register(fake_module_id, cfg)

        try:
            y = mxfp_linear._to_mxfp4_then_scaled_mm(
                x, w,
                use_2dblock_x=False,
                use_2dblock_w=True,
                use_sr_grad=False,
                use_dge=False,
                clip_mode="none",
                use_hadamard=False,
                module_id=fake_module_id,
            )
            y.sum().backward()
            assert w.grad is not None
            assert w.grad.abs().max().item() <= clip_value + 1e-6
        finally:
            grad_clip_registry.deregister(fake_module_id)
