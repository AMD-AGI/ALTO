# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Toy integration test for MX9 W+A dynamic quantization.

This intentionally stays below a real llama3 validation run.  It verifies the
first full ALTO wiring layer:

recipe yaml -> QuantizationModifier -> Linear quantization_scheme ->
post_step dynamic-weight skip -> wrapped Linear forward -> MX9 W+A QDQ.
"""

import importlib.util
import os
import sys
import types

import torch
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _make_pkg(monkeypatch, name: str, path: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load(monkeypatch, modname: str, relpath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, modname, mod)
    spec.loader.exec_module(mod)
    return mod


def _install_torchtitan_stubs(monkeypatch) -> None:
    """Provide the tiny torchtitan surface needed by quantization modules."""
    _make_pkg(monkeypatch, "torchtitan", "")
    _make_pkg(monkeypatch, "torchtitan.tools", "")

    logging_mod = types.ModuleType("torchtitan.tools.logging")

    class _Logger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    logging_mod.logger = _Logger()
    monkeypatch.setitem(sys.modules, "torchtitan.tools.logging", logging_mod)

    utils_mod = types.ModuleType("torchtitan.tools.utils")
    utils_mod.device_type = torch.device("cpu")
    monkeypatch.setitem(sys.modules, "torchtitan.tools.utils", utils_mod)


def _load_quantization_modifier(monkeypatch):
    """Load only the ALTO modules needed for QuantizationModifier.

    Importing top-level ``alto`` currently pulls unrelated torchtitan entry
    points, so the test loads the required modules directly. All ``sys.modules``
    mutations go through ``monkeypatch`` so they are reverted at test teardown
    and cannot leak into other tests in the same process.
    """
    _install_torchtitan_stubs(monkeypatch)

    _make_pkg(monkeypatch, "alto", os.path.join(ROOT, "alto"))
    _make_pkg(monkeypatch, "alto.models", os.path.join(ROOT, "alto/models"))

    # Dynamic W/A uses no observers, but calibration.py imports Observer.
    observers_mod = types.ModuleType("alto.observers")

    class _Observer:
        @staticmethod
        def create_instance(*args, **kwargs):
            raise AssertionError("MX9 dynamic W+A should not create observers")

    observers_mod.Observer = _Observer
    monkeypatch.setitem(sys.modules, "alto.observers", observers_mod)

    _make_pkg(monkeypatch, "alto.modifiers", os.path.join(ROOT, "alto/modifiers"))
    _make_pkg(monkeypatch, "alto.modifiers.utils", os.path.join(ROOT, "alto/modifiers/utils"))
    hooks_mod = _load(monkeypatch, "alto.modifiers.utils.hooks", "alto/modifiers/utils/hooks.py")
    sys.modules["alto.modifiers.utils"].HooksMixin = hooks_mod.HooksMixin

    modifier_base_mod = _load(monkeypatch, "alto.modifiers.base", "alto/modifiers/base.py")
    sys.modules["alto.modifiers"].Modifier = modifier_base_mod.Modifier

    _make_pkg(monkeypatch, "alto.modifiers.quantization", os.path.join(ROOT, "alto/modifiers/quantization"))
    # Inject the format field (and load the MX module) before QuantizationModifier.
    _load(monkeypatch, "alto.modifiers.quantization.format_registry", "alto/modifiers/quantization/format_registry.py")
    quantize_mod = _load(monkeypatch, "alto.modifiers.quantization.mx", "alto/modifiers/quantization/mx.py")
    _load(monkeypatch, "alto.modifiers.quantization.calibration", "alto/modifiers/quantization/calibration.py")
    _load(monkeypatch, "alto.modifiers.quantization.mixin", "alto/modifiers/quantization/mixin.py")

    # base.py imports get_layers for sequential mode; this test uses sequential=False.
    _make_pkg(monkeypatch, "alto.utils", os.path.join(ROOT, "alto/utils"))
    _make_pkg(monkeypatch, "alto.utils.pytorch", os.path.join(ROOT, "alto/utils/pytorch"))
    module_utils = types.ModuleType("alto.utils.pytorch.module")
    module_utils.get_layers = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "alto.utils.pytorch.module", module_utils)

    quant_base_mod = _load(monkeypatch, "alto.modifiers.quantization.base", "alto/modifiers/quantization/base.py")

    patcher_mod = _load(monkeypatch, "alto.models.patcher", "alto/models/patcher.py")
    patcher_mod.ModelPatcher.patch_fake_quantize()

    return quant_base_mod.QuantizationModifier, quantize_mod


def _load_mx9_modifier_from_recipe(monkeypatch):
    recipe_path = os.path.join(ROOT, "alto/models/llama3/configs/mx9_wa_recipe.yaml")
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)

    mod_args = recipe["quantization_stage"]["quantization_modifiers"]["QuantizationModifier"]
    QuantizationModifier, quantize_mod = _load_quantization_modifier(monkeypatch)
    return QuantizationModifier(**mod_args), quantize_mod


def test_mx9_wa_recipe_toy_linear_lifecycle(monkeypatch):
    modifier, quantize_mod = _load_mx9_modifier_from_recipe(monkeypatch)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    linear = model[0]

    modifier.initialize([model])

    scheme = linear.quantization_scheme
    assert getattr(scheme.weights, "format", None) == "mx9"
    assert getattr(scheme.input_activations, "format", None) == "mx9"
    assert scheme.weights.dynamic is True
    assert scheme.input_activations.dynamic is True
    assert not hasattr(linear, "weight_observer")
    assert not hasattr(linear, "input_observer")

    calls = []
    original_mx9 = quantize_mod.mx9_fake_quantize

    def counted_mx9(input_tensor, *args, **kwargs):
        calls.append(tuple(input_tensor.shape))
        return original_mx9(input_tensor, *args, **kwargs)

    monkeypatch.setattr(quantize_mod, "mx9_fake_quantize", counted_mx9)

    x = torch.randn(2, 16)
    modifier.pre_step([model])
    modifier.post_step([model])  # must skip static weight baking for MX9 dynamic weight
    assert not hasattr(linear, "weight_observer")
    assert not hasattr(linear, "weight_scale")

    out = model(x)
    assert out.shape == (2, 16)
    assert len(calls) == 2
    assert (2, 16) in calls      # input activation QDQ
    assert (16, 16) in calls     # weight QDQ

    modifier.finalize([model])
