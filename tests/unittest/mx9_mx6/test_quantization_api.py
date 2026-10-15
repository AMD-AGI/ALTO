# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the model-agnostic quantization API in ``alto.apis``.

Importing top-level ``alto`` pulls torchtitan entry points and probes the GPU
driver, so the modules under test are loaded directly, as in the MX6/MX9
integration tests. Everything except the MX kernel test runs on CPU: an MX
recipe only reaches the Triton kernel in a forward pass with quantization
enabled, and calibration runs with quantization disabled.
"""

import copy
import importlib
import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import yaml
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils import match_named_modules

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

QUANTIZABLE = ("backbone.0", "backbone.2", "head.0", "head.2")


class _Logger:

    def __init__(self):
        self.warnings = []

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)


def _make_pkg(monkeypatch, name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    mod.__package__ = name
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load(monkeypatch, modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, modname, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def apis(monkeypatch):
    before = set(sys.modules)
    forward_module = importlib.import_module("compressed_tensors.quantization.lifecycle.forward")
    monkeypatch.setattr(forward_module, "fake_quantize", forward_module.fake_quantize)

    _make_pkg(monkeypatch, "torchtitan", "")
    _make_pkg(monkeypatch, "torchtitan.tools", "")
    logging_mod = types.ModuleType("torchtitan.tools.logging")
    logging_mod.logger = _Logger()
    monkeypatch.setitem(sys.modules, "torchtitan.tools.logging", logging_mod)
    utils_mod = types.ModuleType("torchtitan.tools.utils")
    utils_mod.device_type = torch.device("cpu")
    monkeypatch.setitem(sys.modules, "torchtitan.tools.utils", utils_mod)

    _make_pkg(monkeypatch, "alto", os.path.join(ROOT, "alto"))
    _make_pkg(monkeypatch, "alto.models", os.path.join(ROOT, "alto/models"))

    _make_pkg(monkeypatch, "alto.observers", os.path.join(ROOT, "alto/observers"))
    _load(monkeypatch, "alto.observers.helpers", "alto/observers/helpers.py")
    observer_base = _load(monkeypatch, "alto.observers.base", "alto/observers/base.py")
    _load(monkeypatch, "alto.observers.minmax", "alto/observers/minmax.py")
    sys.modules["alto.observers"].Observer = observer_base.Observer

    _make_pkg(monkeypatch, "alto.modifiers", os.path.join(ROOT, "alto/modifiers"))
    _make_pkg(monkeypatch, "alto.modifiers.utils", os.path.join(ROOT, "alto/modifiers/utils"))
    hooks_mod = _load(monkeypatch, "alto.modifiers.utils.hooks", "alto/modifiers/utils/hooks.py")
    sys.modules["alto.modifiers.utils"].HooksMixin = hooks_mod.HooksMixin
    modifier_base = _load(monkeypatch, "alto.modifiers.base", "alto/modifiers/base.py")
    sys.modules["alto.modifiers"].Modifier = modifier_base.Modifier

    _make_pkg(monkeypatch, "alto.modifiers.quantization", os.path.join(ROOT, "alto/modifiers/quantization"))
    _load(monkeypatch, "alto.modifiers.quantization.format_registry", "alto/modifiers/quantization/format_registry.py")
    _load(monkeypatch, "alto.modifiers.quantization.mx", "alto/modifiers/quantization/mx.py")
    _load(monkeypatch, "alto.modifiers.quantization.calibration", "alto/modifiers/quantization/calibration.py")
    _load(monkeypatch, "alto.modifiers.quantization.mixin", "alto/modifiers/quantization/mixin.py")
    _make_pkg(monkeypatch, "alto.utils", os.path.join(ROOT, "alto/utils"))
    _make_pkg(monkeypatch, "alto.utils.pytorch", os.path.join(ROOT, "alto/utils/pytorch"))
    module_utils = types.ModuleType("alto.utils.pytorch.module")
    module_utils.get_layers = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "alto.utils.pytorch.module", module_utils)
    quant_base = _load(monkeypatch, "alto.modifiers.quantization.base", "alto/modifiers/quantization/base.py")
    sys.modules["alto.modifiers.quantization"].QuantizationModifier = quant_base.QuantizationModifier

    # Registering by hand keeps Recipe.from_dict from walking, and so importing,
    # every modifier package.
    factory = _load(monkeypatch, "alto.modifiers.factory", "alto/modifiers/factory.py")
    factory.ModifierFactory._main_registry = {"QuantizationModifier": quant_base.QuantizationModifier}
    factory.ModifierFactory._loaded = True
    sys.modules["alto.modifiers"].ModifierFactory = factory.ModifierFactory

    _make_pkg(monkeypatch, "alto.config", os.path.join(ROOT, "alto/config"))
    _load(monkeypatch, "alto.config.utils", "alto/config/utils.py")
    recipe_mod = _load(monkeypatch, "alto.config.recipe", "alto/config/recipe.py")
    sys.modules["alto.config"].Recipe = recipe_mod.Recipe

    patcher = _load(monkeypatch, "alto.models.patcher", "alto/models/patcher.py")
    patcher.ModelPatcher.patch_fake_quantize()

    _make_pkg(monkeypatch, "alto.apis", os.path.join(ROOT, "alto/apis"))
    yield _load(monkeypatch, "alto.apis.quantization", "alto/apis/quantization.py")

    for name in set(sys.modules) - before:
        if name.split(".")[0] in ("alto", "torchtitan"):
            sys.modules.pop(name, None)


class _Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16))
        self.head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.head(self.backbone(x))


class _AttentionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(16, 2, batch_first=True)
        self.proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.proj(self.attn(x, x, x)[0])


def _calibration(model, *shape, batches=4):

    def calibrate():
        for _ in range(batches):
            model(torch.randn(*shape))

    return calibrate


def _config_groups(recipe):
    return {name: group.model_dump() for name, group in recipe.modifiers[0].resolved_config.config_groups.items()}


def _write_yaml(path, stage):
    path.write_text(yaml.safe_dump({"quantization_stage": stage}))
    return path


def _modifiers_section(recipe):
    return yaml.safe_load(recipe.yaml())["quantization_stage"]["quantization_modifiers"]


@pytest.mark.parametrize("scheme, num_bits", [("mx6", 5), ("mx9", 8)])
def test_mx_recipe_blocks_conv2d_along_input_channels(apis, scheme, num_bits):
    groups = apis.create_quantization_recipe(scheme, targets=["Conv2d", "Linear"]).modifiers[0].resolved_config.config_groups

    assert set(groups) == {"group_conv2d", "group_linear"}
    for name, block_axis in (("group_conv2d", -3), ("group_linear", -1)):
        for args in (groups[name].weights, groups[name].input_activations):
            assert (args.num_bits, args.dynamic, args.format, args.block_axis) == (num_bits, True, scheme, block_axis)


def test_create_rejects_unknown_schemes_and_targets(apis):
    with pytest.raises(ValueError, match="unsupported scheme 'W8A8'"):
        apis.create_quantization_recipe("W8A8")
    with pytest.raises(ValueError, match="targets must be drawn from"):
        apis.create_quantization_recipe("mx6", targets=["Conv1d"])


def test_a_bare_ignore_name_covers_its_subtree_only(apis):
    model = nn.ModuleDict({"head": nn.Sequential(nn.Linear(4, 4)), "header": nn.Linear(4, 4)})
    modifier = apis.create_quantization_recipe("mx6", ignore=["head"]).modifiers[0]

    kept = [name for name, _ in match_named_modules(model, modifier.resolved_targets, modifier.ignore)]
    assert kept == ["header"]


def test_a_created_recipe_survives_a_yaml_round_trip(apis, tmp_path):
    recipe = apis.create_quantization_recipe("mx9", ignore=["head"])
    path = tmp_path / "mx9.yaml"
    recipe.yaml(str(path))

    loaded = apis.load_quantization_recipe(path)
    assert _config_groups(loaded) == _config_groups(recipe)
    assert loaded.modifiers[0].ignore == recipe.modifiers[0].ignore


def test_apply_rejects_sequential_mode_and_unsupported_modifiers(apis, tmp_path):
    model = nn.Sequential(nn.Linear(4, 4))
    section = _modifiers_section(apis.create_quantization_recipe("mx6"))
    section["QuantizationModifier"]["sequential"] = True
    path = _write_yaml(tmp_path / "sequential.yaml", {"quantization_modifiers": section})
    recipe = apis.load_quantization_recipe(path)

    with pytest.raises(ValueError, match="sequential=True is not supported"):
        apis.apply_quantization(model, recipe)
    assert not hasattr(model[0], "quantization_scheme")

    QuantizationModifier = sys.modules["alto.modifiers.quantization"].QuantizationModifier
    factory = sys.modules["alto.modifiers.factory"].ModifierFactory
    factory._main_registry["BlockwiseModifier"] = type("BlockwiseModifier", (QuantizationModifier,), {})
    section["BlockwiseModifier"] = section.pop("QuantizationModifier")
    section["BlockwiseModifier"]["sequential"] = False
    path = _write_yaml(tmp_path / "blockwise.yaml", {"quantization_modifiers": section})
    recipe = apis.load_quantization_recipe(path)

    with pytest.raises(ValueError, match="BlockwiseModifier is not supported"):
        apis.apply_quantization(model, recipe)
    assert not hasattr(model[0], "quantization_scheme")


def test_int8_quantizes_every_linear(apis):
    torch.manual_seed(0)
    model = _Net()
    reference = copy.deepcopy(model)
    recipe = apis.create_quantization_recipe("int8")
    groups = recipe.modifiers[0].resolved_config.config_groups
    assert set(groups) == {"group_linear"}
    weights, inputs = groups["group_linear"].weights, groups["group_linear"].input_activations
    assert (weights.num_bits, weights.dynamic, weights.strategy) == (8, False, "tensor")
    assert (inputs.num_bits, inputs.dynamic, inputs.strategy, inputs.observer) == (8, False, "tensor", "minmax")

    result = apis.apply_quantization(model, recipe, calibrate=_calibration(model, 64, 8))
    assert result is model

    for name in QUANTIZABLE:
        module = model.get_submodule(name)
        assert module.quantization_status == QuantizationStatus.FROZEN
        assert torch.isfinite(module.input_scale).all()
        steps = module.weight / module.weight_scale
        assert torch.allclose(steps, steps.round(), atol=1e-3), f"{name} weight is not on the int8 grid"
        assert not torch.equal(module.weight, reference.get_submodule(name).weight)

    x = torch.randn(2, 8)
    out, expected = model(x), reference(x)
    assert not torch.equal(out, expected)
    assert torch.allclose(out, expected, atol=0.05)


def test_static_activations_require_calibration_before_initialization(apis):
    model = nn.Sequential(nn.Linear(4, 4))

    with pytest.raises(ValueError, match="needs calibration data"):
        apis.apply_quantization(model, apis.create_quantization_recipe("int8"))
    assert not hasattr(model[0], "quantization_scheme")


def test_a_dynamic_recipe_skips_calibrate_and_ends_frozen(apis):
    model = _Net()
    calls = []
    recipe = apis.create_quantization_recipe("mx6")
    assert set(recipe.modifiers[0].resolved_config.config_groups) == {"group_linear"}

    apis.apply_quantization(model, recipe, calibrate=lambda: calls.append(None))

    assert calls == []
    assert any("needs no calibration data" in message for message in apis.logger.warnings)
    for name in QUANTIZABLE:
        module = model.get_submodule(name)
        assert module.quantization_scheme.weights.format == "mx6"
        assert module.quantization_status == QuantizationStatus.FROZEN


def test_multihead_attention_out_proj_can_be_explicitly_ignored(apis):
    model = _AttentionNet()
    before = model.attn.out_proj.weight.detach().clone()

    apis.apply_quantization(model,
                            apis.create_quantization_recipe("int8", ignore=["attn.out_proj"]),
                            calibrate=_calibration(model, 2, 5, 16))

    assert not hasattr(model.attn.out_proj, "quantization_scheme")
    assert torch.equal(model.attn.out_proj.weight, before)
    assert torch.isfinite(model.proj.input_scale).all()
    assert model.proj.quantization_status == QuantizationStatus.FROZEN
    assert torch.isfinite(model(torch.randn(2, 5, 16))).all()


def test_one_recipe_object_quantizes_several_models(apis):
    recipe = apis.create_quantization_recipe("mx6")
    first, second = _Net(), _Net()

    apis.apply_quantization(first, recipe)
    apis.apply_quantization(second, recipe)

    assert not recipe.modifiers[0].initialized_
    assert all(hasattr(second.get_submodule(name), "quantization_scheme") for name in QUANTIZABLE)


def test_a_mixed_precision_recipe_runs_from_one_yaml(apis, tmp_path):
    stage = {
        "mx_modifiers": _modifiers_section(apis.create_quantization_recipe("mx6", ignore=["backbone"])),
        "int8_modifiers": _modifiers_section(apis.create_quantization_recipe("int8", ignore=["head"])),
    }
    path = _write_yaml(tmp_path / "mixed.yaml", stage)
    model = _Net()

    apis.apply_quantization(model, path, calibrate=_calibration(model, 2, 8))

    for name in ("head.0", "head.2"):
        module = model.get_submodule(name)
        assert module.quantization_scheme.weights.format == "mx6"
        assert module.quantization_status == QuantizationStatus.FROZEN
    for name in ("backbone.0", "backbone.2"):
        module = model.get_submodule(name)
        assert module.quantization_scheme.weights.dynamic is False
        assert torch.isfinite(module.input_scale).all()
        assert module.quantization_status == QuantizationStatus.FROZEN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MX dispatch launches the Triton kernel")
def test_the_conv2d_block_axis_reaches_the_mx_kernel_for_batched_and_unbatched_input(apis, monkeypatch):
    import alto.kernels.mx as mx_kernels
    from compressed_tensors.quantization.lifecycle import forward

    calls = []
    original = mx_kernels.convert_to_mx

    def counted(data, *args, axis=-1, **kwargs):
        calls.append((tuple(data.shape), axis))
        return original(data, *args, axis=axis, **kwargs)

    monkeypatch.setattr(mx_kernels, "convert_to_mx", counted)
    group = apis.create_quantization_recipe("mx6", targets=["Conv2d"]).modifiers[0].resolved_config.config_groups
    weights, inputs = group["group_conv2d"].weights, group["group_conv2d"].input_activations

    for args, shape in ((weights, (16, 32, 3, 3)), (inputs, (2, 32, 8, 8)), (inputs, (32, 8, 8))):
        x = torch.randn(*shape, device="cuda")
        assert forward.fake_quantize(x, None, None, args, None, None).shape == x.shape
    assert calls == [((16, 32, 3, 3), -3), ((2, 32, 8, 8), -3), ((32, 8, 8), -3)]
