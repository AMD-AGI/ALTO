# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]


def test_converter_defers_patcher_import(monkeypatch):

    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        return result

    class Configurable:

        class Config:
            pass

    class Linear:

        class Config:
            pass

    module("alto", __path__=[str(ROOT / "alto")])
    module("alto.components", __path__=[str(ROOT / "alto/components")])
    module("alto.config", Recipe=object)
    module("torch", nn=module("torch.nn", Module=object))
    module("torchtitan", __path__=[])
    module("torchtitan.components", __path__=[])
    module("torchtitan.components.model_converter", ModelConverter=type("ModelConverter", (), {}))
    module("torchtitan.config", Configurable=Configurable)
    module("torchtitan.distributed", ParallelismContext=object)
    module("torchtitan.models", __path__=[])
    module("torchtitan.models.common", __path__=[])
    module("torchtitan.models.common.linear", Linear=Linear)

    # Neither models nor the patcher may load while the converter is being defined.
    monkeypatch.setitem(sys.modules, "alto.models", None)
    monkeypatch.setitem(sys.modules, "alto.patches", None)
    monkeypatch.setitem(sys.modules, "alto.patches.patcher", None)
    converter_module = load("alto.components.converter", "alto/components/converter.py")

    load("alto.patches", "alto/patches/__init__.py")
    patcher_module = load("alto.patches.patcher", "alto/patches/patcher.py")
    patch_model_config = Mock()
    monkeypatch.setattr(patcher_module.ModelPatcher, "patch_model_config", patch_model_config)

    converter = object.__new__(converter_module.ModelOptConverter)
    modifier = Mock()
    converter.recipe = types.SimpleNamespace(modifiers=[modifier])
    linear_config = types.SimpleNamespace(inplace_wgrad_accum=True)
    model_config = Mock()
    model_config.traverse.return_value = [("linear", linear_config, None, None)]

    converter.convert_config(model_config)

    patch_model_config.assert_called_once_with(model_config)
    model_config.traverse.assert_called_once_with(Linear.Config)
    assert linear_config.inplace_wgrad_accum is False
    modifier.convert_config.assert_called_once_with(model_config)
