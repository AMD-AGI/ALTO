# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Verify ``alto._adahop_bridge`` loads AdaHOP modules without shadowing
ALTO's own ``torchtitan`` submodule."""

import importlib.util
import sys
from pathlib import Path

import pytest

BRIDGE_PATH = Path(__file__).resolve().parents[3] / "alto" / "_adahop_bridge.py"


def _load_bridge_isolated():
    """Load the bridge by file path so this test can run on CPU-only boxes
    where ``import alto`` fails on triton driver initialization."""
    spec = importlib.util.spec_from_file_location("alto_adahop_bridge_under_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge():
    pre = sys.modules.get("torchtitan")
    module = _load_bridge_isolated()
    yield module, pre


def test_bridge_does_not_pollute_torchtitan_namespace(bridge):
    _, pre = bridge
    post = sys.modules.get("torchtitan")
    assert post is pre, ("Bridge inserted or replaced sys.modules['torchtitan']; "
                         "AdaHOP's torchtitan would shadow ALTO's submodule.")


def test_bridge_exports_callables(bridge):
    module, _ = bridge
    for name in [
            "HadamardFactory",
            "HadamardTransform",
            "detect_outlier_pattern",
            "configure_global_transforms",
            "configure_layer_transforms",
            "get_layer_transform_config",
            "should_apply_transform",
            "clear_all_configs",
    ]:
        assert hasattr(module, name), f"bridge missing export: {name}"


def test_transform_config_registry_roundtrip(bridge):
    module, _ = bridge
    module.clear_all_configs()
    module.configure_layer_transforms({
        "layers.0.attention.wq": {
            "forward_y": "hadamard",
            "backward_gw": "none",
            "backward_gx": "none",
        },
    })
    assert module.get_layer_transform_config("layers.0.attention.wq", "forward_y") == "hadamard"
    assert module.should_apply_transform("layers.0.attention.wq", "forward_y") is True
    assert module.should_apply_transform("layers.0.attention.wq", "backward_gw") is False
    assert module.get_layer_transform_config("unknown.layer", "forward_y") == "none"
    module.clear_all_configs()
