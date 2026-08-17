# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for the JSON round-trip schemas in calibration_hooks."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HELPERS_PATH = (Path(__file__).resolve().parents[3] / "alto" / "modifiers" / "lpt" / "adahop_internals" /
                 "calibration_hooks.py")


def _load_helpers():
    name = "alto_adahop_calibration_hooks_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    return _load_helpers()


def test_load_accepts_bare_form(tmp_path, helpers):
    path = tmp_path / "bare.json"
    bare = {"l0": {"forward_y": "hadamard", "backward_gx": "none", "backward_gw": "full_precision"}}
    path.write_text(json.dumps(bare))
    assert helpers.load_modes_from_json(str(path)) == bare


def test_load_accepts_wrapped_form(tmp_path, helpers):
    """``write_modes_json`` output (with ``aggregated_patterns`` +
    ``per_layer_modes``) must be loadable as a modes-only dict."""
    path = tmp_path / "wrapped.json"
    aggregated = {"l0": {"x": "row", "w": "col", "grad_output": "none"}}
    modes = {"l0": {"forward_y": "hadamard", "backward_gx": "none", "backward_gw": "full_precision"}}
    helpers.write_modes_json(str(path), aggregated, modes)
    assert helpers.load_modes_from_json(str(path)) == modes


def test_write_then_load_round_trips(tmp_path, helpers):
    """End-to-end: dump → load yields the same per-layer modes."""
    path = tmp_path / "rt.json"
    aggregated = {
        "layers.0.attention.wq": {
            "x": "row",
            "w": "row",
            "grad_output": "col"
        },
        "layers.1.feed_forward.w1": {
            "x": "none",
            "w": "none",
            "grad_output": "none"
        },
    }
    modes = {
        "layers.0.attention.wq": {
            "forward_y": "hadamard",
            "backward_gx": "hadamard",
            "backward_gw": "hadamard"
        },
        "layers.1.feed_forward.w1": {
            "forward_y": "none",
            "backward_gx": "none",
            "backward_gw": "none"
        },
    }
    helpers.write_modes_json(str(path), aggregated, modes)
    assert helpers.load_modes_from_json(str(path)) == modes
