# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""CPU-side tests for AdaHOPModifier's pure-Python helpers.

The full modifier needs ALTO's triton-backed kernels at import time, but the
helper functions live in a sibling module (``calibration_hooks``) with
stdlib-only deps so they can be tested standalone."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


class _Det:
    """Stand-in tensor that echoes a string from .detach()."""

    def __init__(self, v):
        self.v = v

    def detach(self):
        return self.v


def test_forward_pre_hook_writes_x_and_w_patterns(helpers):
    # module.weight.data._data is the underlying tensor the hook inspects.
    weight = SimpleNamespace(data=SimpleNamespace(_data=_Det("col")))
    module = SimpleNamespace(weight=weight)
    modifier = SimpleNamespace(_per_step_patterns=[{}])

    hook = helpers.make_forward_pre_hook(modifier, "layers.0.wq", lambda t: t)
    hook(module, (_Det("row"),))
    assert modifier._per_step_patterns[-1]["layers.0.wq"] == {"x": "row", "w": "col"}


def test_forward_pre_hook_noop_when_no_step_dict(helpers):
    modifier = SimpleNamespace(_per_step_patterns=[])
    hook = helpers.make_forward_pre_hook(modifier, "layers.0.wq", lambda t: t)
    hook(object(), (_Det("row"),))  # early-returns before touching module
    assert modifier._per_step_patterns == []


def test_backward_hook_writes_grad_output_pattern(helpers):

    class FakeGrad:

        def detach(self):
            return "row"

    modifier = SimpleNamespace(_per_step_patterns=[{}])
    hook = helpers.make_backward_hook(modifier, "layers.0.wq", lambda t: t)
    hook(None, None, (FakeGrad(),))
    assert modifier._per_step_patterns[-1]["layers.0.wq"]["grad_output"] == "row"


def test_backward_hook_ignores_none_grad(helpers):
    modifier = SimpleNamespace(_per_step_patterns=[{}])
    hook = helpers.make_backward_hook(modifier, "l0", lambda t: t)
    hook(None, None, (None,))
    assert modifier._per_step_patterns[-1] == {}


def test_backward_hook_handles_bare_tensor(helpers):

    class FakeGrad:

        def detach(self):
            return "col"

    modifier = SimpleNamespace(_per_step_patterns=[{}])
    hook = helpers.make_backward_hook(modifier, "l0", lambda t: t)
    hook(None, None, FakeGrad())  # not a tuple
    assert modifier._per_step_patterns[-1]["l0"]["grad_output"] == "col"


def test_write_and_load_modes_json(tmp_path, helpers):
    path = tmp_path / "modes.json"
    aggregated = {"l0": {"x": "row", "w": "col", "grad_output": "none"}}
    modes = {"l0": {"forward_y": "hadamard", "backward_gx": "none", "backward_gw": "full_precision"}}
    helpers.write_modes_json(str(path), aggregated, modes)

    raw = json.loads(path.read_text())
    assert raw["aggregated_patterns"] == aggregated
    assert raw["per_layer_modes"] == modes

    # load_modes_from_json expects the bare {fqn: {slot: mode}} format used
    # by the transform_config_path recipe option.
    modes_only = tmp_path / "modes_only.json"
    modes_only.write_text(json.dumps(modes))
    assert helpers.load_modes_from_json(str(modes_only)) == modes
