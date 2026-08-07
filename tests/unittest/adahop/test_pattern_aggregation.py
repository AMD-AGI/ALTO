# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""CPU-side tests for adahop_internals.pattern_aggregation."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ADAHOP_INTERNALS = Path(__file__).resolve().parents[3] / "alto" / "modifiers" / "lpt" / "adahop_internals"


def _load(name: str):
    """Load a module from adahop_internals by file path so this test can run
    on CPU-only boxes where ``import alto`` triggers triton driver init."""
    path = _ADAHOP_INTERNALS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"alto_adahop_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    # Some adahop_internals files use relative imports (`from .transform_mode import ...`).
    # Register the package + sibling so relative resolution works.
    pkg_name = "alto_adahop_internals_under_test"
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name,
            _ADAHOP_INTERNALS / "__init__.py",
            submodule_search_locations=[str(_ADAHOP_INTERNALS)],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg
        pkg_spec.loader.exec_module(pkg)
    # Re-create spec under the package
    qualified = f"{pkg_name}.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aggregation():
    return _load("pattern_aggregation")


@pytest.fixture(scope="module")
def transform_mode():
    return _load("transform_mode")


def test_aggregate_picks_majority_per_tensor(aggregation):
    per_step = [
        {
            "layers.0.wq": {
                "x": "row",
                "w": "col",
                "grad_output": "none"
            }
        },
        {
            "layers.0.wq": {
                "x": "row",
                "w": "col",
                "grad_output": "row"
            }
        },
        {
            "layers.0.wq": {
                "x": "col",
                "w": "col",
                "grad_output": "row"
            }
        },
    ]
    out = aggregation.aggregate_patterns(per_step)
    assert out == {"layers.0.wq": {"x": "row", "w": "col", "grad_output": "row"}}


def test_aggregate_missing_layers_and_tensors_default_to_none(aggregation):
    per_step = [
        {
            "layers.0.wq": {
                "x": "row"
            }
        },
        {
            "layers.1.wk": {
                "w": "col"
            }
        },
    ]
    out = aggregation.aggregate_patterns(per_step)
    assert out["layers.0.wq"] == {"x": "row", "w": "none", "grad_output": "none"}
    assert out["layers.1.wk"] == {"x": "none", "w": "col", "grad_output": "none"}


def test_patterns_to_modes_t1_t2_mapping(aggregation):
    # x=row, w=col → T1=row, T2=opposite(col)=row → "row-row"
    aggregated = {"l0": {"x": "row", "w": "col", "grad_output": "none"}}
    cfg = {"row-row": "hadamard", "none-none": "none"}
    out = aggregation.patterns_to_modes(aggregated, cfg)
    assert out["l0"]["forward_y"] == "hadamard"


def test_patterns_to_modes_unknown_pair_falls_back_to_none(aggregation):
    aggregated = {"l0": {"x": "row", "w": "row", "grad_output": "row"}}
    cfg = {"row-row": "hadamard"}
    # T4 = opposite(row) = col, T5 = row → "col-row" not in cfg → "none"
    out = aggregation.patterns_to_modes(aggregated, cfg)
    assert out["l0"]["backward_gw"] == "none"


def test_assert_mode_supported_accepts_basic_four(transform_mode):
    for mode in ["none", "hadamard", "outer_hadamard", "full_precision"]:
        transform_mode.assert_mode_supported(mode, "forward_y")


def test_assert_mode_supported_rejects_unported_mode(transform_mode):
    with pytest.raises(NotImplementedError):
        transform_mode.assert_mode_supported("inner_outlier_extract_left", "forward_y")


def test_assert_mode_supported_rejects_garbage(transform_mode):
    with pytest.raises(ValueError):
        transform_mode.assert_mode_supported("not_a_mode", "backward_gx")
