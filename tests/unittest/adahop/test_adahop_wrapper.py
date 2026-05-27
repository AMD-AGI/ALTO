# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""CPU-side smoke tests for the AdaHOP wrapper subclasses.

These tests construct the wrappers and exercise the flatten / unflatten
round-trip without firing any MXFP4 kernel. Running on a CPU-only box
without triton drivers — full dispatch tests live in cluster integration.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ALTO_ROOT = Path(__file__).resolve().parents[3]


def _import_alto_kernel_free(name: str, path_parts: list[str]):
    """Import a single ALTO module by file path so ``alto/__init__.py`` (which
    transitively loads triton kernels) does not run."""
    path = _ALTO_ROOT.joinpath(*path_parts)
    qualified = f"alto_under_test_{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


pytest.importorskip("triton")  # adahop_tensor imports through alto.kernels.fp4 -> triton

try:
    from alto.kernels.dispatch.adahop_tensor import MXFP4AdaHOPWrapper, MXFP4CalibrationWrapper
    from alto.kernels.dispatch.config import TrainingOpConfig
except RuntimeError as exc:
    # ALTO's kernels init triton at import; on a no-GPU box that raises
    # "0 active drivers". Skip the whole module — these tests need a GPU box.
    pytest.skip(f"alto import requires triton driver: {exc}", allow_module_level=True)


@pytest.fixture
def mxfp4_config():
    return TrainingOpConfig(precision="mxfp4")


def test_calibration_wrapper_accepts_optional_callback(mxfp4_config):
    t = torch.randn(8, 8)
    w = MXFP4CalibrationWrapper(t, mxfp4_config)
    assert w._calibration_callback is None
    seen = []
    w.attach_calibration_callback(lambda x, ww: seen.append((x.shape, ww.shape)))
    assert w._calibration_callback is not None


def test_adahop_wrapper_rejects_unsupported_mode(mxfp4_config):
    t = torch.randn(8, 8)
    with pytest.raises(NotImplementedError):
        MXFP4AdaHOPWrapper(t, mxfp4_config, forward_y_mode="inner_outlier_extract_left")


def test_adahop_wrapper_rejects_garbage_mode(mxfp4_config):
    t = torch.randn(8, 8)
    with pytest.raises(ValueError):
        MXFP4AdaHOPWrapper(t, mxfp4_config, forward_y_mode="not_a_mode")


def test_adahop_wrapper_flatten_unflatten_round_trips_modes(mxfp4_config):
    t = torch.randn(8, 8)
    w = MXFP4AdaHOPWrapper(
        t,
        mxfp4_config,
        hadamard_transform=None,
        forward_y_mode="hadamard",
        backward_gx_mode="outer_hadamard",
        backward_gw_mode="full_precision",
    )
    inner_names, meta = w.__tensor_flatten__()
    assert inner_names == ["_data"]
    assert meta["forward_y_mode"] == "hadamard"
    assert meta["backward_gx_mode"] == "outer_hadamard"
    assert meta["backward_gw_mode"] == "full_precision"

    rebuilt = MXFP4AdaHOPWrapper.__tensor_unflatten__(
        {"_data": t},
        meta,
        outer_size=t.size(),
        outer_stride=t.stride(),
    )
    assert rebuilt._forward_y_mode == "hadamard"
    assert rebuilt._backward_gx_mode == "outer_hadamard"
    assert rebuilt._backward_gw_mode == "full_precision"
    assert rebuilt._hadamard_transform is None  # not serialized; modifier rebinds on load
