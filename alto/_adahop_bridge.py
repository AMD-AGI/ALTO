# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Bridge to the AdaHOP submodule at ``3rdparty/adahop``.

Loads selected AdaHOP modules by file path under non-colliding names so that
AdaHOP's vendored ``torchtitan`` package never enters ``sys.modules`` and
cannot shadow ALTO's own torchtitan submodule.
"""

import importlib.util
import sys
from pathlib import Path

_ALTO_ROOT = Path(__file__).resolve().parent.parent
_ADAHOP_ROOT = _ALTO_ROOT / "3rdparty" / "adahop"
_HT_DIR = _ADAHOP_ROOT / "torchtitan" / "experiments" / "kernels" / "hadamard_transform"
_TC_PATH = _ADAHOP_ROOT / "torchtitan" / "experiments" / "kernels" / "mxfp4" / "transform_config.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(name: str, pkg_dir: Path):
    init_path = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package {name} from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if not _ADAHOP_ROOT.exists():
    raise ImportError(f"AdaHOP submodule not found at {_ADAHOP_ROOT}. "
                      "Run `git submodule update --init --recursive` from the ALTO root.")

_ht = _load_package("_alto_adahop_ht", _HT_DIR)
_tc = _load_module("_alto_adahop_transform_config", _TC_PATH)

HadamardFactory = _ht.HadamardFactory
HadamardTransform = _ht.HadamardTransform
detect_outlier_pattern = _ht.detect_outlier_pattern

transform_config = _tc
configure_global_transforms = _tc.configure_global_transforms
configure_layer_transforms = _tc.configure_layer_transforms
get_layer_transform_config = _tc.get_layer_transform_config
should_apply_transform = _tc.should_apply_transform
clear_all_configs = _tc.clear_all_configs

__all__ = [
    "HadamardFactory",
    "HadamardTransform",
    "detect_outlier_pattern",
    "transform_config",
    "configure_global_transforms",
    "configure_layer_transforms",
    "get_layer_transform_config",
    "should_apply_transform",
    "clear_all_configs",
]
