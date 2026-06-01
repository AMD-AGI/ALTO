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
from typing import Any

_ALTO_ROOT = Path(__file__).resolve().parent.parent
_ADAHOP_ROOT = _ALTO_ROOT / "3rdparty" / "adahop"
_HT_DIR = _ADAHOP_ROOT / "torchtitan" / "experiments" / "kernels" / "hadamard_transform"
_MXFP4_DIR = _ADAHOP_ROOT / "torchtitan" / "experiments" / "kernels" / "mxfp4"
_TC_PATH = _MXFP4_DIR / "transform_config.py"


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


def _alias_ht_at_absolute_path() -> None:
    """AdaHOP's ``iht_quantization`` does an *absolute* import:
        ``from torchtitan.experiments.kernels.hadamard_transform.hadamard import _build_H_b32``
    ALTO's vendored ``torchtitan`` package does not contain AdaHOP's
    ``hadamard_transform`` subpackage, so the import fails. Alias the
    bridge-loaded hadamard package at the absolute path the file expects.
    Also stub the intermediate ``torchtitan.experiments`` and
    ``torchtitan.experiments.kernels`` packages if they aren't already
    populated, so the dotted lookup resolves.
    """
    import types
    # Build out the chain torchtitan -> .experiments -> .kernels -> .hadamard_transform
    # without clobbering whatever ALTO already has installed.
    if "torchtitan" not in sys.modules:
        sys.modules["torchtitan"] = types.ModuleType("torchtitan")
    if "torchtitan.experiments" not in sys.modules:
        sys.modules["torchtitan.experiments"] = types.ModuleType("torchtitan.experiments")
    if "torchtitan.experiments.kernels" not in sys.modules:
        sys.modules["torchtitan.experiments.kernels"] = types.ModuleType("torchtitan.experiments.kernels")
    sys.modules["torchtitan.experiments.kernels.hadamard_transform"] = _ht
    # The submodule that gets cherry-picked too:
    if hasattr(_ht, "hadamard"):
        sys.modules["torchtitan.experiments.kernels.hadamard_transform.hadamard"] = _ht.hadamard


_alias_ht_at_absolute_path()


def _load_mxfp4_package() -> Any:
    """Load AdaHOP's ``mxfp4/`` package without triggering the colliding
    ``torch.ops.torchtitan.*`` registrations from ``mxfp_linear.py`` and
    ``mxfp_quantization.py``.

    The package's ``__init__.py`` does ``from .mxfp_linear import MXFP4Linear``
    and ``from .mxfp_quantization import convert_to_mxfp4, convert_from_mxfp4,
    BLOCK_SIZE_DEFAULT`` at import time. Both files register ``@triton_op``s
    in the ``torchtitan::`` namespace that ALTO already owns (collision would
    raise ``RuntimeError: Tried to register operator ... twice``). We install
    stubs at the expected ``sys.modules`` names BEFORE exec'ing the package
    so those import lines succeed without touching the real source.

    What we actually need from the package: ``iht_quantization``,
    ``foid.OUTLIER_K``, ``foid.prepare_outlier_clean_row/column``,
    ``outlier_extract.inner_outlier_extract_left_cdna4`` and
    ``..._right_cdna4``. None of these go through the stubbed modules at
    call time — ``outlier_extract`` calls ``torch.ops.torchtitan.``
    ``blockwise_mxfp4_gemm`` (ALTO's registration, cdna-gated) directly,
    and ``iht_quantization`` self-gates cdna3/cdna4 via ``is_cdna4()`` from
    ``fp4_common``.
    """
    pkg_name = "_alto_adahop_mxfp4"
    import types

    # Stub mxfp_linear (only exports MXFP4Linear, referenced by __init__).
    mxfp_linear_stub = types.ModuleType(f"{pkg_name}.mxfp_linear")
    mxfp_linear_stub.MXFP4Linear = object  # placeholder, never instantiated
    sys.modules[f"{pkg_name}.mxfp_linear"] = mxfp_linear_stub

    # Stub mxfp_quantization with the three symbols __init__ imports.
    mxfp_quant_stub = types.ModuleType(f"{pkg_name}.mxfp_quantization")
    mxfp_quant_stub.BLOCK_SIZE_DEFAULT = 32  # matches AdaHOP's real value
    mxfp_quant_stub.convert_to_mxfp4 = None  # never called via this stub
    mxfp_quant_stub.convert_from_mxfp4 = None
    sys.modules[f"{pkg_name}.mxfp_quantization"] = mxfp_quant_stub

    # Stub mxfp_grouped_gemm (only referenced by the lazy mxfp4_grouped_gemm
    # wrapper, never imported eagerly).
    mxfp_grouped_stub = types.ModuleType(f"{pkg_name}.mxfp_grouped_gemm")
    mxfp_grouped_stub.mxfp4_grouped_gemm = None
    sys.modules[f"{pkg_name}.mxfp_grouped_gemm"] = mxfp_grouped_stub

    return _load_package(pkg_name, _MXFP4_DIR)


_mxfp4 = _load_mxfp4_package()

HadamardFactory = _ht.HadamardFactory
HadamardTransform = _ht.HadamardTransform
detect_outlier_pattern = _ht.detect_outlier_pattern

transform_config = _tc
configure_global_transforms = _tc.configure_global_transforms
configure_layer_transforms = _tc.configure_layer_transforms
get_layer_transform_config = _tc.get_layer_transform_config
should_apply_transform = _tc.should_apply_transform
clear_all_configs = _tc.clear_all_configs

iht_quantization = _mxfp4.iht_quantization
OUTLIER_K = _mxfp4.foid.OUTLIER_K
prepare_outlier_clean_row = _mxfp4.foid.prepare_outlier_clean_row
prepare_outlier_clean_column = _mxfp4.foid.prepare_outlier_clean_column
inner_outlier_extract_left_cdna4 = _mxfp4.outlier_extract.inner_outlier_extract_left_cdna4
inner_outlier_extract_right_cdna4 = _mxfp4.outlier_extract.inner_outlier_extract_right_cdna4

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
    "iht_quantization",
    "OUTLIER_K",
    "prepare_outlier_clean_row",
    "prepare_outlier_clean_column",
    "inner_outlier_extract_left_cdna4",
    "inner_outlier_extract_right_cdna4",
]
