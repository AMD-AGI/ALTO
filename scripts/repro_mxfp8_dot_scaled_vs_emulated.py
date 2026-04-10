#!/usr/bin/env python3

import os
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELOPTIMIZER_DIR = REPO_ROOT / "modeloptimizer"
KERNELS_DIR = MODELOPTIMIZER_DIR / "kernels"
MXFP8_DIR = KERNELS_DIR / "mxfp8"
TESTS_DIR = MXFP8_DIR / "tests"

SHAPE = (64, 64, 64)
MXFP_FORMAT = "e4m3"
OUTPUT_DTYPE = torch.float32


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__file__ = str(path / "__init__.py")
    module.__path__ = [str(path)]
    return module


def _bootstrap_test_imports() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    modeloptimizer_pkg = _ensure_package("modeloptimizer", MODELOPTIMIZER_DIR)
    kernels_pkg = _ensure_package("modeloptimizer.kernels", KERNELS_DIR)
    mxfp8_pkg = _ensure_package("modeloptimizer.kernels.mxfp8", MXFP8_DIR)
    tests_pkg = _ensure_package("modeloptimizer.kernels.mxfp8.tests", TESTS_DIR)

    modeloptimizer_pkg.kernels = kernels_pkg
    kernels_pkg.mxfp8 = mxfp8_pkg
    mxfp8_pkg.tests = tests_pkg


_bootstrap_test_imports()

from modeloptimizer.kernels.mxfp8.mxfp8_linear import blockwise_mxfp8_gemm
from modeloptimizer.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    convert_from_mxfp8,
    convert_to_mxfp8,
    is_cdna4,
)
from modeloptimizer.kernels.mxfp8.tests.utils import prepare_data


def _build_inputs():
    m, n, k = SHAPE
    a = prepare_data((m, k), OUTPUT_DTYPE)
    b = prepare_data((k, n), OUTPUT_DTYPE)
    a_lp, a_scales = convert_to_mxfp8(
        a,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format=MXFP_FORMAT,
        axis=-1,
        is_2d_block=False,
    )
    b_lp, b_scales = convert_to_mxfp8(
        b,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format=MXFP_FORMAT,
        axis=-2,
        is_2d_block=False,
    )
    return a_lp, a_scales, b_lp, b_scales


def _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated: bool):
    if use_emulated:
        os.environ["MODELOPTIMIZER_MXFP8_EMULATED_DOT_FORMATS"] = MXFP_FORMAT
    else:
        os.environ.pop("MODELOPTIMIZER_MXFP8_EMULATED_DOT_FORMATS", None)

    return blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=False,
        trans_b=False,
        block_size=BLOCK_SIZE_DEFAULT,
        output_dtype=OUTPUT_DTYPE,
        use_accumulator_add=True,
    )


def _build_reference(a_lp, a_scales, b_lp, b_scales):
    a_dq = convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=OUTPUT_DTYPE,
        block_size=BLOCK_SIZE_DEFAULT,
        axis=-1,
        is_2d_block=False,
    )
    b_dq = convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=OUTPUT_DTYPE,
        block_size=BLOCK_SIZE_DEFAULT,
        axis=-2,
        is_2d_block=False,
    )
    return a_dq @ b_dq


def _max_diff(lhs: torch.Tensor, rhs: torch.Tensor):
    diff = (lhs - rhs).abs().float()
    flat_idx = int(diff.argmax().item())
    row, col = divmod(flat_idx, diff.shape[1])
    return diff[row, col].item(), (row, col)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")
    if not is_cdna4():
        raise RuntimeError("This repro expects a CDNA4 gfx950 target.")

    a_lp, a_scales, b_lp, b_scales = _build_inputs()
    c_scaled = _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated=False)
    c_emulated = _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated=True)
    c_reference = _build_reference(a_lp, a_scales, b_lp, b_scales)

    scaled_vs_emulated, idx = _max_diff(c_scaled, c_emulated)
    scaled_vs_reference, _ = _max_diff(c_scaled, c_reference)
    emulated_vs_reference, _ = _max_diff(c_emulated, c_reference)
    row, col = idx

    print(f"shape={SHAPE} format={MXFP_FORMAT} output_dtype={OUTPUT_DTYPE}")
    print(f"max_abs_diff(scaled, emulated)={scaled_vs_emulated}")
    print(f"max_abs_diff(scaled, reference)={scaled_vs_reference}")
    print(f"max_abs_diff(emulated, reference)={emulated_vs_reference}")
    print(f"mismatch_index={(row, col)}")
    print(f"scaled_value={float(c_scaled[row, col])}")
    print(f"emulated_value={float(c_emulated[row, col])}")
    print(f"reference_value={float(c_reference[row, col])}")

    return 1 if scaled_vs_emulated > 0.0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
