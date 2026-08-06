# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Triton env + sibling-import shim for the AMD-FP4 test suite.

This suite covers both layers of AMD-FP4:

* Low-level UE5M3 dtype primitives (``test_amdfp_dtype.py``,
  ``test_amdfp_triton_pytorch_parity.py``) — formerly under
  ``tests/unittest/ue5m3/``.
* Recipe-level quantize / dequantize / linear / grouped-GEMM ops
  pinned to the AMD-FP4 recipe (``test_amdfp_quantization.py``,
  ``test_amdfp_linear.py``, ``test_amdfp_grouped_gemm.py``,
  ``test_ab_matrix.py``).

The Triton path needs ``TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`` to read
the module-level ``_quantize_ue5m3`` / ``_quantize_e4m3`` JIT helpers,
and a per-suite cache dir so this suite does not collide with other
Triton suites in the repo.
"""

import os
import sys

import pytest

# Sibling import: ``from nvfp4.utils import ...`` lets the AMD-FP4 oracle
# delegate to the NVFP4 PyTorch reference (recipe-level tests share the
# same kernel under different ``scale_format``).
_UNITTEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _UNITTEST_DIR not in sys.path:
    sys.path.insert(0, _UNITTEST_DIR)


@pytest.fixture(autouse=True)
def _configure_triton_env(monkeypatch):
    monkeypatch.setenv("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton-cache-amdfp4-tests")
