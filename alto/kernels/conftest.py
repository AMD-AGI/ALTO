# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest


_KERNELS_TEST_ROOT = Path(__file__).parent

_TRITON_TEST_ENV_BY_SUITE = {
    ("fp4", "mxfp4", "tests"): {
        "TRITON_CACHE_DIR": "/tmp/triton-cache-mxfp4-tests",
    },
    ("fp4", "nvfp4", "tests"): {
        "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS": "1",
        "TRITON_CACHE_DIR": "/tmp/triton-cache-nvfp4-tests",
    },
    ("mxfp8", "tests"): {
        "TRITON_CACHE_DIR": "/tmp/triton-cache-mxfp8-tests",
    },
}


def _resolve_triton_test_env(test_path: Path) -> dict[str, str]:
    relative_parts = test_path.relative_to(_KERNELS_TEST_ROOT).parts
    for suite_parts, env_vars in _TRITON_TEST_ENV_BY_SUITE.items():
        if relative_parts[:len(suite_parts)] == suite_parts:
            return env_vars
    return {}


@pytest.fixture(autouse=True)
def _configure_kernel_test_triton_environment(monkeypatch, request):
    test_path = Path(str(request.node.path))
    for env_name, env_value in _resolve_triton_test_env(test_path).items():
        monkeypatch.setenv(env_name, env_value)
