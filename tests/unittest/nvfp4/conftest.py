# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Pytest fixtures shared by the NVFP4 op-level test suite.

The only fixture defined here today is ``real_data``, which loads a
pre-captured GPT-OSS-20B activation bundle from the shared filesystem
and feeds it to tests marked with ``@pytest.mark.real_data``.

We intentionally keep the fixture *thin* and *lazy*: the actual schema
/ catalog lives in :mod:`alto.kernels.fp4.real_data_utils` so the
matching MXFP4 conftest can re-use it without copy-pasting.  Missing
bundles or schema mismatches translate into ``pytest.skip`` rather
than test failure -- this lets CI machines (which do not mount weka)
run the rest of the NVFP4 suite without spurious red runs, and it
lets contributors run a partial dump (only one layer / shape) while
iterating without breaking unrelated tests.
"""

from __future__ import annotations

import pytest

from alto.kernels.fp4.real_data_utils import (
    RealDataSchemaError,
    load_real_bundle,
)


@pytest.fixture
def real_data(request):
    """Load one real-data bundle.

    Use as ``@pytest.mark.parametrize("real_data", [...], indirect=True)``.
    Accepted parameter shapes:

    * ``str``: a layer tag (e.g. ``"dense_early"``); defaults to
      ``bs=2, seq=2048, step=0``.
    * ``tuple``: ``(tag, bs, seq)`` or ``(tag, bs, seq, step)`` for
      explicit shape / capture-step selection.

    The fixture **skips** rather than fails when the bundle or the
    parent manifest is missing or schema-incompatible, so the test
    file's import path stays usable on machines without the weka mount.
    """
    param = request.param
    if isinstance(param, str):
        tag, bs, seq, step = param, 2, 2048, 0
    elif len(param) == 3:
        tag, bs, seq = param
        step = 0
    elif len(param) == 4:
        tag, bs, seq, step = param
    else:
        raise ValueError(
            f"real_data fixture: bad param {param!r}; expected str or "
            f"(tag, bs, seq[, step])"
        )

    try:
        return load_real_bundle(tag, bs=bs, seq=seq, step=step)
    except RealDataSchemaError as e:
        pytest.skip(str(e))
