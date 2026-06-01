# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the op-level unit-test suites.

Currently this only hosts the ``real_data`` fixture used by the
``@pytest.mark.real_data`` tests, which compare NVFP4 vs MXFP4 op-level SNR
on *real* GPT-OSS-20B activations (as opposed to the synthetic Gaussian +
sparse-outlier data produced by ``prepare_data``).

The real-activation bundles are produced offline by
``scripts/dump_real_test_activations.py`` and stored on Weka (they are far
too large for git).  Tests that request the ``real_data`` fixture are
automatically skipped when the bundle for the requested tag is absent, so
the default CI run (which has no access to the dumped data) stays green.

Bundle directory resolution order:
  1. ``$ALTO_REAL_DATA_DIR`` if set
  2. ``/wekafs/zhitwang/test_data/real_activations/latest`` (symlink the
     dump script maintains to the most recent run)

Each bundle is a ``torch.save`` dict.  Two schemas exist:

  dense-linear bundle (attention wq/wk/wv/wo):
    {
      "kind": "dense_linear",
      "x":  bf16 [*, K],        # module input (real activation)
      "w":  bf16 [N, K],        # module weight (trained, real)
      "y":  bf16 [*, N],        # bf16 forward reference (F.linear(x, w))
      "dy": bf16 [*, N],        # real upstream grad of the module output
      "metadata": {...},
    }

  grouped-gemm bundle (moe.experts mlp1 GEMM):
    {
      "kind": "grouped_gemm",
      "x":  bf16 [M_total, K],          # routed tokens (real)
      "w":  bf16 [E, N, K],             # mlp1 expert weights (trained, real)
      "num_tokens_per_expert": int32 [E],
      "metadata": {...},
    }
"""

import os
from pathlib import Path

import pytest
import torch


def _real_data_root() -> Path:
    env = os.environ.get("ALTO_REAL_DATA_DIR")
    if env:
        return Path(env)
    return Path("/wekafs/zhitwang/test_data/real_activations/latest")


# Catalog of logical case tags -> bundle filename (relative to the root).
# The dump script writes exactly these names; keep the two in sync.
REAL_DATA_CATALOG = {
    # dense linear, one representative layer at each network depth
    "dense_early": "dense_early/bundle.pt",   # layers.1.attention.wq
    "dense_mid":   "dense_mid/bundle.pt",      # layers.12.attention.wq
    "dense_late":  "dense_late/bundle.pt",     # layers.20.attention.wq
    # grouped experts (mlp1 GEMM) at each depth
    "grouped_early": "grouped_early/bundle.pt",  # layers.1.moe.experts
    "grouped_mid":   "grouped_mid/bundle.pt",    # layers.12.moe.experts
    "grouped_late":  "grouped_late/bundle.pt",   # layers.20.moe.experts
}


@pytest.fixture
def real_data(request):
    """Load a real-activation bundle by tag; skip if it isn't on disk.

    Use via indirect parametrization::

        @pytest.mark.real_data
        @pytest.mark.parametrize("real_data", ["dense_early"], indirect=True)
        def test_x(real_data):
            x, w = real_data["x"], real_data["w"]
    """
    tag = request.param
    if tag not in REAL_DATA_CATALOG:
        raise KeyError(f"unknown real_data tag: {tag!r}")

    if not torch.cuda.is_available():
        pytest.skip("real_data tests require CUDA")

    path = _real_data_root() / REAL_DATA_CATALOG[tag]
    if not path.exists():
        pytest.skip(
            f"real-data bundle missing: {path}. "
            "Run scripts/dump_real_test_activations.py first "
            "(only after training has freed the GPUs)."
        )

    bundle = torch.load(path, map_location="cuda", weights_only=False)
    bundle.setdefault("tag", tag)
    return bundle
