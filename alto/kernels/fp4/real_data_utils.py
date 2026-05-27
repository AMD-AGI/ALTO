# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Shared catalog + loader for real-model activation bundles.

This module is the single source of truth for the **real-data op-level SNR**
workflow, used by both:

* ``scripts/dump_real_test_activations.py`` -- the one-shot capture script
  that runs GPT-OSS-20B forward and saves ``{x, w, y, dy_*}`` bundles to
  a shared filesystem.
* ``tests/unittest/{nvfp4,mxfp4}/conftest.py`` -- pytest fixtures that
  reload those bundles and feed them to NVFP4 / MXFP4 op tests so the two
  formats can be compared on the **same real activation distribution**
  against a BF16 reference.

Keeping the schema (layer-tag -> module-path, bundle keys, manifest fields,
``SCHEMA_VERSION``) here means the dump script and the consumer tests are
guaranteed to agree on the on-disk format -- a mismatch raises an explicit
``RealDataSchemaError`` instead of silently producing meaningless SNR.

Why this lives under ``alto.kernels.fp4`` rather than ``tests/unittest``
=======================================================================
* It is consumed by *both* nvfp4 and mxfp4 test packages, which do not
  share a common Python package today.
* It only depends on ``torch`` and on shape conventions of the fp4 ops
  themselves, so it belongs next to ``testing_utils.py`` rather than in
  an ad-hoc test-only helper directory.
* Source code is shipped with the wheel, so any downstream consumer that
  wants to reuse the same bundles in their own tests can ``import`` it.

The module deliberately has no GPU or model-loading dependencies; the
dump script handles those.  This file just describes the format and
provides a safe loader.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "SCHEMA_VERSION",
    "REAL_ROOT_ENV",
    "DEFAULT_REAL_ROOT",
    "LAYER_TAG_TO_PATH",
    "DENSE_TAGS",
    "GROUPED_TAGS",
    "ALL_TAGS",
    "BundleKind",
    "RealDataSchemaError",
    "bundle_filename",
    "manifest_path",
    "load_manifest",
    "load_real_bundle",
]


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Bumped whenever the on-disk bundle / manifest format changes in a way
#: that breaks older readers.  Loader rejects mismatching bundles.
SCHEMA_VERSION = 1

#: Environment variable consumers use to override the bundle root.  Lets
#: a single test run target either the canonical weka path or a local
#: scratch copy without code edits.
REAL_ROOT_ENV = "ALTO_REAL_DATA_ROOT"

#: Default bundle root if ``$ALTO_REAL_DATA_ROOT`` is unset.  Matches the
#: layout the dump script writes to on the GPT-OSS-20B training pod.
DEFAULT_REAL_ROOT = (
    "/wekafs/zhitwang/test_data/real_activations/step5000_ckpt_20260526"
)


# ---------------------------------------------------------------------------
# Layer catalog
# ---------------------------------------------------------------------------
#
# Six representative layers, three dense + three grouped, spanning early /
# mid / late depth.  Picked to match the existing op-level test scope
# (NVFP4 / MXFP4 linear + grouped-GEMM) so each bundle can be fed directly
# to ``_to_{nv,mx}fp4_then_scaled_mm`` / ``{nv,mx}fp4_grouped_gemm``.
#
# Dense target is ``attention.wq`` -- the largest dense projection in
# GPT-OSS attention (``dim -> n_heads*head_dim``).  Using ``wq`` rather
# than ``wo`` keeps the input distribution apples-to-apples with the
# existing synthetic linear tests (post-norm activations, pre-attention).
#
# Grouped target is ``moe.experts`` -- the whole ``GptOssGroupedExperts``
# module.  We capture its input ``(x, num_tokens_per_expert)`` and the two
# ``_grouped_mm`` weight parameters (``mlp1_weight``, ``mlp2_weight``).
# The test re-runs each ``_grouped_mm`` independently in BF16 vs FP4 so
# SwiGLU + bias do not pollute the per-op SNR signal.

LAYER_TAG_TO_PATH: dict[str, str] = {
    "dense_early":   "layers.0.attention.wq",
    "dense_mid":     "layers.12.attention.wq",
    "dense_late":    "layers.22.attention.wq",
    "grouped_early": "layers.5.moe.experts",
    "grouped_mid":   "layers.15.moe.experts",
    "grouped_late":  "layers.20.moe.experts",
}

DENSE_TAGS = ("dense_early", "dense_mid", "dense_late")
GROUPED_TAGS = ("grouped_early", "grouped_mid", "grouped_late")
ALL_TAGS = DENSE_TAGS + GROUPED_TAGS


# ---------------------------------------------------------------------------
# Bundle schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BundleKind:
    """Describes which keys are present in a saved bundle.

    Dense bundles are saved by :class:`torch.nn.Linear` hooks on
    ``attention.wq``; grouped bundles are saved by hooks on
    ``GptOssGroupedExperts``.  Both kinds share the synthetic-dy fields
    (``dy_normal_std_y``, ``dy_ones``); grouped bundles additionally
    carry the routing tensor and the second-mlp weight.

    A bundle's ``kind`` is derived from its layer tag; consumers should
    never need to introspect tensor shapes to disambiguate.
    """

    name: str  # "dense" | "grouped"

    @property
    def is_dense(self) -> bool:
        return self.name == "dense"

    @property
    def is_grouped(self) -> bool:
        return self.name == "grouped"


def kind_for_tag(tag: str) -> BundleKind:
    if tag in DENSE_TAGS:
        return BundleKind("dense")
    if tag in GROUPED_TAGS:
        return BundleKind("grouped")
    raise KeyError(f"unknown layer tag: {tag!r}; known tags: {ALL_TAGS}")


# Bundle dict keys.  Keep these as module-level constants so a typo on
# either the dump or the test side fails fast at import time.
KEY_X = "x"
KEY_W = "w"                 # dense only -- module.weight
KEY_MLP1_W = "mlp1_weight"  # grouped only
KEY_MLP2_W = "mlp2_weight"  # grouped only
KEY_NUM_TOKENS_PER_EXPERT = "num_tokens_per_expert"  # grouped only
KEY_Y = "y"
# Synthetic gradient strategies -- see scripts/dump_real_test_activations.py
# for how each is computed.  Real BF16 ``dy`` requires full backward and
# lives behind ``--with-backward`` (PM-firmware-bug-permitting); when
# present it appears under ``dy_real``.
KEY_DY_REAL = "dy_real"
KEY_DY_NORMAL_STD_Y = "dy_normal_std_y"
KEY_DY_ONES = "dy_ones"
KEY_METADATA = "metadata"


class RealDataSchemaError(RuntimeError):
    """Raised when a bundle / manifest does not match the expected schema.

    Loaders re-raise as ``pytest.skip`` at the call site rather than
    failing the test, so a stale or partial dump does not turn a
    pytest run red -- it just skips the real-data leg.
    """


# ---------------------------------------------------------------------------
# Path / filename helpers
# ---------------------------------------------------------------------------

def bundle_filename(bs: int, seq: int, step: int) -> str:
    """Canonical bundle filename for a given (batch, seq, capture step).

    All three pieces are forced into the name so the test side does not
    have to grep manifest entries to know which file maps to which shape.
    """
    return f"bs{bs}_seq{seq}_step{step}.pt"


def manifest_path(root: Path | str) -> Path:
    return Path(root) / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest loader + validator
# ---------------------------------------------------------------------------

def load_manifest(root: Path | str) -> dict[str, Any]:
    """Load and minimally validate ``manifest.json`` under *root*.

    Returns the parsed JSON dict on success; raises
    :class:`RealDataSchemaError` when the file is missing, malformed, or
    written with an incompatible ``schema_version``.  Test callers should
    catch this and convert to ``pytest.skip``.
    """
    mpath = manifest_path(root)
    if not mpath.exists():
        raise RealDataSchemaError(
            f"manifest not found: {mpath}; run scripts/dump_real_test_activations.py "
            f"to populate {root}"
        )
    try:
        data = json.loads(mpath.read_text())
    except json.JSONDecodeError as e:
        raise RealDataSchemaError(f"manifest at {mpath} is not valid JSON: {e}") from e

    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise RealDataSchemaError(
            f"manifest schema mismatch at {mpath}: have schema_version={sv!r}, "
            f"this build expects {SCHEMA_VERSION}; re-run the dump script with "
            f"the current alto checkout"
        )
    for required in ("run_tag", "model_sha", "captured_at", "layers"):
        if required not in data:
            raise RealDataSchemaError(
                f"manifest at {mpath} missing required field {required!r}"
            )
    return data


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def resolve_root() -> Path:
    """Return the root directory to load bundles from.

    Priority: ``$ALTO_REAL_DATA_ROOT`` if set, else :data:`DEFAULT_REAL_ROOT`.
    """
    return Path(os.environ.get(REAL_ROOT_ENV, DEFAULT_REAL_ROOT))


def load_real_bundle(
    tag: str,
    *,
    bs: int = 2,
    seq: int = 2048,
    step: int = 0,
    map_location: str | torch.device = "cuda",
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Load one real-data bundle from disk.

    Validates that the bundle file exists, that the parent ``manifest.json``
    is schema-compatible, and that the bundle's required keys are present
    (raises :class:`RealDataSchemaError` otherwise so the test layer can
    skip cleanly).

    The returned dict is the raw bundle saved by the dump script -- callers
    are responsible for moving / casting tensors and for selecting which
    ``dy_*`` strategy to use.
    """
    root = Path(root) if root is not None else resolve_root()
    manifest = load_manifest(root)  # also validates schema_version / fields

    if tag not in LAYER_TAG_TO_PATH:
        raise RealDataSchemaError(
            f"unknown layer tag {tag!r}; known tags: {ALL_TAGS}"
        )

    fname = bundle_filename(bs, seq, step)
    bundle_path = root / tag / fname
    if not bundle_path.exists():
        raise RealDataSchemaError(
            f"bundle missing: {bundle_path}; re-run the dump script with "
            f"--layer-tags {tag} --bs-seq {bs}x{seq} --num-steps {step + 1} (at minimum)"
        )

    bundle = torch.load(bundle_path, map_location=map_location)

    kind = kind_for_tag(tag)
    required = [KEY_X, KEY_Y, KEY_METADATA]
    if kind.is_dense:
        required.append(KEY_W)
    else:
        required += [KEY_MLP1_W, KEY_MLP2_W, KEY_NUM_TOKENS_PER_EXPERT]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise RealDataSchemaError(
            f"bundle {bundle_path} missing required keys for kind={kind.name}: "
            f"{missing}; got keys={list(bundle.keys())}"
        )

    # Annotate the bundle so downstream consumers do not have to re-derive.
    bundle.setdefault("_loader", {})
    bundle["_loader"].update({
        "root": str(root),
        "tag": tag,
        "kind": kind.name,
        "bs": bs,
        "seq": seq,
        "step": step,
        "manifest_run_tag": manifest.get("run_tag"),
        "manifest_model_sha": manifest.get("model_sha"),
    })
    return bundle
