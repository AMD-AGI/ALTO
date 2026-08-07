# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Unit tests for moe_pattern_hooks.py — loaded directly to avoid triton imports.

moe_pattern_hooks.py is torch-only (no alto/triton imports) so it can be exec'd
in isolation on a CPU-only box, same pattern as test_observer_hooks.py. The
``detect`` callable is injected, so we use the real AdaHOP algorithm reimplemented
inline here to keep the test self-contained (no adahop submodule needed) AND a
deterministic stub to assert the T-pair combination logic exactly.
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch
import pytest

_HOOKS_PATH = (Path(__file__).resolve().parents[3]
               / "alto" / "modifiers" / "debug" / "moe_pattern_hooks.py")


def _load_hooks():
    name = "_moe_pattern_hooks_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HOOKS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hooks():
    return _load_hooks()


# --- reference detector (verbatim algorithm from AdaHOP outlier_detection.py) ---

def _kurtosis(x):
    mean, std = x.mean(), x.std()
    if std == 0:
        return torch.tensor(0.0)
    z = (x - mean) / std
    return (z ** 4).mean()


def detect_ref(X, threshold_ratio=2.0, kurtosis_threshold=0.0):
    X = X.float()
    if X.dim() > 2:
        X = X.reshape(-1, X.shape[-1])
    row_var = X.var(dim=1)
    col_var = X.var(dim=0)
    cv_row = row_var.std() / (row_var.mean() + 1e-8)
    cv_col = col_var.std() / (col_var.mean() + 1e-8)
    nr, nc = X.shape
    cv_row = cv_row / math.sqrt(2.0 / max(nc - 1, 1))
    cv_col = cv_col / math.sqrt(2.0 / max(nr - 1, 1))
    if cv_row / (cv_col + 1e-8) > threshold_ratio:
        return "row" if _kurtosis(row_var) >= kurtosis_threshold else "none"
    elif cv_col / (cv_row + 1e-8) > threshold_ratio:
        return "col" if _kurtosis(col_var) >= kurtosis_threshold else "none"
    return "none"


# ---------------------------------------------------------------------------
# offs -> bounds
# ---------------------------------------------------------------------------

class TestOffsBounds:
    def test_basic(self, hooks):
        # offs is a cumulative sum; expert 0 = [0,3), expert 1 = [3,7)
        assert hooks._offs_to_bounds([3, 7], 2) == [(0, 3), (3, 7)]

    def test_tail_padding_dropped(self, hooks):
        # trailing padding beyond offs[-1] is never assigned to an expert
        assert hooks._offs_to_bounds([2, 5], 2) == [(0, 2), (2, 5)]

    def test_empty_expert(self, hooks):
        # expert 1 gets zero tokens (offs repeats)
        assert hooks._offs_to_bounds([4, 4], 2) == [(0, 4), (4, 4)]


# ---------------------------------------------------------------------------
# T-pair combination logic (deterministic stub detector)
# ---------------------------------------------------------------------------

class TestTPairLogic:
    def _run(self, hooks, x_pat, w_pat, g_pat):
        # stub detect: return a fixed pattern keyed by tensor identity via shape
        # We tag tensors by a sentinel scalar in [0,0] and map to the desired pat.
        def detect(t):
            tag = int(round(t.reshape(-1)[0].item()))
            return {0: x_pat, 1: w_pat, 2: g_pat}[tag]

        T = 4
        K, N, E = 3, 5, 1
        x = torch.zeros(T, K); x[0, 0] = 0.0          # tag 0
        # w as fed to _grouped_mm is [E, K, N]
        w = torch.zeros(E, K, N); w[0, 0, 0] = 1.0    # tag 1 (any expert slice)
        go = torch.zeros(T, N); go[0, 0] = 2.0        # tag 2
        recs = hooks.build_expert_records(x, w, go, offs=[T], detect=detect)
        return recs[0]

    def test_all_none(self, hooks):
        r = self._run(hooks, "none", "none", "none")
        assert r["forward_y"]["pair"] == "none-none"
        assert r["backward_gx"]["pair"] == "none-none"
        assert r["backward_gw"]["pair"] == "none-none"

    def test_row_row_row(self, hooks):
        r = self._run(hooks, "row", "row", "row")
        # forward_y = x_pat - opposite(w_pat) = row - col
        assert r["forward_y"]["pair"] == "row-col"
        # backward_gx = g_pat - w_pat = row - row
        assert r["backward_gx"]["pair"] == "row-row"
        # backward_gw = opposite(g_pat) - x_pat = col - row
        assert r["backward_gw"]["pair"] == "col-row"

    def test_col_none_row(self, hooks):
        r = self._run(hooks, "col", "none", "row")
        assert r["forward_y"]["pair"] == "col-none"      # col - opposite(none)
        assert r["backward_gx"]["pair"] == "row-none"    # row - none
        assert r["backward_gw"]["pair"] == "col-col"     # opposite(row) - col


# ---------------------------------------------------------------------------
# Per-expert slicing with an injected real outlier
# ---------------------------------------------------------------------------

class TestPerExpertDetection:
    def test_slicing_matches_direct_detect(self, hooks):
        """build_expert_records must classify each expert on exactly its own
        token slice — verified by re-running the reference detector on the
        hand-sliced operands and reconstructing the T-pairs independently."""
        torch.manual_seed(0)
        E, K, N = 2, 8, 6
        t0, t1 = 40, 50
        # expert 0: inject strong per-column outliers into the activation
        x0 = torch.randn(t0, K); x0[:, 2:4] *= 40.0
        x1 = torch.randn(t1, K)
        x = torch.cat([x0, x1], dim=0)
        w = torch.randn(E, K, N)
        go = torch.randn(t0 + t1, N)
        offs = [t0, t0 + t1]

        recs = hooks.build_expert_records(x, w, go, offs, detect_ref)
        assert recs[0]["n_tokens"] == t0
        assert recs[1]["n_tokens"] == t1

        # Independently reconstruct expert 0's forward_y T-pair from its slice.
        opp = {"row": "col", "col": "row", "none": "none"}
        x0_pat = detect_ref(x[0:t0])
        w0_pat = detect_ref(w[0].transpose(-2, -1))
        expected = f"{x0_pat}-{opp[w0_pat]}"
        assert recs[0]["forward_y"]["pair"] == expected
        # the injected column outlier should be picked up on the slice
        assert x0_pat == "col"
        # stats present and finite for a routed expert
        assert recs[0]["stats"]["x"]["absmax"] > 0

    def test_empty_expert_marks_none_activation(self, hooks):
        E, K, N = 2, 4, 3
        x = torch.randn(5, K)
        w = torch.randn(E, K, N)
        go = torch.randn(5, N)
        # expert 1 gets zero tokens
        recs = hooks.build_expert_records(x, w, go, offs=[5, 5], detect=detect_ref)
        assert recs[1]["n_tokens"] == 0
        # activation-derived T1 must be 'none' when no tokens routed
        assert recs[1]["forward_y"]["pair"].startswith("none-")
        assert recs[1]["backward_gw"]["pair"] == "none-none"


# ---------------------------------------------------------------------------
# accumulate_majority
# ---------------------------------------------------------------------------

class TestAccumulateMajority:
    def _rec(self, fy, gx, gw, n_tokens=4):
        return {
            "n_tokens": n_tokens,
            "forward_y": {"pair": fy},
            "backward_gx": {"pair": gx},
            "backward_gw": {"pair": gw},
        }

    def test_clear_majority(self, hooks):
        steps = [
            {0: self._rec("row-col", "col-row", "none-none")},
            {0: self._rec("row-col", "col-row", "row-row")},
            {0: self._rec("none-none", "col-row", "col-col")},
        ]
        out = hooks.accumulate_majority(steps)
        assert out[0]["forward_y"]["pair"] == "row-col"     # 2 of 3
        assert out[0]["backward_gx"]["pair"] == "col-row"   # 3 of 3
        assert out[0]["n_steps"] == 3
        assert out[0]["n_tokens_total"] == 12
        assert out[0]["forward_y"]["votes"] == {"row-col": 2, "none-none": 1}
        assert out[0]["forward_y"]["n"] == 3

    def test_tie_breaks_deterministically(self, hooks):
        # 1 vs 1 tie: winner is the alphabetically-first pair string.
        steps = [
            {0: self._rec("row-col", "none-none", "none-none")},
            {0: self._rec("col-row", "none-none", "none-none")},
        ]
        out = hooks.accumulate_majority(steps)
        assert out[0]["forward_y"]["pair"] == "col-row"

    def test_expert_missing_from_some_steps(self, hooks):
        # Expert 1 is only routed in one of the two steps.
        steps = [
            {0: self._rec("row-col", "col-row", "none-none"),
             1: self._rec("col-col", "row-row", "none-none")},
            {0: self._rec("row-col", "col-row", "none-none")},
        ]
        out = hooks.accumulate_majority(steps)
        assert set(out.keys()) == {0, 1}
        assert out[0]["n_steps"] == 2
        assert out[1]["n_steps"] == 1
        assert out[1]["forward_y"]["pair"] == "col-col"

    def test_empty_input(self, hooks):
        assert hooks.accumulate_majority([]) == {}


# ---------------------------------------------------------------------------
# extract_offs
# ---------------------------------------------------------------------------

class TestExtractOffs:
    def test_kwarg(self, hooks):
        offs = torch.tensor([2, 4], dtype=torch.int32)
        got = hooks.extract_offs((torch.randn(4, 3), torch.randn(1, 3, 5)), {"offs": offs})
        assert torch.equal(got, offs)

    def test_positional_int_tensor(self, hooks):
        offs = torch.tensor([2, 4], dtype=torch.int64)
        got = hooks.extract_offs((torch.randn(4, 3), torch.randn(1, 3, 5), offs), {})
        assert torch.equal(got, offs)

    def test_missing(self, hooks):
        got = hooks.extract_offs((torch.randn(4, 3),), {})
        assert got is None
