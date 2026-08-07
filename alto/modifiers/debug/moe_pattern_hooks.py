# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Pure-Python helpers for ``MoEMatmulPatternObserverModifier``.

torch-only (no alto / triton imports) so it can be exec'd in isolation on a
CPU-only box, following the same testing pattern as ``observer_hooks.py``.

A grouped-GEMM expert layer runs two ``torch._grouped_mm`` calls (MLP1 then
MLP2). Each grouped GEMM's forward+backward is three matmuls whose inputs are
drawn from ``{x, w, grad_output}``:

    forward_y   : x @ w                (operands x, w)
    backward_gx : grad_output @ wᵀ     (operands grad_output, w)
    backward_gw : grad_outputᵀ @ x     (operands grad_output, x)

For each expert we detect the AdaHOP outlier pattern (row/col/none) of each
operand and combine them into the AdaHOP "T-pair" per matmul path, matching
``alto/modifiers/lpt/adahop_internals/pattern_aggregation.py``.

Weight-orientation note: the model calls ``_grouped_mm(x, W.transpose(-2,-1))``
so the ``w`` operand reaching the patched op is ``[E, K, N]`` (in, out). AdaHOP
classifies weights in their ``[out, in]`` orientation, so we transpose each
expert slice back to ``[N, K]`` before ``detect`` — keeping the T-pairs directly
comparable to AdaHOP calibration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import torch

# The three matmul paths and which operands feed each (for stats bookkeeping).
MATMUL_PATHS = ("forward_y", "backward_gx", "backward_gw")


def _opposite_pattern(pattern: str) -> str:
    """row<->col, none->none. Mirrors pattern_aggregation._opposite_pattern."""
    if pattern == "row":
        return "col"
    if pattern == "col":
        return "row"
    return "none"


def _to_2d_float(t: torch.Tensor) -> torch.Tensor:
    """Detach, upcast to float32 (bf16 var/std are imprecise), flatten to 2D."""
    t = t.detach().float()
    if t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])
    return t


def _cv_row_col(t2d: torch.Tensor) -> tuple[float, float]:
    """Shape-normalized coefficient of variation of row/col variances — the
    quantity ``detect_outlier_pattern`` thresholds on. Returned for insight."""
    import math

    if t2d.numel() == 0 or t2d.dim() != 2:
        return float("nan"), float("nan")
    num_rows, num_cols = t2d.shape
    row_var = t2d.var(dim=1)
    col_var = t2d.var(dim=0)
    cv_row = (row_var.std() / (row_var.mean() + 1e-8)).item()
    cv_col = (col_var.std() / (col_var.mean() + 1e-8)).item()
    cv_row /= math.sqrt(2.0 / max(num_cols - 1, 1))
    cv_col /= math.sqrt(2.0 / max(num_rows - 1, 1))
    return cv_row, cv_col


def _operand_stats(t2d: torch.Tensor) -> Dict[str, float]:
    """Lightweight per-operand summary — plain Python floats only."""
    if t2d.numel() == 0:
        return {"absmax": float("nan"), "std": float("nan"),
                "cv_row": float("nan"), "cv_col": float("nan")}
    cv_row, cv_col = _cv_row_col(t2d)
    return {
        "absmax": t2d.abs().max().item(),
        "std": t2d.std().item(),
        "cv_row": cv_row,
        "cv_col": cv_col,
    }


def _offs_to_bounds(offs: List[int], num_experts: int) -> List[tuple[int, int]]:
    """Turn a cumulative-sum offsets list into [start, end) row bounds per
    expert. offs[i] is the running token count through expert i; tokens beyond
    offs[num_experts-1] are grouped-GEMM tail padding and are dropped."""
    bounds = []
    prev = 0
    for e in range(num_experts):
        end = int(offs[e]) if e < len(offs) else prev
        bounds.append((prev, end))
        prev = end
    return bounds


def build_expert_records(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_output: torch.Tensor,
    offs: List[int],
    detect: Callable[[torch.Tensor], str],
) -> Dict[int, Dict[str, Any]]:
    """Per (local) expert, detect operand patterns and build the T-pair records.

    Args:
        x:           activation into this GEMM, ``[T, K]`` (as fed to the matmul).
        w:           weight operand as fed to ``_grouped_mm``, ``[E, K, N]``.
        grad_output: gradient of this GEMM's output, ``[T, N]``.
        offs:        cumulative per-expert token counts (Python ints).
        detect:      ``detect_outlier_pattern``-style callable returning
                     ``"row"|"col"|"none"`` for a 2D tensor.

    Returns ``{local_expert_id: {"n_tokens": int,
        "forward_y":{"pair":str}, "backward_gx":{"pair":str},
        "backward_gw":{"pair":str},
        "stats":{"x":{...},"w":{...},"grad_output":{...}}}}``.
    """
    x2 = _to_2d_float(x)
    go2 = _to_2d_float(grad_output)
    num_experts = int(w.shape[0])
    bounds = _offs_to_bounds(offs, num_experts)

    records: Dict[int, Dict[str, Any]] = {}
    for e in range(num_experts):
        start, end = bounds[e]
        n_tokens = max(0, end - start)
        # weight back to [out, in] = [N, K] to match AdaHOP's convention.
        w_e = _to_2d_float(w[e].transpose(-2, -1))

        if n_tokens == 0:
            # No tokens routed to this expert this step: activation/grad patterns
            # are undefined. Weight pattern is still meaningful.
            w_pat = detect(w_e)
            records[e] = {
                "n_tokens": 0,
                "forward_y": {"pair": f"none-{_opposite_pattern(w_pat)}"},
                "backward_gx": {"pair": f"none-{w_pat}"},
                "backward_gw": {"pair": "none-none"},
                "stats": {"x": _operand_stats(x2[0:0]),
                          "w": _operand_stats(w_e),
                          "grad_output": _operand_stats(go2[0:0])},
            }
            continue

        x_e = x2[start:end]
        go_e = go2[start:end]
        x_pat = detect(x_e)
        w_pat = detect(w_e)
        g_pat = detect(go_e)

        records[e] = {
            "n_tokens": n_tokens,
            # forward_y : T1=x_pat, T2=opposite(w_pat)
            "forward_y": {"pair": f"{x_pat}-{_opposite_pattern(w_pat)}"},
            # backward_gx : T7=grad_pat, T8=w_pat
            "backward_gx": {"pair": f"{g_pat}-{w_pat}"},
            # backward_gw : T4=opposite(grad_pat), T5=x_pat
            "backward_gw": {"pair": f"{_opposite_pattern(g_pat)}-{x_pat}"},
            "stats": {"x": _operand_stats(x_e),
                      "w": _operand_stats(w_e),
                      "grad_output": _operand_stats(go_e)},
        }
    return records


def _majority_pair(pairs: List[str]) -> Dict[str, Any]:
    """Majority-vote a list of per-step T-pair strings for one matmul path.

    Ties break deterministically by the pair string (so re-runs agree). Returns
    the winning pair, the full per-pair vote counts, and the vote total.
    """
    counter = Counter(pairs)
    if not counter:
        return {"pair": "none-none", "votes": {}, "n": 0}
    top = max(counter.values())
    winner = sorted(p for p, c in counter.items() if c == top)[0]
    return {"pair": winner, "votes": dict(counter), "n": sum(counter.values())}


def accumulate_majority(
    per_step_records: List[Dict[int, Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Collapse a list of per-step ``{expert_id: record}`` maps (one per observed
    step, same GEMM) into a single ``{expert_id: majority_record}``.

    For each expert and each of the three matmul paths, the reported ``pair`` is
    the majority across the steps in which that expert was routed at least one
    token (empty-expert steps contribute a ``none-*`` vote, same as the raw
    records, so a rarely-routed expert honestly shows up as mostly ``none``).

    Returns ``{expert_id: {"n_steps": int, "n_tokens_total": int,
        "forward_y": {"pair","votes","n"}, "backward_gx": {...},
        "backward_gw": {...}}}``.
    """
    expert_ids = sorted({eid for step in per_step_records for eid in step})
    out: Dict[int, Dict[str, Any]] = {}
    for eid in expert_ids:
        step_recs = [step[eid] for step in per_step_records if eid in step]
        rec: Dict[str, Any] = {
            "n_steps": len(step_recs),
            "n_tokens_total": int(sum(r.get("n_tokens", 0) for r in step_recs)),
        }
        for path in MATMUL_PATHS:
            pairs = [r[path]["pair"] for r in step_recs if path in r]
            rec[path] = _majority_pair(pairs)
        out[eid] = rec
    return out


def extract_offs(args: tuple, kwargs: dict) -> Optional[torch.Tensor]:
    """Pull the ``offs`` tensor from a ``torch._grouped_mm`` call. It is passed
    as a keyword in the model, but accept a trailing positional too."""
    if "offs" in kwargs and kwargs["offs"] is not None:
        return kwargs["offs"]
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() == 1 and a.dtype in (torch.int32, torch.int64):
            return a
    return None
