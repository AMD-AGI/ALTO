#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Standalone visualizer for MoEMatmulPatternObserverModifier dumps.

Renders, per MoE layer and per Grouped GEMM (mlp1/mlp2), a heatmap of the
AdaHOP outlier-pattern T-pair for each of the three matmuls of each expert:

    rows = experts (global id, merged across per-rank dumps)
    cols = matmul paths  [forward_y, backward_gx, backward_gw]
    cell = the 9-way T-pair category (fixed color legend)

No alto import — only torch + matplotlib + numpy required.

Usage
-----
# Summary only:
python scripts/moe_pattern_viz.py --dump-glob './outputs/moe_patterns_rank*.pt' --summary-only

# Full heatmaps (merges all per-rank dumps of a run):
python scripts/moe_pattern_viz.py \\
    --dump-glob './outputs/moe_patterns_rank*.pt' \\
    --out-dir ./viz

# Filter to specific layers:
python scripts/moe_pattern_viz.py \\
    --dump-glob './outputs/moe_patterns_rank*.pt' \\
    --layer-regex 'layers.0' --out-dir ./viz
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# 9-way T-pair palette (matches AdaHOP pattern_visualization.PATTERN_COLORS).
PATTERN_COLORS = {
    "row-row": "#e74c3c",
    "row-col": "#f39c12",
    "row-none": "#f1c40f",
    "col-row": "#3498db",
    "col-col": "#2ecc71",
    "col-none": "#1abc9c",
    "none-row": "#9b59b6",
    "none-col": "#e91e63",
    "none-none": "#95a5a6",
}
_PAIR_ORDER = list(PATTERN_COLORS.keys())
_PAIR_INDEX = {p: i for i, p in enumerate(_PAIR_ORDER)}
MATMUL_PATHS = ["forward_y", "backward_gx", "backward_gw"]
GEMMS = ["mlp1", "mlp2"]

# 3-way single-operand palette (each matmul input is classified row/col/none).
SINGLE_PATTERN_COLORS = {
    "row": "#e74c3c",
    "col": "#3498db",
    "none": "#95a5a6",
}
_SINGLE_ORDER = list(SINGLE_PATTERN_COLORS.keys())
_SINGLE_INDEX = {p: i for i, p in enumerate(_SINGLE_ORDER)}

# Each matmul is A @ B. The T-pair string is "A_pattern-B_pattern"; these labels
# name the two operands per path (matches moe_pattern_hooks T-pair construction).
# IMPORTANT: patterns are shown in the AS-FED-TO-MATMUL orientation — the axis the
# low-precision GEMM actually scales. An operand marked (ᵀ) enters transposed, so
# its row/col reads FLIPPED vs the stored tensor (a stored 'row' weight shows as
# 'col' under weightᵀ). Same physical tensor, different axis — not a contradiction.
#   forward_y   = input @ weightᵀ        (weight enters transposed)
#   backward_gx = grad_output @ weight   (weight in stored orientation)
#   backward_gw = grad_outputᵀ @ input   (grad_output enters transposed)
# The bool marks whether that operand is transposed relative to its stored form.
OPERAND_LABELS = {
    "forward_y":   (("input", False),        ("weight (ᵀ)", True)),
    "backward_gx": (("grad_output", False),  ("weight", False)),
    "backward_gw": (("grad_output (ᵀ)", True), ("input", False)),
}


def _split_pair(pair: str) -> tuple[str, str]:
    """'row-col' -> ('row', 'col'); tolerant of the 'none-none' default."""
    a, _, b = (pair or "none-none").partition("-")
    a = a if a in SINGLE_PATTERN_COLORS else "none"
    b = b if b in SINGLE_PATTERN_COLORS else "none"
    return a, b


# ---------------------------------------------------------------------------
# Loading + merge
# ---------------------------------------------------------------------------

def load_and_merge(dump_paths: List[str]) -> tuple[dict, dict, dict]:
    """Merge per-rank dumps into one structure.

    Returns (merged, majority, meta) where
        merged[fqn][step][gemm][global_expert_id]   = per-step record
        majority[fqn][gemm][global_expert_id]        = majority-vote record
        meta = combined _meta (num_global_experts, etc.)
    """
    merged: Dict[str, Any] = {}
    majority: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    ep_sizes = set()
    global_counts = set()
    ranks = []
    steps_all = set()

    for p in dump_paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        m = blob.pop("_meta", {})
        maj = blob.pop("_majority", {})
        ranks.append(m.get("rank", "?"))
        ep_sizes.add(m.get("ep_size", 1))
        global_counts.add(m.get("num_global_experts", 0))
        steps_all.update(m.get("iterations_captured", []))
        for fqn, per_step in blob.items():
            f = merged.setdefault(fqn, {})
            for step, per_gemm in per_step.items():
                s = f.setdefault(step, {})
                for gemm, records in per_gemm.items():
                    g = s.setdefault(gemm, {})
                    g.update(records)  # global ids are disjoint across ranks
        for fqn, per_gemm in maj.items():
            mf = majority.setdefault(fqn, {})
            for gemm, records in per_gemm.items():
                mf.setdefault(gemm, {}).update(records)  # disjoint global ids

    meta = {
        "ranks": sorted(ranks, key=str),
        "ep_size": max(ep_sizes) if ep_sizes else 1,
        "num_global_experts": max(global_counts) if global_counts else 0,
        "iterations_captured": sorted(steps_all),
    }
    return merged, majority, meta


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(majority: dict, meta: dict) -> None:
    print("\n=== MoE matmul pattern summary (majority over captured steps) ===")
    print(f"  ranks merged: {meta.get('ranks')}")
    print(f"  ep_size: {meta.get('ep_size')}  global experts: {meta.get('num_global_experts')}")
    print(f"  steps captured: {meta.get('iterations_captured')}")
    print(f"  layers: {len(majority)}")
    print()
    for fqn in sorted(majority):
        per_gemm = majority[fqn]
        for gemm in GEMMS:
            records = per_gemm.get(gemm)
            if not records:
                continue
            # Tally the winning majority T-pair across experts per matmul path.
            print(f"  {fqn} | {gemm} | {len(records)} experts")
            for path in MATMUL_PATHS:
                counts: Dict[str, int] = {}
                for rec in records.values():
                    pair = rec.get(path, {}).get("pair", "none-none")
                    counts[pair] = counts.get(pair, 0) + 1
                tally = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                print(f"      {path:<12} {tally}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pair_grid(records: dict, expert_ids: List[int]):
    import numpy as np
    grid = np.full((len(expert_ids), len(MATMUL_PATHS)), _PAIR_INDEX["none-none"], dtype=int)
    for r, eid in enumerate(expert_ids):
        rec = records.get(eid, {})
        for c, path in enumerate(MATMUL_PATHS):
            pair = rec.get(path, {}).get("pair", "none-none")
            grid[r, c] = _PAIR_INDEX.get(pair, _PAIR_INDEX["none-none"])
    return grid


def plot_gemm(fqn: str, step, gemm: str, records: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed; skipping plots", file=sys.stderr)
        return

    expert_ids = sorted(records.keys())
    if not expert_ids:
        return
    grid = _pair_grid(records, expert_ids)

    cmap = ListedColormap([PATTERN_COLORS[p] for p in _PAIR_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(_PAIR_ORDER) + 0.5, 1), cmap.N)

    height = max(3.0, 0.28 * len(expert_ids) + 1.5)
    fig, ax = plt.subplots(figsize=(6, height))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(MATMUL_PATHS)))
    ax.set_xticklabels(MATMUL_PATHS, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(expert_ids)))
    ax.set_yticklabels([f"e{e}" for e in expert_ids], fontsize=6)
    ax.set_ylabel("expert (global id)")
    ax.set_title(f"{fqn}\nstep {step} | {gemm} — matmul-input outlier pattern", fontsize=9)

    legend = [Patch(facecolor=PATTERN_COLORS[p], label=p) for p in _PAIR_ORDER]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=6, title="T-pair", title_fontsize=7)

    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", f"{fqn}_step{step}_{gemm}")
    out_path = out_dir / f"{safe}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_gemm_operands(fqn: str, step, gemm: str, records: dict, out_dir: Path) -> None:
    """Per-operand view: y = the 6 matmul operands (3 matmuls, each A @ B split
    into its two rows), x = experts. Each cell is colored by that operand's
    single pattern (row/col/none) with the pattern name written inside."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed; skipping plots", file=sys.stderr)
        return

    expert_ids = sorted(records.keys())
    if not expert_ids:
        return

    # Build the row axis: one line per operand, grouped by matmul path.
    row_labels: List[str] = []
    row_specs: List[tuple[str, int]] = []  # (matmul_path, operand_idx 0=A/1=B)
    for path in MATMUL_PATHS:
        (a_lbl, _), (b_lbl, _) = OPERAND_LABELS[path]
        row_labels.append(f"{path}\nA: {a_lbl}")
        row_specs.append((path, 0))
        row_labels.append(f"{path}\nB: {b_lbl}")
        row_specs.append((path, 1))

    n_rows, n_cols = len(row_specs), len(expert_ids)
    grid = np.full((n_rows, n_cols), _SINGLE_INDEX["none"], dtype=int)
    text = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for r, (path, operand_idx) in enumerate(row_specs):
        for c, eid in enumerate(expert_ids):
            pair = records.get(eid, {}).get(path, {}).get("pair", "none-none")
            pat = _split_pair(pair)[operand_idx]
            grid[r, c] = _SINGLE_INDEX[pat]
            text[r][c] = pat

    cmap = ListedColormap([SINGLE_PATTERN_COLORS[p] for p in _SINGLE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(_SINGLE_ORDER) + 0.5, 1), cmap.N)

    width = max(6.0, 0.6 * n_cols + 2.5)
    fig, ax = plt.subplots(figsize=(width, 0.7 * n_rows + 1.5))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"e{e}" for e in expert_ids], fontsize=7)
    ax.set_xlabel("expert (global id)")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(f"{fqn}\nstep {step} | {gemm} — per-operand outlier pattern", fontsize=9)
    ax.text(0.0, -0.16,
            "patterns shown in as-fed-to-matmul orientation; (ᵀ) = operand enters "
            "transposed, so its row/col is flipped vs the stored tensor",
            transform=ax.transAxes, fontsize=6.5, style="italic",
            color="#555555", ha="left", va="top", wrap=True)

    # Separator lines between the three matmul groups (every 2 rows).
    for r in range(2, n_rows, 2):
        ax.axhline(r - 0.5, color="white", linewidth=2.0)

    # Write the pattern name inside each cell.
    for r in range(n_rows):
        for c in range(n_cols):
            ax.text(c, r, text[r][c], ha="center", va="center",
                    fontsize=7, color="black")

    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", f"{fqn}_step{step}_{gemm}_operands")
    out_path = out_dir / f"{safe}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MoE matmul pattern dumps")
    parser.add_argument("--dump-glob", required=True,
                        help="Glob for per-rank .pt dumps, e.g. './outputs/moe_patterns_rank*.pt'")
    parser.add_argument("--out-dir", default="./viz", help="Directory for PNGs")
    parser.add_argument("--layer-regex", default=None, help="Regex to filter layer FQNs")
    parser.add_argument("--summary-only", action="store_true", help="Print summary, skip plotting")
    parser.add_argument("--per-step", action="store_true",
                        help="Also render the raw per-step plots (default: majority only)")
    args = parser.parse_args()

    dump_paths = sorted(glob.glob(args.dump_glob))
    if not dump_paths:
        print(f"No dumps matched {args.dump_glob!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Merging {len(dump_paths)} dump(s): {dump_paths}")

    merged, majority, meta = load_and_merge(dump_paths)

    if args.layer_regex:
        pat = re.compile(args.layer_regex)
        merged = {fqn: v for fqn, v in merged.items() if pat.search(fqn)}
        majority = {fqn: v for fqn, v in majority.items() if pat.search(fqn)}

    # Older dumps predate the accumulated majority; fall back to the last step.
    if not majority and merged:
        print("No _majority block found; falling back to the last captured step.",
              file=sys.stderr)
        for fqn, per_step in merged.items():
            last = max(per_step) if per_step else None
            if last is not None:
                majority[fqn] = per_step[last]

    print_summary(majority, meta)
    if args.summary_only:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Majority plots (the default view) — labeled "majority" in title/filename.
    for fqn in sorted(majority):
        for gemm in GEMMS:
            records = majority[fqn].get(gemm)
            if records:
                plot_gemm(fqn, "majority", gemm, records, out_dir)
                plot_gemm_operands(fqn, "majority", gemm, records, out_dir)

    if args.per_step:
        for fqn in sorted(merged):
            for step in sorted(merged[fqn]):
                for gemm in GEMMS:
                    records = merged[fqn][step].get(gemm)
                    if records:
                        plot_gemm(fqn, step, gemm, records, out_dir)
                        plot_gemm_operands(fqn, step, gemm, records, out_dir)
    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
