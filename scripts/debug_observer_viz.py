#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Standalone visualizer for DebugObserverModifier dumps.

No alto import — only torch + matplotlib required.

Usage
-----
# Summary only (no plots written):
python scripts/debug_observer_viz.py --dump ./outputs/debug_obs_lpt.pt --summary-only

# Full plots:
python scripts/debug_observer_viz.py \\
    --dump ./outputs/debug_obs_lpt.pt \\
    --out-dir ./viz

# With BF16 baseline overlay:
python scripts/debug_observer_viz.py \\
    --dump ./outputs/debug_obs_lpt.pt \\
    --baseline ./outputs/debug_obs_bf16.pt \\
    --out-dir ./viz

# Filter to specific layers:
python scripts/debug_observer_viz.py \\
    --dump ./outputs/debug_obs_lpt.pt \\
    --layer-regex 'experts'
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dump(path: str) -> tuple[dict, dict]:
    """Returns (layers_dict, meta_dict).

    layers_dict: {fqn: {step_idx: {"input": T, ...}}}
    meta_dict: the "_meta" entry
    """
    blob: dict = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.pop("_meta", {})
    return blob, meta


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _absmax(t: torch.Tensor) -> float:
    return t.float().abs().max().item()


def _std(t: torch.Tensor) -> float:
    return t.float().std().item()


def _collect_series(layer_data: dict, key: str) -> tuple[list[int], list[float], list[float]]:
    """Returns (steps, absmax_values, std_values) for a given tensor key."""
    steps, abs_maxes, stds = [], [], []
    for step in sorted(s for s in layer_data if isinstance(s, int)):
        tensors = layer_data[step]
        if key not in tensors:
            continue
        t = tensors[key]
        steps.append(step)
        abs_maxes.append(_absmax(t))
        stds.append(_std(t))
    return steps, abs_maxes, stds


def _align_baseline(run_data: dict, baseline_data: dict) -> dict:
    """Return a copy of baseline_data re-keyed to match run_data's step numbers.

    BF16 and MXFP4 runs may capture at different absolute step indices (e.g.
    the calibration pre-steps shift the MXFP4 counter).  We match by ordinal
    position (1st capture ↔ 1st capture, 2nd ↔ 2nd, …) so overlays always
    show the same training epoch regardless of step numbering.
    """
    run_steps = sorted(s for s in run_data if isinstance(s, int))
    base_steps = sorted(s for s in baseline_data if isinstance(s, int))
    remapped = {}
    for run_step, base_step in zip(run_steps, base_steps):
        remapped[run_step] = baseline_data[base_step]
    # preserve the "active" gate key if present
    if "active" in baseline_data:
        remapped["active"] = baseline_data["active"]
    return remapped


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(layers: dict, meta: dict) -> None:
    print(f"\n=== DebugObserver summary ===")
    print(f"  rank: {meta.get('rank', 'unknown')}")
    print(f"  iterations captured: {meta.get('iterations_captured', [])}")
    print(f"  layers: {len(layers)}")
    print(f"  capture_every: {meta.get('capture_every', '?')}  max_captures: {meta.get('max_captures', '?')}")
    print(f"  mlp2 input captured: {meta.get('mlp2_input_captured', False)}")
    print()
    print(f"{'Layer':<60}  {'keys':>30}  {'input absmax':>12}  {'gw absmax':>12}")
    print("-" * 120)
    for fqn, layer_data in sorted(layers.items()):
        steps = [s for s in layer_data if isinstance(s, int)]
        if not steps:
            continue
        sample = layer_data[min(steps)]
        keys = ",".join(sorted(sample.keys()))
        inp_absmax = _absmax(sample["input"]) if "input" in sample else float("nan")
        gw_key = "grad_weight" if "grad_weight" in sample else (
            "grad_mlp1_weight" if "grad_mlp1_weight" in sample else None
        )
        gw_absmax = _absmax(sample[gw_key]) if gw_key else float("nan")
        print(f"{fqn:<60}  {keys:>30}  {inp_absmax:>12.4f}  {gw_absmax:>12.4f}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_time_series(
    ax_absmax,
    ax_std,
    steps: list[int],
    absmax_vals: list[float],
    std_vals: list[float],
    label: str,
    color: str,
) -> None:
    ax_absmax.plot(steps, absmax_vals, marker="o", label=label, color=color)
    ax_std.plot(steps, std_vals, marker="o", label=label, color=color, linestyle="--")


def _plot_histogram(ax, t: torch.Tensor, label: str, color: str, alpha: float = 0.5) -> None:
    vals = t.float().abs().flatten().numpy()
    ax.hist(vals, bins=50, alpha=alpha, label=label, color=color, density=True)


def plot_layer(
    fqn: str,
    layer_data: dict,
    out_dir: Path,
    baseline_data: Optional[dict] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots", file=sys.stderr)
        return

    # Remap baseline step numbers to match the run's step numbers by ordinal
    # position, so overlays work even when step counters differ between runs.
    aligned_baseline = _align_baseline(layer_data, baseline_data) if baseline_data is not None else None

    tensor_keys = sorted({
        k for step, tensors in layer_data.items()
        if isinstance(step, int)
        for k in tensors
    })

    safe_fqn = re.sub(r"[^a-zA-Z0-9_.\-]", "_", fqn)
    layer_out = out_dir / safe_fqn
    layer_out.mkdir(parents=True, exist_ok=True)

    for key in tensor_keys:
        steps, absmax_vals, std_vals = _collect_series(layer_data, key)
        if not steps:
            continue

        # --- time-series plot ---
        fig, (ax_absmax, ax_std) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"{fqn}\n{key}", fontsize=9)
        _plot_time_series(ax_absmax, ax_std, steps, absmax_vals, std_vals, label="run", color="blue")
        if aligned_baseline is not None:
            bsteps, babs, bstd = _collect_series(aligned_baseline, key)
            if bsteps:
                _plot_time_series(ax_absmax, ax_std, bsteps, babs, bstd, label="baseline", color="red")
        ax_absmax.set_ylabel("absmax")
        ax_absmax.legend(fontsize=7)
        ax_absmax.grid(True, alpha=0.3)
        ax_std.set_ylabel("std")
        ax_std.set_xlabel("step")
        ax_std.legend(fontsize=7)
        ax_std.grid(True, alpha=0.3)
        ts_path = layer_out / f"{key}_timeseries.png"
        fig.tight_layout()
        fig.savefig(ts_path, dpi=120)
        plt.close(fig)

        # --- per-iteration histogram ---
        steps_sorted = sorted(steps)
        n_cols = min(5, len(steps_sorted))
        n_rows = (len(steps_sorted) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        fig.suptitle(f"{fqn} | {key} | abs-value histograms", fontsize=9)
        for idx, step in enumerate(steps_sorted):
            ax = axes[idx // n_cols][idx % n_cols]
            t = layer_data[step].get(key)
            if t is None:
                ax.set_visible(False)
                continue
            _plot_histogram(ax, t, label=f"step {step}", color="steelblue")
            if aligned_baseline is not None and step in aligned_baseline:
                bt = aligned_baseline[step].get(key)
                if bt is not None:
                    _plot_histogram(ax, bt, label="baseline", color="red")
            ax.set_title(f"step {step}", fontsize=7)
            ax.legend(fontsize=6)
            ax.set_xlabel("|value|", fontsize=6)
        # hide unused axes
        for idx in range(len(steps_sorted), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
        hist_path = layer_out / f"{key}_histograms.png"
        fig.tight_layout()
        fig.savefig(hist_path, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _match_baseline_fqn(fqn: str, baseline_layers: dict) -> Optional[dict]:
    """Look up baseline data for a given FQN.

    Tries exact match first, then falls back to matching by the longest common
    suffix so that FQN differences caused by wrapper modules (e.g. FSDP shards
    or LPT submodule renaming) don't break the overlay.
    """
    if fqn in baseline_layers:
        return baseline_layers[fqn]
    # suffix match: find the baseline FQN whose suffix best matches
    best_fqn, best_len = None, 0
    for bfqn in baseline_layers:
        # find longest common suffix component-by-component
        fqn_parts = fqn.split(".")
        bfqn_parts = bfqn.split(".")
        common = 0
        for a, b in zip(reversed(fqn_parts), reversed(bfqn_parts)):
            if a == b:
                common += 1
            else:
                break
        if common > best_len:
            best_len, best_fqn = common, bfqn
    if best_len >= 2:  # require at least 2 matching suffix components
        return baseline_layers[best_fqn]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DebugObserverModifier dumps")
    parser.add_argument("--dump", required=True, help="Path to the .pt dump file")
    parser.add_argument("--baseline", default=None, help="Optional BF16 baseline .pt dump")
    parser.add_argument("--out-dir", default="./viz", help="Directory to write PNG files")
    parser.add_argument("--layer-regex", default=None, help="Regex to filter layer FQNs")
    parser.add_argument("--summary-only", action="store_true", help="Print summary, skip plotting")
    parser.add_argument("--debug-fqns", action="store_true",
                        help="Print FQNs from both dumps and their matches, then exit")
    args = parser.parse_args()

    layers, meta = load_dump(args.dump)
    baseline_layers, baseline_meta = load_dump(args.baseline) if args.baseline else ({}, {})

    if args.debug_fqns:
        print(f"\n=== FQNs in dump ({len(layers)}) ===")
        for fqn in sorted(layers):
            matched = _match_baseline_fqn(fqn, baseline_layers)
            tag = "[matched]" if matched is not None else "[NO MATCH]"
            print(f"  {tag}  {fqn}")
        print(f"\n=== FQNs in baseline ({len(baseline_layers)}) ===")
        for fqn in sorted(baseline_layers):
            print(f"  {fqn}")
        print(f"\ndump steps:     {meta.get('iterations_captured', [])}")
        print(f"baseline steps: {baseline_meta.get('iterations_captured', [])}")
        return

    if args.layer_regex:
        pat = re.compile(args.layer_regex)
        layers = {fqn: v for fqn, v in layers.items() if pat.search(fqn)}
        baseline_layers = {fqn: v for fqn, v in baseline_layers.items() if pat.search(fqn)}

    print_summary(layers, meta)

    if args.summary_only:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fqn, layer_data in sorted(layers.items()):
        print(f"Plotting {fqn} ...", end=" ", flush=True)
        baseline_data = _match_baseline_fqn(fqn, baseline_layers)
        plot_layer(fqn, layer_data, out_dir, baseline_data=baseline_data)
        print("done")

    print(f"\nPlots written to {out_dir}")


if __name__ == "__main__":
    main()
