#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Collect val-loss + grad_norm results from a recipe sweep.

Reads the sweep state file (which lists each config's RUN_TAG) and parses the
corresponding training logs for:
  * validation loss at each val step
  * grad_norm trajectory stats (mean / max / #spikes>2)
  * final train loss

Usage:
  python3 scripts/collect_sweep_results.py <SWEEP_TAG>
  python3 scripts/collect_sweep_results.py            # uses newest sweep
"""

import re
import sys
from pathlib import Path

SWEEP_ROOT = Path("/wekafs/zhitwang/alto_runs/sweeps")
LOG_ROOT = Path("/wekafs/zhitwang/alto_runs/orchestrator_logs")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s+(\d+)\s+loss:\s+([\d.]+)\s+grad_norm:\s+([\d.eE+-]+)")
VAL_RE = re.compile(r"validate step:\s+(\d+)\s+loss:\s+([\d.]+)")


def _resolve_tag(arg):
    if arg:
        return arg
    states = sorted(SWEEP_ROOT.glob("sweep*.state"),
                    key=lambda p: p.stat().st_mtime)
    if not states:
        sys.exit("no sweep state files found")
    return states[-1].stem


def parse_run(run_tag):
    log = LOG_ROOT / f"{run_tag}.log"
    if not log.exists():
        return None
    text = ANSI.sub("", log.read_text(errors="replace"))
    steps, vals = [], []
    for line in text.splitlines():
        if "[rank0]" not in line:
            continue
        mv = VAL_RE.search(line)
        if mv:
            vals.append((int(mv.group(1)), float(mv.group(2))))
            continue
        ms = STEP_RE.search(line)
        if ms:
            steps.append((int(ms.group(1)), float(ms.group(2)),
                          float(ms.group(3))))
    # de-dup by step
    vd = dict(vals)
    gn = [g for _, _, g in steps]
    out = {
        "n_steps": len(steps),
        "final_train_loss": steps[-1][1] if steps else None,
        "val": dict(sorted(vd.items())),
        "grad_mean": (sum(gn) / len(gn)) if gn else None,
        "grad_max": max(gn) if gn else None,
        "grad_spikes_gt2": sum(1 for g in gn if g > 2.0),
        # Precise check: only flag an actual non-finite loss/grad_norm value,
        # not the substring "inf" inside "INFO" log lines.
        "has_nan": bool(re.search(
            r"(loss|grad_norm):\s+(nan|inf|-inf)", text, re.IGNORECASE)),
    }
    return out


def main():
    tag = _resolve_tag(sys.argv[1] if len(sys.argv) > 1 else None)
    state = SWEEP_ROOT / f"{tag}.state"
    if not state.exists():
        sys.exit(f"state file not found: {state}")

    print(f"# Sweep results: {tag}\n")
    rows = []
    for line in state.read_text().splitlines():
        if not line.strip():
            continue
        name = line.split()[0]
        rc = next((p.split("=")[1] for p in line.split() if p.startswith("rc=")), "?")
        run_tag = next((p.split("=")[1] for p in line.split()
                        if p.startswith("run_tag=")), None)
        res = parse_run(run_tag) if run_tag else None
        rows.append((name, rc, run_tag, res))

    for name, rc, run_tag, res in rows:
        print(f"## {name}  (rc={rc})")
        if not res:
            print(f"   no log parsed (run_tag={run_tag})\n")
            continue
        print(f"   steps={res['n_steps']}  final_train_loss={res['final_train_loss']}")
        print(f"   grad_norm: mean={res['grad_mean']:.3f} max={res['grad_max']:.2f} "
              f"spikes>2={res['grad_spikes_gt2']}  nan/inf={res['has_nan']}")
        if res["val"]:
            vstr = "  ".join(f"{s}:{v:.4f}" for s, v in res["val"].items())
            print(f"   val_loss @ step: {vstr}")
        print()

    # compact val comparison table
    print("## Val-loss comparison (step -> loss)")
    all_steps = sorted({s for _, _, _, r in rows if r for s in r["val"]})
    header = "config".ljust(28) + "".join(f"{s:>9}" for s in all_steps)
    print(header)
    for name, rc, run_tag, res in rows:
        if not res:
            continue
        line = name.ljust(28)
        for s in all_steps:
            v = res["val"].get(s)
            line += f"{v:>9.4f}" if v is not None else f"{'-':>9}"
        print(line)


if __name__ == "__main__":
    main()
