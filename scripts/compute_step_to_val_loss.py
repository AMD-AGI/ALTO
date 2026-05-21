#!/usr/bin/env python3
"""Extract train loss / grad_norm + validate step / val_loss from torchtitan
logs and compute the ``step_to_val_loss=X.XX`` metric.

Usage
-----
    python scripts/compute_step_to_val_loss.py \
        --log /path/to/e2e100_<tag>_<stamp>.log \
        --target 8.0

    # Multi-log mode (one row per recipe):
    python scripts/compute_step_to_val_loss.py \
        --log path/to/bf16.log \
        --log path/to/mxfp4.log \
        --log path/to/nvfp4_default.log \
        --log path/to/nvfp4_test_2.log \
        --target 8.0 --target 8.5 --target 9.0

Outputs a human-readable summary plus a CSV at ``--csv-out`` (default
``/tmp/step_to_val_loss_<ts>.csv``).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Optional

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?step:\s*(?P<step>\d+)\s+loss:\s*"
    r"(?P<loss>[0-9.]+)\s+grad_norm:\s*(?P<grad>[0-9.eE+-]+)"
)
VAL_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?validate step:\s*(?P<step>\d+)\s+loss:\s*"
    r"(?P<loss>[0-9.]+)"
)


def parse_log(path: Path):
    train: dict[int, tuple[float, float]] = {}
    val: dict[int, float] = {}
    if not path.exists():
        return train, val
    text = path.read_text(errors="ignore")
    for raw in text.splitlines():
        s = ANSI.sub("", raw)
        m = VAL_PAT.search(s)
        if m:
            val[int(m.group("step"))] = float(m.group("loss"))
            continue
        m = TRAIN_PAT.search(s)
        if m:
            train[int(m.group("step"))] = (float(m.group("loss")), float(m.group("grad")))
    return train, val


def step_to_val_loss(val: dict[int, float], target: float) -> Optional[int]:
    for step in sorted(val):
        if val[step] <= target:
            return step
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, action="append", required=True,
                    help="path(s) to torchtitan run log(s); pass multiple to compare")
    ap.add_argument("--target", type=float, action="append",
                    help="val-loss target(s) for step_to_val_loss metric")
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.target:
        args.target = [8.0, 8.5, 9.0]

    rows = []
    for log in args.log:
        train, val = parse_log(log)
        tag = log.stem
        last_train = max(train.items(), default=None)
        first_val = min(val.items(), default=None)
        last_val = max(val.items(), default=None)
        targets = {t: step_to_val_loss(val, t) for t in args.target}
        rows.append({
            "log_path": str(log),
            "tag": tag,
            "n_train_steps": len(train),
            "n_val_steps": len(val),
            "train_step_last": last_train[0] if last_train else None,
            "train_loss_last": last_train[1][0] if last_train else None,
            "train_grad_last": last_train[1][1] if last_train else None,
            "val_step_first": first_val[0] if first_val else None,
            "val_loss_first": first_val[1] if first_val else None,
            "val_step_last": last_val[0] if last_val else None,
            "val_loss_last": last_val[1] if last_val else None,
            "step_to_val_loss": targets,
        })

    print()
    print("=== train / val summary ===")
    print(f"{'tag':<55} {'train_steps':>11} {'val_steps':>9} {'final_train_loss':>16} {'final_val_loss':>14}")
    for r in rows:
        ftl = f"{r['train_loss_last']:.4f}" if r['train_loss_last'] is not None else "-"
        fvl = f"{r['val_loss_last']:.4f}" if r['val_loss_last'] is not None else "-"
        print(f"{r['tag']:<55} {r['n_train_steps']:>11} {r['n_val_steps']:>9} {ftl:>16} {fvl:>14}")

    print()
    print(f"=== step_to_val_loss (first step where val_loss ≤ target) ===")
    header = "tag".ljust(55) + "".join(f"  t={t:.2f}".rjust(11) for t in args.target)
    print(header)
    for r in rows:
        cells = [r["tag"].ljust(55)]
        for t in args.target:
            v = r["step_to_val_loss"].get(t)
            cells.append((f"{v}" if v is not None else "—").rjust(11))
        print("".join(cells))

    csv_out = args.csv_out or Path(f"/tmp/step_to_val_loss_{int(time.time())}.csv")
    fieldnames = [
        "tag", "log_path",
        "n_train_steps", "n_val_steps",
        "train_step_last", "train_loss_last", "train_grad_last",
        "val_step_first", "val_loss_first", "val_step_last", "val_loss_last",
    ] + [f"step_to_val_loss_t{t:g}" for t in args.target]
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in fieldnames if k in r}
            for t in args.target:
                out[f"step_to_val_loss_t{t:g}"] = r["step_to_val_loss"].get(t)
            w.writerow(out)
    print()
    print(f"csv={csv_out}")


if __name__ == "__main__":
    main()
