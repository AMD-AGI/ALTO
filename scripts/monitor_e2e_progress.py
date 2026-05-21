#!/usr/bin/env python3
"""Periodically print E2E training progress for a multi-recipe sweep.

Designed for the long-horizon NVFP4 E2E experiments produced by
``scripts/nvfp4_recipe_e2e_probe.py`` under
``/wekafs/.../orchestrator_logs/`` with a shared ``<stamp>`` suffix.

Each tick prints, per-recipe:
  - current train step (vs ``--total-steps``)
  - validation pass count
  - latest train loss / grad_norm
  - latest val loss
  - measured sec/step over the last window
  - status (running / DONE / FAILED / waiting)

Status output goes to stdout AND is appended to ``--out`` so the run can
be tailed by another agent / user from a separate session.

Usage
-----
  python scripts/monitor_e2e_progress.py \\
      --stamp longhorizon_1k_20260520_025621 \\
      --tags bf16,nvfp4_default,nvfp4_test_2 \\
      --total-steps 1000 \\
      --interval 600

If ``--stamp`` is omitted the script picks the most recent stamp matching
``--log-prefix`` in ``--log-dir``.

Stops automatically once every recipe reports ``DONE`` (or ``FAILED``);
override with ``--no-stop``.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?step:\s*(?P<step>\d+)\s+loss:\s*"
    r"(?P<loss>[0-9.]+)\s+grad_norm:\s*(?P<grad>[0-9.eE+-]+)"
)
VAL_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?validate step:\s*(?P<step>\d+)\s+loss:\s*"
    r"(?P<loss>[0-9.]+)"
)
TIME_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\]\s+(?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),"
    r"(?P<ms>\d+)"
)


def parse_log(path: Path):
    """Return train rows {step: (loss, grad, dt)}, val rows {step: (loss, dt)},
    and an end-state marker ('DONE' / 'FAILED' / None)."""
    train: dict[int, tuple[float, float, datetime.datetime | None]] = {}
    val: dict[int, tuple[float, datetime.datetime | None]] = {}
    state = None
    if not path.exists():
        return train, val, state
    text = path.read_text(errors="ignore")
    for raw in text.splitlines():
        s = ANSI.sub("", raw)
        dt = None
        m_time = TIME_PAT.match(s)
        if m_time:
            try:
                dt = datetime.datetime.strptime(
                    f"{m_time.group('dt')}.{m_time.group('ms')}",
                    "%Y-%m-%d %H:%M:%S.%f",
                )
            except ValueError:
                dt = None
        m = VAL_PAT.search(s)
        if m:
            val[int(m.group("step"))] = (float(m.group("loss")), dt)
            continue
        m2 = TRAIN_PAT.search(s)
        if m2:
            train[int(m2.group("step"))] = (
                float(m2.group("loss")),
                float(m2.group("grad")),
                dt,
            )
        if state is None:
            if "Training completed" in s:
                state = "DONE"
            elif "ChildFailedError" in s or "RuntimeError:" in s or "Traceback" in s:
                state = "FAILED"
    return train, val, state


def fmt(v, w: int = 8, prec: int = 4) -> str:
    return f"{v:>{w}.{prec}f}" if v is not None else f"{'-':>{w}}"


def fmt_int(v, w: int = 5) -> str:
    return f"{v:>{w}d}" if v is not None else f"{'-':>{w}}"


def make_snapshot(
    log_dir: Path, stamp: str, tags: list[str], log_prefix: str, total_steps: int
):
    rows = []
    for tag in tags:
        path = log_dir / f"{log_prefix}_{tag}_{stamp}.log"
        if not path.exists():
            rows.append({
                "tag": tag,
                "exists": False,
                "status": "waiting",
            })
            continue
        train, val, state = parse_log(path)
        last_train = max(train.keys(), default=None)
        last_val = max(val.keys(), default=None)
        train_lg = train.get(last_train) if last_train is not None else None
        val_l = val.get(last_val) if last_val is not None else None
        # sec/step over last 20 train rows
        sec_per_step = None
        if train and len(train) >= 5:
            sorted_steps = sorted(k for k, v in train.items() if v[2] is not None)
            tail = sorted_steps[-20:]
            if len(tail) >= 2:
                dts = [train[s][2] for s in tail]
                ds = (dts[-1] - dts[0]).total_seconds() / max(1, tail[-1] - tail[0])
                if ds > 0:
                    sec_per_step = ds
        if state == "DONE":
            status = "DONE"
        elif state == "FAILED":
            status = "FAILED"
        elif last_train is not None:
            status = "running"
        else:
            status = "init"
        rows.append({
            "tag": tag,
            "exists": True,
            "last_train_step": last_train,
            "last_val_step": last_val,
            "train_loss": train_lg[0] if train_lg else None,
            "train_grad": train_lg[1] if train_lg else None,
            "val_loss": val_l[0] if val_l else None,
            "val_count": len(val),
            "sec_per_step": sec_per_step,
            "status": status,
        })
    return rows


def render_block(stamp: str, rows: list[dict], total_steps: int) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append(f"========== {ts}  stamp={stamp}  total_steps={total_steps} ==========")
    header = (
        f"{'tag':<18} {'step':>10} {'val_n':>6} {'train_loss':>10} "
        f"{'grad':>8} {'val_loss':>10} {'sec/step':>9}  status"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        if not r["exists"]:
            lines.append(f"{r['tag']:<18} {'(not started)':>40}   {r['status']}")
            continue
        step_str = f"{r['last_train_step'] or 0:>5d}/{total_steps:<4d}"
        line = (
            f"{r['tag']:<18} {step_str:>10} "
            f"{fmt_int(r['val_count'], 6)} "
            f"{fmt(r['train_loss'], 10)} "
            f"{fmt(r['train_grad'], 8)} "
            f"{fmt(r['val_loss'], 10)} "
            f"{fmt(r['sec_per_step'], 9, 2)}  {r['status']}"
        )
        lines.append(line)
    return "\n".join(lines)


def auto_detect_stamp(log_dir: Path, log_prefix: str) -> str | None:
    pattern = re.compile(rf"{re.escape(log_prefix)}_(?P<tag>[^_]+)_(?P<stamp>[^.]+)\.log")
    candidates: dict[str, float] = {}
    for p in log_dir.glob(f"{log_prefix}_*_*.log"):
        m = pattern.match(p.name)
        if m:
            stamp = m.group("stamp")
            candidates[stamp] = max(candidates.get(stamp, 0.0), p.stat().st_mtime)
    if not candidates:
        return None
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/wekafs/hanwang2/workspace/alto_runs/orchestrator_logs"),
    )
    ap.add_argument(
        "--log-prefix",
        type=str,
        default="e2e1k_longhorizon",
        help="log filename prefix shared across recipes",
    )
    ap.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="run stamp; auto-detect latest if omitted",
    )
    ap.add_argument(
        "--tags",
        type=str,
        default="bf16,nvfp4_default,nvfp4_test_2",
        help="comma-separated recipe tags in run order",
    )
    ap.add_argument("--total-steps", type=int, default=1000)
    ap.add_argument(
        "--interval", type=int, default=600, help="seconds between ticks"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="append-to file for the heartbeat log (defaults under --log-dir)",
    )
    ap.add_argument(
        "--no-stop",
        action="store_true",
        help="keep running after all recipes hit DONE/FAILED",
    )
    args = ap.parse_args()

    stamp = args.stamp or auto_detect_stamp(args.log_dir, args.log_prefix)
    if stamp is None:
        raise SystemExit(
            f"could not auto-detect stamp under {args.log_dir} "
            f"with prefix {args.log_prefix}"
        )
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    out_path = args.out or (
        args.log_dir / f"monitor_{args.log_prefix}_{stamp}.log"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[monitor] stamp={stamp}  tags={tags}  total_steps={args.total_steps}  "
        f"interval={args.interval}s  out={out_path}",
        flush=True,
    )

    while True:
        rows = make_snapshot(args.log_dir, stamp, tags, args.log_prefix, args.total_steps)
        block = render_block(stamp, rows, args.total_steps)
        print(block, flush=True)
        with out_path.open("a") as f:
            f.write(block + "\n\n")

        active_states = {r["status"] for r in rows}
        all_terminal = all(r["status"] in ("DONE", "FAILED") for r in rows)
        if all_terminal and not args.no_stop:
            print(
                f"[monitor] all recipes reached terminal state "
                f"({sorted(active_states)}); exiting.",
                flush=True,
            )
            with out_path.open("a") as f:
                f.write(
                    f"[monitor] all recipes reached terminal state at "
                    f"{datetime.datetime.utcnow().isoformat()}Z\n"
                )
            return

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("[monitor] interrupted by user", flush=True)
            return


if __name__ == "__main__":
    main()
