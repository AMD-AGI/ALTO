#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${TRAIN_LOG_DIR:-/wekafs/zhitwang/alto_runs/orchestrator_logs}"
LOG_FILE="${1:-}"

if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="$(
    python - <<'PY' "${LOG_DIR}"
from pathlib import Path
import sys

log_dir = Path(sys.argv[1])
logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
print(logs[0] if logs else "")
PY
  )"
fi

echo "Training status @ $(date -Is)"
echo "================================================================================"

python - <<'PY'
import subprocess

out = subprocess.check_output(["ps", "-eo", "pid,etime,pcpu,pmem,args"], text=True)
torchruns = []
workers = []
orchestrators = []
for line in out.splitlines():
    if "subprocess.check_output" in line or "check_training_status.sh" in line:
        continue
    if "torchrun" in line:
        torchruns.append(line)
    elif "-m alto.train" in line:
        workers.append(line)
    elif "run_baselines_1k" in line or "phase_" in line and ".sh" in line:
        orchestrators.append(line)

def split_ps(line: str):
    parts = line.split(None, 4)
    if len(parts) < 5:
        return ("?", "?", "?", "?", line)
    return parts

print("Processes")
print("---------")
if not torchruns and not workers and not orchestrators:
    print("No active torchrun / alto.train process found.")
else:
    rows = []
    for kind, lines in (("orchestrator", orchestrators), ("torchrun", torchruns), ("worker", workers)):
        for line in lines:
            pid, etime, pcpu, pmem, args = split_ps(line)
            config = "-"
            for token in args.split():
                if token.startswith("gpt_oss_"):
                    config = token
            rows.append((kind, pid, etime, pcpu, pmem, config))
    print(f"{'kind':<13} {'pid':>8} {'elapsed':>12} {'cpu%':>7} {'mem%':>7}  config")
    print(f"{'-'*13} {'-'*8} {'-'*12} {'-'*7} {'-'*7}  {'-'*24}")
    for row in rows:
        print(f"{row[0]:<13} {row[1]:>8} {row[2]:>12} {row[3]:>7} {row[4]:>7}  {row[5]}")
PY
echo

if [[ -z "${LOG_FILE}" || ! -f "${LOG_FILE}" ]]; then
  echo "Progress"
  echo "--------"
  echo "No log file found. Set TRAIN_LOG_DIR or pass a log path:"
  echo "  $0 /path/to/train.log"
else
  python - <<'PY' "${LOG_FILE}"
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys
import yaml

log_path = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
ts_re = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
step_re = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.]+)")
grad_re = re.compile(r"grad_norm:\s*([0-9.]+)")
val_re = re.compile(r"validate step:\s*(\d+).*?loss:\s*([0-9.]+)")
total_re = re.compile(r"total steps\s+(\d+)")
tb_re = re.compile(r"TensorBoard logging enabled\. Logs will be saved at (.+)")
config_re = re.compile(r"--config(?:=|\s+)(\S+)")
recipe_re = re.compile(r"Loading recipe from file (.+)")


def clean(line: str) -> str:
    return ansi.sub("", line).strip()


def parse_time(line: str) -> datetime | None:
    match = ts_re.search(line)
    if not match:
        return None
    return datetime.strptime(".".join(match.groups()), "%Y-%m-%d %H:%M:%S.%f")


lines = log_path.read_text(errors="replace").splitlines()
total_steps = None
train = []
valid = []
checkpoint_lines = []
completed = False
failed = False
hard_error = False
tb_dir = None
config_name = None
recipe_path = None

for raw in lines:
    line = clean(raw)
    if config_name is None:
        match = config_re.search(line)
        if match:
            config_name = match.group(1)
    if recipe_path is None:
        match = recipe_re.search(line)
        if match:
            recipe_path = match.group(1).strip()
    match = tb_re.search(line)
    if match:
        tb_dir = match.group(1)
    if total_steps is None:
        match = total_re.search(line)
        if match:
            total_steps = int(match.group(1))

    if "Training completed" in line:
        completed = True
    if "Traceback" in line or "ChildFailedError" in line or "RuntimeError:" in line:
        failed = True
        hard_error = True

    if "Saving the checkpoint" in line or "Finished saving the checkpoint" in line:
        checkpoint_lines.append(line)

    t = parse_time(line)
    match = step_re.search(line)
    if match and "validate step" not in line:
        grad_match = grad_re.search(line)
        grad_norm = float(grad_match.group(1)) if grad_match else None
        train.append((int(match.group(1)), float(match.group(2)), grad_norm, t, line))
        continue

    match = val_re.search(line)
    if match:
        valid.append((int(match.group(1)), float(match.group(2)), t, line))

if not train:
    print("Progress")
    print("--------")
    print(f"Log: {log_path}")
    print("No training step lines found yet.")
    if failed:
        print("Status: failed or errored before logging a training step.")
    raise SystemExit(0)

first_step, first_loss, first_grad, first_time, _ = train[0]
last_step, last_loss, last_grad, last_time, last_line = train[-1]

status = "COMPLETED" if completed else "FAILED" if hard_error else "RUNNING or INCOMPLETE"
progress = f"{last_step}/{total_steps}" if total_steps else str(last_step)
progress_pct = f"{100.0 * last_step / total_steps:.1f}%" if total_steps else "-"

latest_val = "-"
if valid:
    v_step, v_loss, _, v_line = valid[-1]
    latest_val = f"step {v_step}, loss {v_loss:.5f}"

elapsed_s = None
sec_per_step = None
eta_s = None

if first_time and last_time and last_time >= first_time:
    elapsed = last_time - first_time
    elapsed_s = elapsed.total_seconds()
    done_steps = max(last_step - first_step, 1)
    sec_per_step = elapsed.total_seconds() / done_steps
    if total_steps and not completed and last_step < total_steps:
        remain_steps = total_steps - last_step
        eta_s = remain_steps * sec_per_step

def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def recipe_summary(path_str: str | None) -> dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {"path": str(path), "error": "not found"}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        lpt = data["training_stage"]["lpt_modifiers"]["LowPrecisionTrainingModifier"]
    except Exception as exc:
        return {"path": str(path), "error": repr(exc)}
    keys = [
        "scheme",
        "targets",
        "ignore",
        "use_2dblock_x",
        "use_2dblock_w",
        "use_hadamard",
        "use_sr_grad",
        "use_dge",
        "clip_mode",
        "two_level_scaling",
    ]
    return {"path": str(path), **{key: lpt.get(key) for key in keys}}

print("Progress")
print("--------")
print(f"Log         : {log_path}")
if tb_dir:
    print(f"TensorBoard : {tb_dir}")
print(f"Status      : {status}")
print(f"Step        : {progress:<12} ({progress_pct})")
print(f"Train loss  : {last_loss:.5f}    (first logged: {first_loss:.5f} @ step {first_step})")
if last_grad is not None:
    first_grad_s = f"{first_grad:.4f}" if first_grad is not None else "-"
    print(f"Grad norm   : {last_grad:.4f}    (first logged: {first_grad_s} @ step {first_step})")
else:
    print("Grad norm   : -")
print(f"Val loss    : {latest_val}")
print(f"Elapsed     : {fmt_duration(elapsed_s)}")
print(f"Avg/step    : {sec_per_step:.2f}s" if sec_per_step else "Avg/step    : -")
print(f"ETA         : {fmt_duration(eta_s)}")
print()

print("Recipe")
print("------")
print(f"Config      : {config_name or '-'}")
summary = recipe_summary(recipe_path)
if summary is None:
    print("Recipe file : - (BF16 / no low-precision recipe detected)")
elif "error" in summary:
    print(f"Recipe file : {summary['path']}")
    print(f"Error       : {summary['error']}")
else:
    print(f"Recipe file : {summary['path']}")
    print(f"Scheme      : {summary.get('scheme')}")
    print(f"Targets     : {summary.get('targets')}")
    print(f"Ignore      : {summary.get('ignore')}")
    print(
        "Knobs       : "
        f"x2d={summary.get('use_2dblock_x')}, "
        f"w2d={summary.get('use_2dblock_w')}, "
        f"hadamard={summary.get('use_hadamard')}, "
        f"sr_grad={summary.get('use_sr_grad')}, "
        f"dge={summary.get('use_dge')}, "
        f"clip={summary.get('clip_mode')}, "
        f"scaling={summary.get('two_level_scaling')}"
    )
print()

sample = train[-6:]
print("Recent train steps")
print("------------------")
print(f"{'step':>8} {'loss':>12} {'grad_norm':>12} {'time':>12}")
for step, loss, grad_norm, t, _line in sample:
    t_s = t.strftime("%H:%M:%S") if t else "-"
    grad_s = f"{grad_norm:.4f}" if grad_norm is not None else "-"
    print(f"{step:>8} {loss:>12.5f} {grad_s:>12} {t_s:>12}")

print()
print("Loss / grad trend")
print("-----------------")
print(f"{'window':>14} {'step':>8} {'loss':>12} {'loss_delta':>12} {'grad_norm':>12} {'grad_delta':>12}")
milestones: list[tuple[str, int]] = [
    ("start", 0),
    ("25%", max(0, len(train) // 4 - 1)),
    ("50%", max(0, len(train) // 2 - 1)),
    ("75%", max(0, (len(train) * 3) // 4 - 1)),
    ("latest", len(train) - 1),
]
seen = set()
base_loss = first_loss
base_grad = first_grad
for label, idx in milestones:
    if idx in seen:
        continue
    seen.add(idx)
    step, loss, grad_norm, _t, _line = train[idx]
    loss_delta = loss - base_loss
    if grad_norm is not None and base_grad is not None:
        grad_delta = grad_norm - base_grad
        grad_s = f"{grad_norm:.4f}"
        grad_delta_s = f"{grad_delta:+.4f}"
    else:
        grad_s = "-"
        grad_delta_s = "-"
    print(
        f"{label:>14} {step:>8} {loss:>12.5f} {loss_delta:>+12.5f} "
        f"{grad_s:>12} {grad_delta_s:>12}"
    )

if valid:
    print()
    print("Recent validations")
    print("------------------")
    print(f"{'step':>8} {'loss':>12} {'time':>12}")
    for step, loss, t, _line in valid[-4:]:
        t_s = t.strftime("%H:%M:%S") if t else "-"
        print(f"{step:>8} {loss:>12.5f} {t_s:>12}")

if checkpoint_lines:
    print()
    print("Recent checkpoints")
    print("------------------")
    for item in checkpoint_lines[-4:]:
        print(item)
PY
fi
echo

echo "GPU utilization"
echo "---------------"
if command -v amd-smi >/dev/null 2>&1; then
  gpu_output="$(timeout 10s amd-smi monitor 2>/dev/null || true)"
  if [[ -z "${gpu_output}" ]]; then
    gpu_output="$(rocm-smi 2>/dev/null || true)"
  fi
  GPU_OUTPUT="${gpu_output}" python - <<'PY'
import os

lines = [line.rstrip() for line in os.environ.get("GPU_OUTPUT", "").splitlines() if line.strip()]
if not lines:
    print("No GPU utilization output available.")
    raise SystemExit(0)

header = None
rows = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("GPU "):
        header = stripped
    elif stripped[:1].isdigit():
        rows.append(stripped)

if header:
    print(header)
if rows:
    for row in rows:
        print(row)
else:
    print("\n".join(lines[-12:]))
PY
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi || true
else
  echo "No amd-smi or rocm-smi found."
fi
