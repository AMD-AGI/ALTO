#!/usr/bin/env bash
#
# Background launcher for NVFP4-test-2 × 16,000-step GPT-OSS-20B training.
#
# This wraps scripts/entrypoint_nvfp4_gpt_oss_20b_test2_16k.sh with:
#   1. nohup-detached background execution
#   2. retry-on-non-zero-exit (SIGABRT / NCCL hang / transient HIP)
#   3. resumable attempts via DCP auto-resume from same dump_folder
#   4. central runfile holding RUN_TAG + PIDs so you can stop / inspect later
#
# History note:
#   Previously this script also started a monitor sidecar that periodically
#   `du -sb`'d the output dump folder.  That `du` traversed the entire
#   234 GB DCP checkpoint tree every 10 min, hammering the Weka metadata
#   server cluster-wide and causing dataloader stalls during training.
#   The monitor has been removed entirely; if you need status digests, run
#   them on-demand (e.g. tail train log + grep step:).
#
# Usage (foreground, just for smoke):
#   bash scripts/launch_nvfp4_test2_16k.sh
#
# Usage (background, intended):
#   nohup bash scripts/launch_nvfp4_test2_16k.sh > /dev/null 2>&1 &
#   disown
#   # tail train log:
#   tail -F /wekafs/zhitwang/alto_runs/orchestrator_logs/nvfp4_test2_16k_<STAMP>.log
#
# Stop:
#   bash scripts/launch_nvfp4_test2_16k.sh stop <STAMP>
#
# Env knobs (overridable):
#   RUN_TAG          (auto:  nvfp4_test2_16k_<UTC stamp>)
#   MAX_ATTEMPTS     5     hard cap on retries
#   RETRY_BACKOFF_S  60    sleep between retries (GPU state settle)
#
# Exit codes treated as retryable: any non-zero (see is_retryable).

set -uo pipefail

ROOT=/root/alto
ENTRYPOINT="${ROOT}/scripts/entrypoint_nvfp4_gpt_oss_20b_test2_16k.sh"

LOG_ROOT=/wekafs/zhitwang/alto_runs/orchestrator_logs
mkdir -p "$LOG_ROOT"

ACTION=${1:-start}

# --- stop branch -----------------------------------------------------------
# Bug fix (2026-05-27): the entrypoint is launched via `setsid`, which puts
# torchrun + its workers in a NEW session / new process group, detached from
# DRIVER_PID's group.  So `kill -- -$DRIVER_PID` could not cascade-kill them.
# We now persist ENTRYPOINT_PGID in the runfile at spawn time, and `stop`
# uses that pgid to reach torchrun + all worker children in one signal.
if [ "$ACTION" = "stop" ]; then
  STAMP=${2:-}
  if [ -z "$STAMP" ]; then
    echo "usage: $0 stop <STAMP>" >&2
    exit 2
  fi
  # Bug fix (2026-05-31): previously hardcoded the "nvfp4_test2_16k_" prefix,
  # which broke `stop` for custom RUN_TAGs (e.g. sweep runs).  Accept either a
  # full RUN_TAG or the legacy <STAMP> form.
  if [ -f "${LOG_ROOT}/${STAMP}.runfile" ]; then
    RUNFILE="${LOG_ROOT}/${STAMP}.runfile"
  else
    RUNFILE="${LOG_ROOT}/nvfp4_test2_16k_${STAMP}.runfile"
  fi
  if [ ! -f "$RUNFILE" ]; then
    echo "runfile not found: $RUNFILE (tried ${STAMP}.runfile and nvfp4_test2_16k_${STAMP}.runfile)" >&2
    exit 2
  fi
  source "$RUNFILE"

  echo "stopping training driver pid=$DRIVER_PID  entrypoint_pgid=${ENTRYPOINT_PGID:-?}"

  # 1) kill the entrypoint's own PG (catches torchrun + python workers cleanly)
  if [ -n "${ENTRYPOINT_PGID:-}" ]; then
    kill -TERM -- -"$ENTRYPOINT_PGID" 2>/dev/null || true
  fi
  # 2) kill the launcher driver shell itself (stops the retry loop)
  kill -TERM "$DRIVER_PID" 2>/dev/null || true

  # Drain SIGTERM cascade (worker shutdown).
  sleep 10

  # 3) Belt-and-suspenders: catch anything still bearing our RUN_TAG.
  pkill -KILL -f "torchrun.*${STAMP}" 2>/dev/null || true
  pkill -KILL -f "alto\.train.*${STAMP}" 2>/dev/null || true
  if [ -n "${ENTRYPOINT_PGID:-}" ]; then
    kill -KILL -- -"$ENTRYPOINT_PGID" 2>/dev/null || true
  fi
  sleep 3

  echo "post-stop check:"
  pgrep -af "torchrun.*${STAMP}|alto.train.*${STAMP}" | grep -v grep || echo "(nothing left)"
  exit 0
fi

# --- start branch ----------------------------------------------------------
RUN_TAG=${RUN_TAG:-"nvfp4_test2_16k_$(date -u +%Y%m%d_%H%M%S)"}
export RUN_TAG

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_BACKOFF_S=${RETRY_BACKOFF_S:-60}

# All outputs of this job land under one umbrella for easy GC at end.
OUTPUT_DIR=${OUTPUT_DIR:-"/wekafs/zhitwang/alto_runs/${RUN_TAG}"}
export OUTPUT_DIR
mkdir -p "$OUTPUT_DIR"

TRAIN_LOG="${LOG_ROOT}/${RUN_TAG}.log"
DRIVER_LOG="${LOG_ROOT}/${RUN_TAG}.driver.log"
RUNFILE="${LOG_ROOT}/${RUN_TAG}.runfile"

# Sanity: refuse to start a second copy with the same OUTPUT_DIR
if pgrep -af "torchrun.*${RUN_TAG}" >/dev/null 2>&1; then
  echo "ERROR: a training process with RUN_TAG=${RUN_TAG} is already running" >&2
  exit 1
fi

cat > "$RUNFILE" <<EOF
RUN_TAG=${RUN_TAG}
OUTPUT_DIR=${OUTPUT_DIR}
TRAIN_LOG=${TRAIN_LOG}
DRIVER_LOG=${DRIVER_LOG}
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DRIVER_PID=$$
EOF

# Persist triton cache across attempts and across retries.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/wekafs/zhitwang/triton-cache}
mkdir -p "$TRITON_CACHE_DIR"

is_retryable() {
  # Empirical observation (2026-05-25): torchrun wraps any single-rank
  # SIGABRT into ChildFailedError which raises a Python exception and
  # exits with code 1 — not -6/134.  So the original whitelist of
  # specific signal codes was a dead path.  We treat **any non-zero
  # exit** as retryable, and rely on MAX_ATTEMPTS as the upper bound.
  # Below we also grep the train log for evidence of the failure class
  # purely as a log-side breadcrumb (does not affect the return value).
  local exit_code=$1
  [ "$exit_code" -eq 0 ] && return 1
  if [ -f "$TRAIN_LOG" ] \
      && grep -qE 'ChildFailedError|exitcode: -6|Signal 6|SIGABRT|exitcode: -15|exitcode: 134|exitcode: 137' "$TRAIN_LOG" 2>/dev/null; then
    echo "  (detected SIGABRT/ChildFailedError signature in $TRAIN_LOG)" \
        | tee -a "$DRIVER_LOG"
  fi
  return 0
}

ATTEMPT=1
LAST_EXIT=99
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[${TS}] ATTEMPT=${ATTEMPT}/${MAX_ATTEMPTS} starting; log=${TRAIN_LOG}" \
      | tee -a "$DRIVER_LOG"

  # New process group so we can kill the whole subtree on stop.
  # IMPORTANT: append (>>) not truncate (>), so historical training log
  # from earlier attempts survives — used by post-mortem analysis.  A
  # banner line marks each attempt boundary so log readers can find
  # the right offset.
  {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "║ ATTEMPT=${ATTEMPT}/${MAX_ATTEMPTS}   start=${TS}   RUN_TAG=${RUN_TAG}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
  } >> "$TRAIN_LOG"

  # Bug fix (2026-05-27): record the entrypoint child's PGID so the `stop`
  # branch (and watchdog hang-kill) can actually reach torchrun + workers.
  # With `setsid`, the child is in its own process group, detached from
  # DRIVER_PID's PG, so `kill -- -DRIVER_PID` previously couldn't cascade.
  setsid bash "$ENTRYPOINT" \
      >> "$TRAIN_LOG" 2>&1 &
  ENTRYPOINT_PID=$!
  # PGID == PID for setsid-spawned process (it's the session leader).
  ENTRYPOINT_PGID=$ENTRYPOINT_PID
  # Persist for stop-branch use (overwrite previous attempt's entry).
  sed -i '/^ENTRYPOINT_PGID=/d' "$RUNFILE" 2>/dev/null || true
  echo "ENTRYPOINT_PGID=${ENTRYPOINT_PGID}" >> "$RUNFILE"
  echo "  spawned entrypoint pid=${ENTRYPOINT_PID} pgid=${ENTRYPOINT_PGID}" \
      | tee -a "$DRIVER_LOG"

  # Wait for entrypoint subshell to finish; capture its exit code.
  wait "$ENTRYPOINT_PID"
  LAST_EXIT=$?

  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[${TS}] ATTEMPT=${ATTEMPT} exit=${LAST_EXIT}" | tee -a "$DRIVER_LOG"

  if [ $LAST_EXIT -eq 0 ]; then
    echo "[${TS}] training completed cleanly on attempt ${ATTEMPT}" \
        | tee -a "$DRIVER_LOG"
    break
  fi

  if is_retryable "$LAST_EXIT"; then
    echo "[${TS}] retryable failure (exit=${LAST_EXIT}); sleeping ${RETRY_BACKOFF_S}s then retrying (DCP auto-resume)" \
        | tee -a "$DRIVER_LOG"
    ATTEMPT=$((ATTEMPT + 1))
    sleep "$RETRY_BACKOFF_S"
    continue
  fi

  echo "[${TS}] non-retryable failure (exit=${LAST_EXIT}); aborting driver" \
      | tee -a "$DRIVER_LOG"
  break
done

FINAL_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[${FINAL_TS}] driver done attempts=${ATTEMPT} last_exit=${LAST_EXIT}" \
    | tee -a "$DRIVER_LOG"

exit $LAST_EXIT
