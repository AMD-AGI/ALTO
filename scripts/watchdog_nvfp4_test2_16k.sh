#!/usr/bin/env bash
#
# External watchdog for the 16k NVFP4-test-2 training.
#
# Polls every ${INTERVAL_S} seconds.  Handles three lifecycle states
# the in-process retry loop CANNOT recover from:
#
#   1. Launcher process itself died (e.g. nohup got SIGHUP'd, OOM-killer,
#      stale shell)  → respawn the launcher with the same RUN_TAG so DCP
#      auto-resumes from the latest checkpoint.
#
#   2. Training appears to be HUNG (no new step logged for
#      ${HANG_THRESHOLD_S} seconds, but torchrun is still alive)
#      → kill the torchrun process group; the launcher's retry loop
#      will then catch the non-zero exit and respawn.
#
#   3. Training is healthy → only update the watchdog log and continue.
#
# Termination conditions (watchdog exits cleanly):
#   * "Training completed" line observed in $TRAIN_LOG
#   * Driver log records "driver done" with last_exit=0
#   * MAX_RESPAWNS exceeded (default 5)
#
# Usage:
#   nohup bash scripts/watchdog_nvfp4_test2_16k.sh <RUN_TAG> > /dev/null 2>&1 &
#   disown
#
# Stop:
#   pkill -f "watchdog_nvfp4_test2_16k.sh <RUN_TAG>"

set -uo pipefail

ROOT=/root/alto
LAUNCHER="${ROOT}/scripts/launch_nvfp4_test2_16k.sh"
LOG_ROOT=/wekafs/zhitwang/alto_runs/orchestrator_logs

RUN_TAG=${1:?"usage: $0 <RUN_TAG>"}
INTERVAL_S=${INTERVAL_S:-600}        # 10 min poll
HANG_THRESHOLD_S=${HANG_THRESHOLD_S:-600}  # 10 min no-step = hang
MAX_RESPAWNS=${MAX_RESPAWNS:-5}

RUNFILE="${LOG_ROOT}/${RUN_TAG}.runfile"
WD_LOG="${LOG_ROOT}/${RUN_TAG}.watchdog.log"

log() {
  local ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[${ts}] $*" | tee -a "$WD_LOG"
}

respawn_launcher() {
  local why=$1
  log "RESPAWN launcher: reason=${why}"
  # Use the same RUN_TAG so DCP auto-resume kicks in (dump_folder unchanged).
  RUN_TAG="${RUN_TAG}" nohup bash "$LAUNCHER" > /dev/null 2>&1 &
  disown
  sleep 15  # give launcher time to write a fresh runfile
}

last_train_step_time_epoch() {
  # Returns the unix epoch (seconds) of the most recent "step: <N>" log line,
  # or 0 if none.
  #
  # Bug fix (2026-05-27): two issues in the original implementation:
  #   1. grep without -a treats the train log as binary (it contains stray
  #      stderr / NCCL trace bytes from earlier attempts) and just prints
  #      "binary file matches", so ts always came back empty → watchdog
  #      forever said "no train step lines yet" even with thousands of
  #      step lines in the file.
  #   2. `\s+` is not portable inside grep -E (ERE); GNU grep may or may
  #      not honor it depending on version.  Use POSIX [[:space:]]+ which
  #      always works in ERE.
  local log_file=$1
  [ -f "$log_file" ] || { echo 0; return; }
  local ts
  ts=$(sed -E 's/\x1B\[[0-9;]*[mK]//g' "$log_file" \
        | grep -aE '\[titan\][^|]*step:[[:space:]]+[0-9]+[[:space:]]+loss:' \
        | grep -av 'validate step' \
        | tail -1 \
        | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' \
        | head -1)
  if [ -z "$ts" ]; then
    echo 0
    return
  fi
  date -u -d "$ts" +%s 2>/dev/null || echo 0
}

is_torchrun_alive() {
  pgrep -af "torchrun.*${RUN_TAG}" 2>/dev/null | grep -v grep >/dev/null
}

is_completed() {
  [ -f "$1" ] || return 1
  # Same -a fix as last_train_step_time_epoch: train log may contain binary
  # bytes from earlier failed attempts, and grep without -a would refuse to
  # search and return false negative.
  grep -aqE 'Training completed|driver done attempts=[0-9]+ last_exit=0' "$1" 2>/dev/null
}

RESPAWNS=0
log "watchdog starting   RUN_TAG=${RUN_TAG}   interval=${INTERVAL_S}s   hang_threshold=${HANG_THRESHOLD_S}s   max_respawns=${MAX_RESPAWNS}"

while true; do
  sleep "$INTERVAL_S"

  if [ ! -f "$RUNFILE" ]; then
    log "WARN  runfile missing: ${RUNFILE} (perhaps launcher about to write it)"
    continue
  fi

  # Re-source each tick because PIDs change across respawns.
  source "$RUNFILE"
  TRAIN_LOG=${TRAIN_LOG:-${LOG_ROOT}/${RUN_TAG}.log}
  DRIVER_LOG=${DRIVER_LOG:-${LOG_ROOT}/${RUN_TAG}.driver.log}

  # ── Completion check first ───────────────────────────────────────
  if is_completed "$TRAIN_LOG" || is_completed "$DRIVER_LOG"; then
    log "training completed; watchdog exiting cleanly"
    exit 0
  fi

  # ── Driver process alive? ───────────────────────────────────────
  if ! ps -p "$DRIVER_PID" >/dev/null 2>&1; then
    log "DRIVER DEAD  driver_pid=${DRIVER_PID} not running"
    if [ "$RESPAWNS" -ge "$MAX_RESPAWNS" ]; then
      log "FATAL  respawn budget exhausted (${RESPAWNS}/${MAX_RESPAWNS}); watchdog exiting"
      exit 1
    fi
    RESPAWNS=$((RESPAWNS + 1))
    respawn_launcher "driver-pid-${DRIVER_PID}-dead"
    continue
  fi

  # ── torchrun (the actual worker spawner) alive? ─────────────────
  if ! is_torchrun_alive; then
    # The launcher may be in the retry backoff sleep, which is fine —
    # we should give it room to spawn a new torchrun before declaring
    # the whole thing dead.  Re-check after one more interval.
    log "INFO  torchrun not running but launcher pid=${DRIVER_PID} alive (likely in retry backoff); continuing"
    continue
  fi

  # ── Progress check: did a new train step land within HANG_THRESHOLD? ─
  LAST_EPOCH=$(last_train_step_time_epoch "$TRAIN_LOG")
  NOW_EPOCH=$(date -u +%s)
  AGE=$((NOW_EPOCH - LAST_EPOCH))

  if [ "$LAST_EPOCH" -eq 0 ]; then
    log "INFO  no train step lines yet in ${TRAIN_LOG}; presumably still warming up"
    continue
  fi

  if [ "$AGE" -gt "$HANG_THRESHOLD_S" ]; then
    log "HANG DETECTED  last_step_age=${AGE}s > threshold=${HANG_THRESHOLD_S}s"
    log "killing torchrun process group + worker python procs to force launcher retry"
    # Bug fix (2026-05-27): the previous version only matched
    # `torchrun.*${RUN_TAG}` which leaves the 8 `python -m alto.train`
    # worker processes alive if any of them is hung inside a HIP call.
    # Now we kill BOTH the torchrun parent and the worker subprocesses.
    # We also try the entrypoint PGID first if the runfile recorded it
    # (newer launcher versions do), which is the cleanest cascade.
    if [ -n "${ENTRYPOINT_PGID:-}" ]; then
      kill -TERM -- -"$ENTRYPOINT_PGID" 2>/dev/null || true
    fi
    pkill -TERM -f "torchrun.*${RUN_TAG}" 2>/dev/null || true
    pkill -TERM -f "alto\.train.*${RUN_TAG}" 2>/dev/null || true
    sleep 15
    # Anything still alive after SIGTERM → SIGKILL.
    if is_torchrun_alive \
        || pgrep -af "alto\.train.*${RUN_TAG}" | grep -qv grep; then
      if [ -n "${ENTRYPOINT_PGID:-}" ]; then
        kill -KILL -- -"$ENTRYPOINT_PGID" 2>/dev/null || true
      fi
      pkill -KILL -f "torchrun.*${RUN_TAG}" 2>/dev/null || true
      pkill -KILL -f "alto\.train.*${RUN_TAG}" 2>/dev/null || true
    fi
    continue
  fi

  # ── Healthy ─────────────────────────────────────────────────────
  log "OK    driver=alive  torchrun=alive  last_step_age=${AGE}s  respawns=${RESPAWNS}/${MAX_RESPAWNS}"
done
