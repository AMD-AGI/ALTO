#!/usr/bin/env bash
#
# Sequential quick-validation sweep for NVFP4 / BF16 GPT-OSS-20B recipes.
#
# Runs a fixed list of configs ONE AT A TIME (each on all 8 GPUs), 500 steps
# each, validation every 100 steps, capturing val loss + grad_norm.  Each
# config:
#   * fresh OUTPUT_DIR (HF init, independent run)
#   * crash recovery via launch_nvfp4_test2_16k.sh internal retry (MAX_ATTEMPTS)
#   * per-run wall-clock timeout (hang guard); on timeout we kill + move on
#   * failure isolation: a failed/timed-out run is logged and the sweep
#     continues to the next config
#   * idempotent: re-running skips configs already marked done in the state file
#
# No intermediate checkpoints are written (CKPT_INTERVAL > steps) — these are
# throwaway quick-validation runs, so we avoid the 234 GB/run DCP writes.
#
# Usage:
#   nohup bash scripts/sweep_nvfp4_recipes.sh > /dev/null 2>&1 & disown
#   tail -F /wekafs/zhitwang/alto_runs/sweeps/<SWEEP_TAG>.log
#
# Stop the whole sweep:
#   pkill -f sweep_nvfp4_recipes.sh ; then kill leftover run via its runfile.

set -uo pipefail

ROOT=/root/alto
LAUNCHER="${ROOT}/scripts/launch_nvfp4_test2_16k.sh"
LOG_ROOT=/wekafs/zhitwang/alto_runs/orchestrator_logs
SWEEP_ROOT=/wekafs/zhitwang/alto_runs/sweeps
RECIPE_DIR=/wekafs/zhitwang/recipes
mkdir -p "$SWEEP_ROOT" "$LOG_ROOT"

# Allow resuming an existing sweep by passing its tag; otherwise start fresh.
SWEEP_TAG=${1:-"sweep5_$(date -u +%Y%m%d_%H%M%S)"}
STATE="${SWEEP_ROOT}/${SWEEP_TAG}.state"
SWEEP_LOG="${SWEEP_ROOT}/${SWEEP_TAG}.log"
touch "$STATE"

slog() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SWEEP_LOG"; }

# Shared per-run knobs.
export TRAINING_STEPS=500
export VAL_FREQ=100
export VAL_STEPS=64
export CKPT_INTERVAL=100000   # > steps => no mid-run checkpoint writes
export CKPT_KEEP=2            # checkpointer hard-requires keep_latest_k >= 2
export MAX_ATTEMPTS=3         # per-run crash retries (DCP resume)
export RETRY_BACKOFF_S=60

# ---- Config table, in execution order 3, 7, 1, 2 --------------------------
# Fields:  NAME | CONFIG | RECIPE_FILE | GBS | LOCAL_BS | TIMEOUT_S
#
# NOTE: c5_gbs64_comp_on_all (NVFP4 + outlier compensation lora_rank=32) is
# DEFERRED — it crashes in DecomposedLinear.from_linear (alto/nn/
# decomposed_linear.py:36) because the lora-compensation path is incompatible
# with the meta-device / DTensor model build.  Re-enable once that bug is
# fixed.  The remaining 4 configs all use lora_rank=0 and are unaffected.
CONFIGS=(
  "c3_bf16_gbs64|gpt_oss_20b_pretrain|NONE|64|2|46800"
  "c7_gbs64_comp_off_all|gpt_oss_20b_lpt|${RECIPE_DIR}/nvfp4_alllayers_rank0.yaml|64|2|46800"
  "c1_nvfp4_test2_gbs64|gpt_oss_20b_lpt|${RECIPE_DIR}/nvfp4_first1last2_rank0.yaml|64|2|46800"
  "c2_nvfp4_test2_gbs16|gpt_oss_20b_lpt|${RECIPE_DIR}/nvfp4_first1last2_rank0.yaml|16|1|14400"
)

is_done() { grep -q "^${1} " "$STATE" 2>/dev/null; }

cleanup_run() {
  # Kill any lingering processes for a finished/timed-out run, using the
  # ENTRYPOINT_PGID recorded by the launcher in the runfile.
  local run_tag=$1
  local runfile="${LOG_ROOT}/${run_tag}.runfile"
  if [ -f "$runfile" ]; then
    local pgid
    pgid=$(grep '^ENTRYPOINT_PGID=' "$runfile" | tail -1 | cut -d= -f2)
    [ -n "$pgid" ] && kill -KILL -- -"$pgid" 2>/dev/null || true
  fi
  pkill -KILL -f "torchrun.*${run_tag}" 2>/dev/null || true
  pkill -KILL -f "alto\.train.*${run_tag}" 2>/dev/null || true
  sleep 5
}

slog "=== sweep ${SWEEP_TAG} start; ${#CONFIGS[@]} configs; steps=${TRAINING_STEPS} val_freq=${VAL_FREQ} ==="

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME CFG RCP GBS_V LBS_V TMO <<< "$entry"

  if is_done "$NAME"; then
    slog "SKIP ${NAME} (already done)"
    continue
  fi

  STAMP=$(date -u +%Y%m%d_%H%M%S)
  export RUN_TAG="${NAME}_${STAMP}"
  export OUTPUT_DIR="/wekafs/zhitwang/alto_runs/${RUN_TAG}"
  export CONFIG="$CFG"
  export RECIPE_FILE="$RCP"
  export GBS="$GBS_V"
  export LOCAL_BS="$LBS_V"

  slog "START ${NAME}  RUN_TAG=${RUN_TAG}  CONFIG=${CFG}  RECIPE=${RCP}  GBS=${GBS_V} LOCAL_BS=${LBS_V}  timeout=${TMO}s"

  START_EPOCH=$(date -u +%s)
  timeout "$TMO" bash "$LAUNCHER"
  RC=$?
  ELAPSED=$(( $(date -u +%s) - START_EPOCH ))

  cleanup_run "$RUN_TAG"

  if [ "$RC" -eq 124 ]; then
    slog "TIMEOUT ${NAME} after ${ELAPSED}s (rc=124); killed, moving on"
    echo "${NAME} rc=timeout elapsed=${ELAPSED} run_tag=${RUN_TAG} done_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
  elif [ "$RC" -eq 0 ]; then
    slog "DONE ${NAME} ok in ${ELAPSED}s"
    echo "${NAME} rc=0 elapsed=${ELAPSED} run_tag=${RUN_TAG} done_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
  else
    slog "FAIL ${NAME} rc=${RC} after ${ELAPSED}s (retries exhausted); moving on"
    echo "${NAME} rc=${RC} elapsed=${ELAPSED} run_tag=${RUN_TAG} done_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
  fi
done

slog "=== sweep ${SWEEP_TAG} complete ==="
slog "results: python3 ${ROOT}/scripts/collect_sweep_results.py ${SWEEP_TAG}"
