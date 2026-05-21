#!/usr/bin/env bash
# Node-health sidecar for NVFP4 long-horizon runs.
#
# Polls rocm-smi + dmesg every $INTERVAL seconds and writes the snapshots
# next to the training logs.  The sidecar runs *outside* the training
# torchrun so it cannot perturb training numerics; if it crashes the
# training keeps going.
#
# Usage:
#   bash scripts/nvfp4_node_health_sidecar.sh start <out_dir>   # nohup-detached
#   bash scripts/nvfp4_node_health_sidecar.sh stop  <out_dir>
#
# Recommended out_dir:
#   /wekafs/hanwang2/workspace/alto_runs/orchestrator_logs/probe_debug_<tag>/node_health
#
# Designed for AMD MI300X (rocm-smi).  If amd-smi is available it is also
# used.  Falls back gracefully if a tool is missing.

set -uo pipefail

ACTION=${1:-start}
OUT_DIR=${2:-}
INTERVAL=${INTERVAL:-30}
DMESG_LINES=${DMESG_LINES:-200}
PID_FILE=

if [[ -z "$OUT_DIR" ]]; then
  echo "usage: $0 {start|stop} <out_dir>" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/rocm" "$OUT_DIR/dmesg" "$OUT_DIR/amd"
PID_FILE="$OUT_DIR/.sidecar.pid"

stop_existing() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}

snapshot_loop() {
  local ts
  while true; do
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    {
      echo "=== sample $ts ==="
      if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showtemp --showuse --showmeminfo vram \
                 --showxgmierr --showpids 2>&1 || true
      else
        echo "rocm-smi NOT FOUND"
      fi
    } > "$OUT_DIR/rocm/sample_${ts}.txt"

    if command -v amd-smi >/dev/null 2>&1; then
      amd-smi metric --ecc --ecc-blocks --power --temperature \
              --usage 2>&1 > "$OUT_DIR/amd/sample_${ts}.txt" || true
    fi

    {
      echo "=== dmesg snapshot $ts (last $DMESG_LINES lines) ==="
      dmesg -T 2>/dev/null | tail -"$DMESG_LINES" || true
    } > "$OUT_DIR/dmesg/sample_${ts}.txt"

    sleep "$INTERVAL"
  done
}

case "$ACTION" in
  start)
    stop_existing
    nohup bash -c "$(declare -f snapshot_loop); \
                   OUT_DIR='$OUT_DIR' INTERVAL='$INTERVAL' \
                   DMESG_LINES='$DMESG_LINES' snapshot_loop" \
          > "$OUT_DIR/sidecar.out" 2> "$OUT_DIR/sidecar.err" &
    echo $! > "$PID_FILE"
    echo "[sidecar] started pid=$(cat "$PID_FILE") interval=${INTERVAL}s out_dir=$OUT_DIR"
    ;;
  stop)
    stop_existing
    {
      echo "=== final dmesg dump at $(date -u +%Y%m%dT%H%M%SZ) ==="
      dmesg -T 2>/dev/null | tail -500 || true
    } > "$OUT_DIR/dmesg/final.txt"
    if command -v rocm-smi >/dev/null 2>&1; then
      rocm-smi --showxgmierr --showtemp --showuse 2>&1 \
        > "$OUT_DIR/rocm/final.txt" || true
    fi
    echo "[sidecar] stopped, final snapshots written under $OUT_DIR"
    ;;
  *)
    echo "usage: $0 {start|stop} <out_dir>" >&2
    exit 2
    ;;
esac
