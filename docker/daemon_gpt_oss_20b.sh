#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-start}"
RUN_MODE="${2:-suite}"
SESSION="alto-gpt-oss-20b-${RUN_MODE}"
LOG_FILE="${ALTO_DIR}/logs/gpt_oss_20b_${RUN_MODE}.log"

case "${ACTION}" in
    start)
        if tmux has-session -t "${SESSION}" 2> /dev/null; then
            echo "Already running in tmux session: ${SESSION}" >&2
            exit 1
        fi

        mkdir -p "$(dirname "${LOG_FILE}")"
        tmux new-session -d -s "${SESSION}" \
            "cd \"${ALTO_DIR}\" && exec bash ./docker/run_gpt_oss_20b.sh ${RUN_MODE} >> \"${LOG_FILE}\" 2>&1"
        echo "Started tmux session: ${SESSION}"
        echo "Log file: ${LOG_FILE}"
        ;;
    status)
        tmux has-session -t "${SESSION}"
        ;;
    attach)
        exec tmux attach-session -t "${SESSION}"
        ;;
    stop)
        tmux kill-session -t "${SESSION}"
        ;;
    *)
        echo "Usage: $0 {start|status|attach|stop} [suite|compare-1k|bf16|attn|linear|linear-attn]" >&2
        exit 2
        ;;
esac
