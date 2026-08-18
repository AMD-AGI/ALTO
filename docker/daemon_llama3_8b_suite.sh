#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION="alto-llama3-8b-mxfp8-ablation"
LOG_FILE="${ALTO_DIR}/logs/llama3_8b_mxfp8_ablation.log"
ACTION="${1:-start}"
RUN_MODE="suite"

if [[ "${ACTION}" == "linear" ]]; then
    SESSION="${SESSION}-linear"
    LOG_FILE="${ALTO_DIR}/logs/llama3_8b_mxfp8_linear.log"
    RUN_MODE="linear"
fi

case "${ACTION}" in
    start|linear)
        if tmux has-session -t "${SESSION}" 2> /dev/null; then
            echo "Experiment suite is already running in tmux session: ${SESSION}" >&2
            exit 1
        fi

        mkdir -p "$(dirname "${LOG_FILE}")"
        tmux new-session -d -s "${SESSION}" \
            "cd \"${ALTO_DIR}\" && exec bash ./docker/run_llama3_8b.sh ${RUN_MODE} >> \"${LOG_FILE}\" 2>&1"
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
        echo "Usage: $0 {start|linear|status|attach|stop}" >&2
        exit 2
        ;;
esac
