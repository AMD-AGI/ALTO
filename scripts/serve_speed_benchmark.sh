#!/usr/bin/env bash
set -euo pipefail

# Generic serving benchmark wrapper.
# - BACKEND can be vllm or sglang
# - MODEL can be a local path or a model id supported by that backend
# - DATASET defaults to a built-in synthetic prompt mix, so no external dataset
#   is required
# - MAX_CONCURRENCY=auto runs a short sweep and picks the best token throughput

# ---- Server config ---------------------------------------------------------
BACKEND="vllm"
MODEL="/path/to/model-or-hf-id"
SERVED_MODEL_NAME=""
HOST="127.0.0.1"
PORT=""
API_KEY=""
SKIP_LAUNCH="false"
KEEP_SERVER="false"

# Server parallelism. Use auto to consume all visible GPUs for tensor parallel.
TENSOR_PARALLEL="auto"
PIPELINE_PARALLEL="1"

# Optional backend launch tuning.
MAX_MODEL_LEN=""
TRUST_REMOTE_CODE="false"
GPU_MEMORY_UTILIZATION="0.9"   # vLLM only
MEM_FRACTION_STATIC="0.9"      # SGLang only
SERVER_EXTRA_ARGS=""

# ---- Benchmark config ------------------------------------------------------
NUM_REQUESTS=256
WARMUP_REQUESTS=4
DATASET="mixed"                # mixed or shared-prefix
PROMPTS_FILE=""                # optional TXT / JSON / JSONL prompt file
APPROX_PROMPT_TOKENS=1024
OUTPUT_TOKENS=256

# Client concurrency. auto picks the best value from a short sweep.
MAX_CONCURRENCY="auto"
AUTO_CONCURRENCY_CANDIDATES=""
AUTO_TUNE_REQUESTS=64

TEMPERATURE="0.0"
TOP_P="1.0"
REQUEST_TIMEOUT=300
STARTUP_TIMEOUT=1800
EXTRA_REQUEST_BODY='{}'

# Optional output override. Leave empty to write under results/benchmark/.
RESULTS_DIR=""
TAG=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(
    --backend "$BACKEND"
    --model "$MODEL"
    --host "$HOST"
    --tensor-parallel "$TENSOR_PARALLEL"
    --pipeline-parallel "$PIPELINE_PARALLEL"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --num-requests "$NUM_REQUESTS"
    --warmup-requests "$WARMUP_REQUESTS"
    --dataset "$DATASET"
    --approx-prompt-tokens "$APPROX_PROMPT_TOKENS"
    --output-tokens "$OUTPUT_TOKENS"
    --max-concurrency "$MAX_CONCURRENCY"
    --auto-tune-requests "$AUTO_TUNE_REQUESTS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --request-timeout "$REQUEST_TIMEOUT"
    --startup-timeout "$STARTUP_TIMEOUT"
    --extra-request-body "$EXTRA_REQUEST_BODY"
)

if [[ -n "$SERVED_MODEL_NAME" ]]; then
    ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$PORT" ]]; then
    ARGS+=(--port "$PORT")
fi

if [[ -n "$API_KEY" ]]; then
    ARGS+=(--api-key "$API_KEY")
fi

if [[ "$SKIP_LAUNCH" == "true" ]]; then
    ARGS+=(--skip-launch)
fi

if [[ "$KEEP_SERVER" == "true" ]]; then
    ARGS+=(--keep-server)
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
    ARGS+=(--trust-remote-code)
fi

if [[ -n "$SERVER_EXTRA_ARGS" ]]; then
    ARGS+=(--server-extra-args "$SERVER_EXTRA_ARGS")
fi

if [[ -n "$PROMPTS_FILE" ]]; then
    ARGS+=(--prompts-file "$PROMPTS_FILE")
fi

if [[ -n "$AUTO_CONCURRENCY_CANDIDATES" ]]; then
    ARGS+=(--auto-concurrency-candidates "$AUTO_CONCURRENCY_CANDIDATES")
fi

if [[ -n "$RESULTS_DIR" ]]; then
    ARGS+=(--results-dir "$RESULTS_DIR")
fi

if [[ -n "$TAG" ]]; then
    ARGS+=(--tag "$TAG")
fi

python3 "$SCRIPT_DIR/serve_speed_benchmark.py" "${ARGS[@]}"
