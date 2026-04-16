#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONWARNINGS="ignore"

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH="/wekafs/guanchen/Model-Optimizer/outputs/hf_compressed"
DTYPE="bfloat16"
BATCH_SIZE="auto"
OUTPUT_DIR="/wekafs/guanchen/Model-Optimizer/outputs/eval_results_vllm"
# Choose benchmark suite: "v1" or "v2" or "both"
BENCHMARK="v1"
# vllm tensor-parallel: number of GPUs for model sharding
TP_SIZE=1
# Max model sequence length (reduce if OOM)
MAX_MODEL_LEN=4096

# ── Environment ────────────────────────────────────────────────────────────
export HF_DATASETS_CACHE=/wekafs/guanchen/datasets/hf_cache

mkdir -p "$OUTPUT_DIR"

# ── Build model_args ───────────────────────────────────────────────────────
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},tensor_parallel_size=${TP_SIZE},max_model_len=${MAX_MODEL_LEN}"

# ── Open LLM Leaderboard v1 (archived Jun 2024) ───────────────────────────
run_v1() {
    echo ">> Running Open LLM Leaderboard v1 (vllm backend)"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks arc_challenge --num_fewshot 25 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_arc_challenge.json"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks hellaswag --num_fewshot 10 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_hellaswag.json"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks truthfulqa_mc2 --num_fewshot 0 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_truthfulqa.json"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks winogrande --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_winogrande.json"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks mmlu --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_mmlu.json"

    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks gsm8k --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v1_gsm8k.json"
}

# ── Open LLM Leaderboard v2 (current) ─────────────────────────────────────
run_v2() {
    echo ">> Running Open LLM Leaderboard v2 (vllm backend)"
    python3 -m lm_eval --model vllm --model_args "$MODEL_ARGS" \
        --tasks leaderboard \
        --batch_size "$BATCH_SIZE" \
        --output_path "${OUTPUT_DIR}/v2.json"
}

# ── Main ───────────────────────────────────────────────────────────────────
case "$BENCHMARK" in
    v1)   run_v1 ;;
    v2)   run_v2 ;;
    both) run_v1; run_v2 ;;
    *)    echo "Error: BENCHMARK must be v1|v2|both" >&2; exit 1 ;;
esac
echo "Done. Results saved to ${OUTPUT_DIR}/"
