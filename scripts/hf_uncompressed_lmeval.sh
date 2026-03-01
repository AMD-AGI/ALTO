#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONWARNINGS="ignore"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH="/wekafs/guanchen/Model-Optimizer/outputs/hf"
DTYPE="bfloat16"
BATCH_SIZE="auto"
OUTPUT_DIR="/wekafs/guanchen/Model-Optimizer/outputs/eval_results"

# Choose benchmark suite: "v1" or "v2" or "both"
BENCHMARK="both"

# Multi-GPU: set number of GPUs (1 = single GPU, >1 = auto device_map)
NUM_GPUS=2

# ── Environment ────────────────────────────────────────────────────────────
export HF_DATASETS_CACHE=/wekafs/guanchen/datasets/hf_cache

mkdir -p "$OUTPUT_DIR"

# ── Build model_args ───────────────────────────────────────────────────────
if [[ "$NUM_GPUS" -gt 1 ]]; then
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE},parallelize=True"
    DEVICE_FLAG=""
else
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE}"
    DEVICE_FLAG="--device cuda:0"
fi

# ── Open LLM Leaderboard v1 (archived Jun 2024) ───────────────────────────
#   arc_challenge (25-shot), hellaswag (10-shot), mmlu (5-shot),
#   truthfulqa_mc2 (0-shot), winogrande (5-shot), gsm8k (5-shot)
run_v1() {
    echo ">> Running Open LLM Leaderboard v1 benchmarks"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks arc_challenge --num_fewshot 25 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_arc_challenge.json"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks hellaswag --num_fewshot 10 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_hellaswag.json"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks truthfulqa_mc2 --num_fewshot 0 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_truthfulqa.json"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks winogrande --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_winogrande.json"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks mmlu --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_mmlu.json"

    python3 -m lm_eval --model hf --model_args "$MODEL_ARGS" \
        --tasks gsm8k --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" $DEVICE_FLAG \
        --output_path "${OUTPUT_DIR}/v1_gsm8k.json"
}

# ── Open LLM Leaderboard v2 (current) ─────────────────────────────────────
#   IFEval (0-shot), BBH (3-shot), MATH Lvl5 (4-shot),
#   GPQA (0-shot), MuSR (0-shot), MMLU-PRO (5-shot)
run_v2() {
    echo ">> Running Open LLM Leaderboard v2 benchmarks"
    python3 -m lm_eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks leaderboard \
        --batch_size "$BATCH_SIZE" \
        $DEVICE_FLAG \
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
