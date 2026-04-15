#!/usr/bin/env bash
set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH="/wekafs/guanchen/Model-Optimizer/outputs/hf"
DTYPE="bfloat16"
SEQLEN=4096
BATCH_SIZE=1
DATASETS="c4 wikitext2 ptb"
OUTPUT_DIR="/wekafs/guanchen/Model-Optimizer/outputs/ppl_results"
NUM_GPUS=1

# ── Environment ────────────────────────────────────────────────────────────
export HF_DATASETS_CACHE=/wekafs/guanchen/datasets/hf_cache
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR"

# ── Run ────────────────────────────────────────────────────────────────────
python3 -m alto.utils.evaluation.eval_ppl \
    --model_path "$MODEL_PATH" \
    --datasets $DATASETS \
    --seqlen "$SEQLEN" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --num_gpus "$NUM_GPUS" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --output_path "${OUTPUT_DIR}/ppl_seq${SEQLEN}.json"
