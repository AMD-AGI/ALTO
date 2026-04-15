#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG_FILE="alto/models/llama3/configs/llama3_8b.toml"
EXPORT_DTYPE="bfloat16"       # float16 | bfloat16 | float32
EXPORT_MODE="both"    # compressed | uncompressed | both

# ── Resolve paths ──────────────────────────────────────────────────────────
CONVERT_SCRIPT="alto/utils/exportation/convert_to_hf.py"

DUMP_FOLDER=$(python3 -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    print(tomllib.load(f).get('job', {}).get('dump_folder', './outputs'))
")
HF_DIR="${DUMP_FOLDER}/hf"
HF_COMPRESSED_DIR="${DUMP_FOLDER}/hf_compressed"

# ── Export ─────────────────────────────────────────────────────────────────
export_uncompressed() {
    echo ">> Exporting standard safetensors → ${HF_DIR}"
    python3 "$CONVERT_SCRIPT" "$CONFIG_FILE" \
        --export_dtype "$EXPORT_DTYPE" \
        --save_uncompressed \
        --disable_sparse_compression
}

export_compressed() {
    echo ">> Exporting compressed-tensors → ${HF_COMPRESSED_DIR}"
    python3 "$CONVERT_SCRIPT" "$CONFIG_FILE" \
        --export_dtype "$EXPORT_DTYPE"
    rm -rf "$HF_COMPRESSED_DIR"
    mv "$HF_DIR" "$HF_COMPRESSED_DIR"
}

case "$EXPORT_MODE" in
    uncompressed)  export_uncompressed ;;
    compressed)    export_compressed ;;
    both)          export_compressed; export_uncompressed ;;
    *)             echo "Error: EXPORT_MODE must be compressed|uncompressed|both" >&2; exit 1 ;;
esac
echo "Done."
