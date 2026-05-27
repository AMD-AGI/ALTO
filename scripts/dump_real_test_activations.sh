#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Convenience wrapper around ``scripts/dump_real_test_activations.py``.
# Iterates over multiple ``(bs, seq)`` shapes back-to-back, each as its
# own torchrun invocation (cleanest way to retarget ConfigManager's
# ``--training.local_batch_size`` / ``--training.seq_len``).
#
# Required env:
#   ALTO_DUMP_OUT     -- absolute output dir, e.g.
#                        /wekafs/zhitwang/test_data/real_activations/step5000_ckpt_20260526
# Optional env:
#   ALTO_DUMP_TAGS    -- comma-separated layer tags; defaults to all six
#   ALTO_DUMP_NUM_STEPS  -- captures per shape (default 3)
#   ALTO_DUMP_BS_SEQ  -- comma-separated ``BSxSEQ`` pairs, default ``2x2048,4x4096``
#   ALTO_DUMP_WITH_BACKWARD -- "1" to also capture real dy (needs PM-firmware-bug
#                              fixed first; default "0", forward only)
#   ALTO_DUMP_MODULE / ALTO_DUMP_CONFIG -- forwarded to ConfigManager
#                                          (defaults: gpt_oss / gpt_oss_20b)
#   NGPU              -- torchrun nproc_per_node (default 1)
#
# Example:
#   ALTO_DUMP_OUT=/wekafs/zhitwang/test_data/real_activations/step5000_ckpt_20260526 \
#   ALTO_DUMP_BS_SEQ=2x2048,4x4096 \
#   bash scripts/dump_real_test_activations.sh

set -euo pipefail

: "${ALTO_DUMP_OUT:?must set ALTO_DUMP_OUT}"
: "${ALTO_DUMP_BS_SEQ:=2x2048,4x4096}"
: "${ALTO_DUMP_NUM_STEPS:=3}"
: "${ALTO_DUMP_TAGS:=dense_early,dense_mid,dense_late,grouped_early,grouped_mid,grouped_late}"
: "${ALTO_DUMP_WITH_BACKWARD:=0}"
: "${ALTO_DUMP_MODULE:=gpt_oss}"
: "${ALTO_DUMP_CONFIG:=gpt_oss_20b}"
: "${NGPU:=1}"

# HSA_NO_SCRATCH_RECLAIM is the canonical knob the GPT-OSS / NVFP4
# pod uses to keep MI300X queues from re-claiming scratch under HSA;
# pass through if the caller already set it, otherwise default to 1.
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

mkdir -p "${ALTO_DUMP_OUT}"

IFS=',' read -ra PAIRS <<< "${ALTO_DUMP_BS_SEQ}"
for pair in "${PAIRS[@]}"; do
    BS="${pair%x*}"
    SEQ="${pair#*x}"
    echo ""
    echo "============================================================"
    echo "[dump.sh] bs=${BS} seq=${SEQ} tags=${ALTO_DUMP_TAGS} steps=${ALTO_DUMP_NUM_STEPS}"
    echo "[dump.sh] with_backward=${ALTO_DUMP_WITH_BACKWARD} out=${ALTO_DUMP_OUT}"
    echo "============================================================"

    ALTO_DUMP_OUT="${ALTO_DUMP_OUT}" \
    ALTO_DUMP_TAGS="${ALTO_DUMP_TAGS}" \
    ALTO_DUMP_NUM_STEPS="${ALTO_DUMP_NUM_STEPS}" \
    ALTO_DUMP_WITH_BACKWARD="${ALTO_DUMP_WITH_BACKWARD}" \
    torchrun --nproc_per_node="${NGPU}" \
        --rdzv_backend=c10d --rdzv_endpoint="localhost:0" \
        -m scripts.dump_real_test_activations \
        --module "${ALTO_DUMP_MODULE}" \
        --config "${ALTO_DUMP_CONFIG}" \
        --training.local_batch_size "${BS}" \
        --training.seq_len "${SEQ}" \
        --training.steps 0 \
        --validator.enable false \
        --checkpoint.enable true \
        "$@"
done

echo ""
echo "[dump.sh] all shapes done; bundles under ${ALTO_DUMP_OUT}"
