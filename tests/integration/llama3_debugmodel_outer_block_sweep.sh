#!/bin/bash
# Drive the §4 / §6.7 outer-block ablation sweep.
#
# Usage:
#   bash tests/integration/llama3_debugmodel_outer_block_sweep.sh [phase]
#
# Phases:
#   phase0  : (A) BF16 baseline, 1 seed                   (~15 min on 2 GPUs)
#   phase1  : (B), (C-N), (C) fail-fast, 1 seed each      (~3-4 GPU·h)
#   phase2  : (A/B/B'/C-N/C/C+align/C+sr/C+rht) × 3 seeds (~13-16 GPU·h)
#   all     : run phase0 → phase1 → phase2 sequentially
#
# Defaults: NGPU=2 (small footprint so the sweep can share a busy node).
#
# !!! Before launching: verify GPUs are free.  This script blocks until each
# !!! arm finishes; killing it mid-run will not corrupt later arms.

set -e

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/../.."
PHASE=${1:-phase0}
NGPU=${NGPU:-2}
STEPS=${STEPS:-2000}

run_arm() {
    local arm=$1
    local seed=$2
    echo "=== ARM=$arm SEED=$seed ==="
    ARM="$arm" SEED="$seed" NGPU="$NGPU" STEPS="$STEPS" \
        bash tests/integration/llama3_debugmodel_outer_block_run_arm.sh
}

case "$PHASE" in
    phase0)
        run_arm A 1234
        ;;
    phase1)
        run_arm A 1234   # sanity check; reuses prior run cache if you'd rather skip
        run_arm B 1234
        run_arm CN 1234
        run_arm C 1234
        ;;
    phase2)
        for seed in 1234 2024 4242; do
            for arm in A B Bp CN C Calign Csr Crht; do
                run_arm "$arm" "$seed"
            done
        done
        ;;
    all)
        bash tests/integration/llama3_debugmodel_outer_block_sweep.sh phase0
        bash tests/integration/llama3_debugmodel_outer_block_sweep.sh phase1
        bash tests/integration/llama3_debugmodel_outer_block_sweep.sh phase2
        ;;
    *)
        echo "ERROR: unknown phase=$PHASE; expected one of phase0|phase1|phase2|all"
        exit 1
        ;;
esac
