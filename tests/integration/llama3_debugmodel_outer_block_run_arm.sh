#!/bin/bash
# Run a single outer-block ablation arm on llama3_debugmodel.
#
# Usage:
#   ARM=<arm_id> SEED=<seed> NGPU=<n> bash tests/integration/llama3_debugmodel_outer_block_run_arm.sh
#
# Where <arm_id> is one of:
#   A    | bf16                                        : llama3_debugmodel_lpt_outer_block_bf16
#   B    | pts | pts_w16x16                            : llama3_debugmodel_lpt_outer_block_pts
#   Bp   | pts_w1x16                                   : llama3_debugmodel_lpt_outer_block_pts_w1x16
#   CN   | outer_w16x16  | nvidia_recipe                : llama3_debugmodel_lpt_outer_block_w16x16
#   C    | outer | outer_w1x16                         : llama3_debugmodel_lpt_outer_block
#   Calign | outer_align                               : llama3_debugmodel_lpt_outer_block_align
#   Csr   | outer_align_sr                             : llama3_debugmodel_lpt_outer_block_align_sr
#   Crht  | outer_align_sr_rht                         : llama3_debugmodel_lpt_outer_block_align_sr_rht
#
# Defaults:
#   ARM    = C
#   SEED   = 1234
#   NGPU   = 2
#   STEPS  = 2000  (override via STEPS=...)
#
# IMPORTANT: this script does NOT check GPU availability — caller should
# ensure GPUs are free before launching.  Default NGPU=2 keeps the resource
# footprint small so a multi-arm sweep can share a node.

set -ex

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/../.."

ARM=${ARM:-C}
SEED=${SEED:-1234}
NGPU=${NGPU:-2}
STEPS=${STEPS:-2000}

case "$ARM" in
    A|bf16)
        CONFIG=llama3_debugmodel_lpt_outer_block_bf16
        ;;
    B|pts|pts_w16x16)
        CONFIG=llama3_debugmodel_lpt_outer_block_pts
        ;;
    Bp|pts_w1x16)
        CONFIG=llama3_debugmodel_lpt_outer_block_pts_w1x16
        ;;
    CN|outer_w16x16|nvidia_recipe)
        CONFIG=llama3_debugmodel_lpt_outer_block_w16x16
        ;;
    C|outer|outer_w1x16)
        CONFIG=llama3_debugmodel_lpt_outer_block
        ;;
    Calign|outer_align)
        CONFIG=llama3_debugmodel_lpt_outer_block_align
        ;;
    Csr|outer_align_sr)
        CONFIG=llama3_debugmodel_lpt_outer_block_align_sr
        ;;
    Crht|outer_align_sr_rht)
        CONFIG=llama3_debugmodel_lpt_outer_block_align_sr_rht
        ;;
    *)
        echo "ERROR: unknown ARM=$ARM"
        exit 1
        ;;
esac

RUN_DIR="runs/outer_block/${ARM}/seed_${SEED}"
mkdir -p "$RUN_DIR"

# Stash a manifest so we can reconstruct what was run.
cat > "$RUN_DIR/manifest.json" <<EOF
{
  "arm": "$ARM",
  "config": "$CONFIG",
  "seed": $SEED,
  "ngpu": $NGPU,
  "steps": $STEPS,
  "git_sha": "$(git rev-parse --short HEAD)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

NGPU=$NGPU \
MODULE=llama3 \
CONFIG=$CONFIG \
./examples/run.sh \
    --training.steps "$STEPS" \
    --debug.seed "$SEED" \
    2>&1 | tee "$RUN_DIR/train.log"
