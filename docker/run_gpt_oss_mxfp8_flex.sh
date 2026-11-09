#!/usr/bin/env bash
# GPT-OSS with MXFP8 Flex attention in the PyTorch 2.14 / ROCm 10.1 image.
#
#   ./docker/run_gpt_oss_mxfp8_flex.sh test          # mxfp8 attention kernel + LPT Flex unit tests
#   ./docker/run_gpt_oss_mxfp8_flex.sh debug-bf16    # 2-GPU debugmodel, BF16 Flex
#   ./docker/run_gpt_oss_mxfp8_flex.sh debug-mxfp8   # 2-GPU debugmodel, MXFP8 Flex
#   ./docker/run_gpt_oss_mxfp8_flex.sh bf16-flex     # 8-GPU 20b_megatron baseline
#   ./docker/run_gpt_oss_mxfp8_flex.sh mxfp8-attn    # 8-GPU 20b_megatron, MXFP8 attention
#
# Extra arguments go to examples/run.sh, e.g. `mxfp8-attn --training.steps 100`.
# Every non-shell run tees its output to logs/ in the repo.
# The 20b modes need HF_HUB_DIR (holds models--openai--gpt-oss-20b) and
# MEGATRON_DATA_DIR (holds the c4 .bin/.idx files); they are mounted where
# alto/models/gpt_oss/config_registry.py expects them. The repo is mounted over
# the image's editable /opt/alto install, so no package install is needed.
set -euo pipefail

IMAGE="${IMAGE:-wanghanthu/alto:pytorch2.14.0-rocm10.1.0}"
DOCKER="${DOCKER:-sudo docker}"
# On a borrowed node the login user may not own the repo; run as the owner.
RUN_AS="${RUN_AS:-$(id -u):$(id -g)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-shell}"
shift || true
EXTRA_ARGS="$*"

case "${MODE}" in
    shell) TRAIN_COMMAND="exec bash" ;;
    test) TRAIN_COMMAND="python3 -m pytest -q -rfE -p no:cacheprovider $(printf 'tests/unittest/mxfp8/%s ' test_mxfp8_attention.py test_mxfp8_attention_reference.py test_mxfp8_blockmask.py test_mxfp8_lpt_flex_attention.py) ${EXTRA_ARGS}" ;;
    debug-bf16) TRAIN_COMMAND="NGPU=2 MODULE=gpt_oss CONFIG=gpt_oss_debugmodel_flex ./examples/run.sh --parallelism.expert_parallel_degree 2 ${EXTRA_ARGS}" ;;
    debug-mxfp8) TRAIN_COMMAND="NGPU=2 MODULE=gpt_oss CONFIG=gpt_oss_debugmodel_mxfp8_attn ./examples/run.sh --parallelism.expert_parallel_degree 2 ${EXTRA_ARGS}" ;;
    bf16-flex) TRAIN_COMMAND="NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_pretrain_flex ./examples/run.sh ${EXTRA_ARGS}" ;;
    mxfp8-attn) TRAIN_COMMAND="NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_mxfp8_attn ./examples/run.sh ${EXTRA_ARGS}" ;;
    *)
        echo "Usage: $0 {shell|test|debug-bf16|debug-mxfp8|bf16-flex|mxfp8-attn} [run.sh args...]" >&2
        exit 2
        ;;
esac
# Borrowed nodes can be reclaimed at any time, so logs live in the repo on shared storage.
if [[ "${MODE}" != shell ]]; then
    LOG_FILE="logs/gpt_oss_${MODE//-/_}_$(date -u +%Y%m%dT%H%M%SZ).log"
    echo "Logging to ${ALTO_DIR}/${LOG_FILE}"
    TRAIN_COMMAND="mkdir -p logs && ( ${TRAIN_COMMAND} ) 2>&1 | tee ${LOG_FILE}"
fi

DATA_MOUNTS=()
if [[ "${MODE}" == bf16-flex || "${MODE}" == mxfp8-attn ]]; then
    : "${HF_HUB_DIR:?set HF_HUB_DIR to the host dir containing models--openai--gpt-oss-20b}"
    : "${MEGATRON_DATA_DIR:?set MEGATRON_DATA_DIR to the host dir containing c4-train.en_6_text_document.idx}"
    DATA_MOUNTS=(
        -v "${HF_HUB_DIR}:/huggingface/hub:ro"
        -v "${MEGATRON_DATA_DIR}:/workspace/workspace/megatron_dataset/data:ro"
    )
fi

WANDB_MOUNT=()
if [[ "${WANDB_MODE:-online}" != offline && -f "${HOME}/.netrc" ]]; then
    WANDB_MOUNT=(-v "${HOME}/.netrc:/tmp/wandb.netrc:ro")
fi

${DOCKER} run --rm --init \
    --ulimit core=0 \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    --network host \
    --ipc=host \
    --shm-size 8G \
    --user "${RUN_AS}" \
    -e HOME=/tmp/alto-home \
    -e WANDB_MODE="${WANDB_MODE:-online}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-alto-gpt-oss-20b-mxfp8-flex}" \
    --workdir /opt/alto \
    -v "${ALTO_DIR}:/opt/alto" \
    "${DATA_MOUNTS[@]}" \
    "${WANDB_MOUNT[@]}" \
    "${IMAGE}" \
    bash -lc '
set -euo pipefail
mkdir -p "$HOME"
if [[ -f /tmp/wandb.netrc ]]; then
    cp /tmp/wandb.netrc "$HOME/.netrc"
    chmod 600 "$HOME/.netrc"
fi
'"${TRAIN_COMMAND}"
