#!/usr/bin/env bash
# GPT-OSS with MXFP8 Flex attention in the PyTorch 2.14 / ROCm 10.1 image.
#
#   ./docker/run_gpt_oss_mxfp8_flex.sh test          # mxfp8 kernel + LPT Flex unit tests
#   ./docker/run_gpt_oss_mxfp8_flex.sh debug-bf16    # 2-GPU debugmodel, BF16 Flex
#   ./docker/run_gpt_oss_mxfp8_flex.sh debug-mxfp8   # 2-GPU debugmodel, MXFP8 Flex
#   ./docker/run_gpt_oss_mxfp8_flex.sh bf16-flex     # 8-GPU 20b_megatron baseline
#   ./docker/run_gpt_oss_mxfp8_flex.sh mxfp8-attn    # 8-GPU 20b_megatron, MXFP8 attention
#
# Extra arguments go to examples/run.sh, e.g. `mxfp8-attn --training.steps 100`.
# The 20b modes need HF_HUB_DIR (holds models--openai--gpt-oss-20b) and
# MEGATRON_DATA_DIR (holds the c4 .bin/.idx files); they are mounted where
# alto/models/gpt_oss/config_registry.py expects them.
set -euo pipefail

IMAGE="${IMAGE:-wanghanthu/alto:pytorch2.14.0-rocm10.1.0}"
DOCKER="${DOCKER:-sudo docker}"
# On a borrowed node the login user may not own the repo; run as the owner.
RUN_AS="${RUN_AS:-$(id -u):$(id -g)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# The venv is tied to the image's torch; sharing one across images mixes wheels.
VENV=".venv-${IMAGE##*:}"
MODE="${1:-shell}"
shift || true
EXTRA_ARGS="$*"

case "${MODE}" in
    shell) TRAIN_COMMAND="exec bash" ;;
    test) TRAIN_COMMAND="python -m pytest -q tests/unittest/mxfp8 ${EXTRA_ARGS}" ;;
    debug-bf16) TRAIN_COMMAND="NGPU=2 MODULE=gpt_oss CONFIG=gpt_oss_debugmodel_flex ./examples/run.sh --parallelism.expert_parallel_degree 2 ${EXTRA_ARGS}" ;;
    debug-mxfp8) TRAIN_COMMAND="NGPU=2 MODULE=gpt_oss CONFIG=gpt_oss_debugmodel_mxfp8_attn ./examples/run.sh --parallelism.expert_parallel_degree 2 ${EXTRA_ARGS}" ;;
    bf16-flex) TRAIN_COMMAND="NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_pretrain_flex ./examples/run.sh ${EXTRA_ARGS}" ;;
    mxfp8-attn) TRAIN_COMMAND="NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_mxfp8_attn ./examples/run.sh ${EXTRA_ARGS}" ;;
    *)
        echo "Usage: $0 {shell|test|debug-bf16|debug-mxfp8|bf16-flex|mxfp8-attn} [run.sh args...]" >&2
        exit 2
        ;;
esac

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
if [[ -f "${HOME}/.netrc" ]]; then
    WANDB_MOUNT=(-v "${HOME}/.netrc:/tmp/wandb.netrc:ro")
fi

${DOCKER} run --rm --init \
    --ulimit core=0 \
    --privileged \
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
    -e VENV="${VENV}" \
    --workdir /workspace/ALTO \
    -v "${ALTO_DIR}:/workspace/ALTO" \
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
if [[ ! -x "$VENV/bin/python" ]]; then
    python -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --no-build-isolation -e 3rdparty/torchtitan
    "$VENV/bin/pip" install -e .
    # The image torchrun runs the system python, which cannot see the editable installs.
    printf "%s\n" "#!/usr/bin/env bash" "exec \"\$(dirname \"\$0\")/python\" -m torch.distributed.run \"\$@\"" \
        > "$VENV/bin/torchrun"
    chmod +x "$VENV/bin/torchrun"
fi
source "$VENV/bin/activate"
'"${TRAIN_COMMAND}"
