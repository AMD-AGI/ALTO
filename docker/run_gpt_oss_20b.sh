#!/usr/bin/env bash
set -euo pipefail

IMAGE="wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_MODELS_DIR="${HF_MODELS_DIR:-${ALTO_DIR}/../hf_models}"
GID_RENDER="$(getent group render | cut -d: -f3)"
GID_VIDEO="$(getent group video | cut -d: -f3)"
WANDB_TEAM="${WANDB_TEAM:-yue-sun2-amd}"
WANDB_PROJECT="${WANDB_PROJECT:-alto-gpt-oss-20b-mxfp8-attn}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpt-oss-20b-mxfp8-c4-ablation}"
MODE="${1:-shell}"
WANDB_NETRC="${HOME}/.netrc"
SUITE="bf16 attn"
IDLE_POLLS=5
POLL_SECONDS=60

busy_gpus() {
    rocm-smi --showuse --showmemuse --json | python3 -c '
import json, sys
cards = json.load(sys.stdin)
print(" ".join(name for name, card in cards.items()
               if int(card["GPU Memory Allocated (VRAM%)"]) > 1 or int(card["GPU use (%)"]) > 10))'
}

# Someone else's back-to-back jobs leave short gaps, so the node must stay idle
# for IDLE_POLLS consecutive polls before a run takes all 8 GPUs.
wait_for_idle_gpus() {
    local idle=0 busy last="unset"
    while true; do
        busy="$(busy_gpus)"
        if [[ "${busy}" != "${last}" ]]; then
            echo "[$(date -u +%FT%TZ)] busy GPUs: ${busy:-none}"
            last="${busy}"
        fi
        if [[ -z "${busy}" ]]; then
            idle=$((idle + 1))
        else
            idle=0
        fi
        if (( idle >= IDLE_POLLS )); then
            return
        fi
        sleep "${POLL_SECONDS}"
    done
}

config_of() {
    case "$1" in
        bf16) echo gpt_oss_20b_c4 ;;
        attn) echo gpt_oss_20b_c4_mxfp8_attn ;;
        linear) echo gpt_oss_20b_c4_mxfp8_linear ;;
        linear-attn) echo gpt_oss_20b_c4_mxfp8_linear_attn ;;
        *) echo "Unknown group: $1 (expected bf16|attn|linear|linear-attn)" >&2; exit 2 ;;
    esac
}

train_command() {
    echo "NGPU=8 MODULE=gpt_oss CONFIG=$1 WANDB_RUN_NAME=$2 ./examples/run.sh ${3:-}"
}

case "${MODE}" in
    shell)
        TRAIN_COMMAND="exec bash"
        ;;
    bf16|attn|linear|linear-attn)
        CONFIG_NAME="$(config_of "${MODE}")"
        TRAIN_COMMAND="$(train_command "${CONFIG_NAME}" "${WANDB_RUN_NAME:-gpt-oss-20b-${MODE}}" "${*:2}")"
        ;;
    smoke)
        GROUP="${2:-linear-attn}"
        CONFIG_NAME="$(config_of "${GROUP}")"
        TRAIN_COMMAND="$(train_command "${CONFIG_NAME}" "gpt-oss-20b-${GROUP}-smoke" \
            "--training.steps 10 --validator.no-enable --checkpoint.no-enable --metrics.no-enable-wandb --dump_folder gpt_oss_20b-c4-smoke-outputs")"
        ;;
    suite)
        for group in ${SUITE}; do
            "${BASH_SOURCE[0]}" "${group}"
        done
        exit 0
        ;;
    compare-1k)
        for group in ${SUITE}; do
            WANDB_RUN_NAME="gpt-oss-20b-${group}-1k" "${BASH_SOURCE[0]}" "${group}" \
                --training.steps 1000 --validator.freq 250 --checkpoint.no-enable \
                --dump_folder "gpt_oss_20b-c4-${group}-1k-outputs"
        done
        exit 0
        ;;
    *)
        echo "Usage: $0 {shell|<group> [torchtitan flags...]|smoke [group]|suite|compare-1k}" >&2
        echo "  group: bf16|attn|linear|linear-attn" >&2
        exit 2
        ;;
esac

if [[ "${MODE}" != "shell" ]]; then
    if [[ ! -f "${WANDB_NETRC}" ]] || ! python3 -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'; then
        echo "W&B authentication is required. Run 'wandb login' on the host." >&2
        exit 1
    fi
fi

if [[ ! -f "${HF_MODELS_DIR}/gpt-oss-20b/tokenizer.json" ]]; then
    echo "Missing tokenizer assets in ${HF_MODELS_DIR}/gpt-oss-20b" >&2
    echo "Copy the gpt-oss-20b asset directory there, or point HF_MODELS_DIR at it. See docker/README.md." >&2
    exit 1
fi
HF_MODELS_DIR="$(cd "${HF_MODELS_DIR}" && pwd)"

if [[ ! -f "${ALTO_DIR}/3rdparty/torchtitan/pyproject.toml" ]]; then
    git -C "${ALTO_DIR}" submodule update --init --recursive
fi

if [[ "${MODE}" != "shell" ]]; then
    wait_for_idle_gpus
fi

sudo docker run --rm --init \
    --ulimit core=0 \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add "${GID_RENDER}" \
    --group-add "${GID_VIDEO}" \
    --network host \
    --ipc=host \
    --shm-size 8G \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/alto-home \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e WANDB_TEAM="${WANDB_TEAM}" \
    -e WANDB_PROJECT="${WANDB_PROJECT}" \
    -e WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" \
    -e TOKENIZERS_PARALLELISM=false \
    -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
    --workdir /workspace/ALTO-mxfp8-attn \
    -v "${ALTO_DIR}:/workspace/ALTO-mxfp8-attn" \
    -v "${HF_MODELS_DIR}:/workspace/hf_models:ro" \
    -v "${WANDB_NETRC}:/tmp/wandb.netrc:ro" \
    "${IMAGE}" \
    bash -lc '
set -euo pipefail
mkdir -p "$HOME"
cp /tmp/wandb.netrc "$HOME/.netrc"
chmod 600 "$HOME/.netrc"

if [[ ! -x .venv/bin/python ]]; then
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install --no-build-isolation -e 3rdparty/torchtitan
    pip install -e .
    pip install --force-reinstall --no-deps torchao==0.17.0
fi

source .venv/bin/activate
printf "%s\n" \
    "#!/usr/bin/env bash" \
    "exec \"\$(dirname \"\$0\")/python\" -m torch.distributed.run \"\$@\"" \
    > .venv/bin/torchrun
chmod +x .venv/bin/torchrun
'"${TRAIN_COMMAND}"
