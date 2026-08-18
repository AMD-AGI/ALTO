#!/usr/bin/env bash
set -euo pipefail

IMAGE="wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ALTO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_MODELS_DIR="${HF_MODELS_DIR:-$(cd "${ALTO_DIR}/../hf_models" && pwd)}"
GID_RENDER="$(getent group render | cut -d: -f3)"
GID_VIDEO="$(getent group video | cut -d: -f3)"
WANDB_TEAM="${WANDB_TEAM:-yue-sun2-amd}"
WANDB_PROJECT="${WANDB_PROJECT:-alto-llama3-8b-mxfp8-ablation}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-llama3-8b-mxfp8-c4-fsdp-ablation}"
MODE="${1:-shell}"
WANDB_NETRC="${HOME}/.netrc"

case "${MODE}" in
    shell)
        TRAIN_COMMAND="exec bash"
        ;;
    smoke)
        TRAIN_COMMAND='NGPU=8 MODULE=llama3 CONFIG=llama3_8b_light_mxfp8_attn WANDB_RUN_NAME=llama3-8b-mxfp8-attn-smoke ./examples/run.sh --training.steps 1 --validator.no-enable'
        ;;
    bf16)
        TRAIN_COMMAND='NGPU=8 MODULE=llama3 CONFIG=llama3_8b WANDB_RUN_NAME=llama3-8b-bf16 ./examples/run.sh'
        ;;
    bf16-gbs4)
        TRAIN_COMMAND='NGPU=2 MODULE=llama3 CONFIG=llama3_8b WANDB_RUN_NAME=llama3-8b-bf16-gbs4 ./examples/run.sh --training.local_batch_size 2 --dump_folder llama3_8b-c4-bf16-gbs4-outputs'
        ;;
    attn)
        TRAIN_COMMAND='NGPU=8 MODULE=llama3 CONFIG=llama3_8b_light_mxfp8_attn WANDB_RUN_NAME=llama3-8b-mxfp8-attn ./examples/run.sh'
        ;;
    linear-attn)
        TRAIN_COMMAND='NGPU=8 MODULE=llama3 CONFIG=llama3_8b_light_mxfp8_linear_attn WANDB_RUN_NAME=llama3-8b-mxfp8-linear-attn ./examples/run.sh'
        ;;
    linear)
        TRAIN_COMMAND='NGPU=8 MODULE=llama3 CONFIG=llama3_8b_light_mxfp8_linear WANDB_RUN_NAME=llama3-8b-mxfp8-linear ./examples/run.sh'
        ;;
    suite)
        for experiment in bf16 attn linear-attn; do
            "${BASH_SOURCE[0]}" "${experiment}"
        done
        exit 0
        ;;
    *)
        echo "Usage: $0 {shell|smoke|bf16|bf16-gbs4|attn|linear-attn|linear|suite}" >&2
        exit 2
        ;;
esac

if [[ "${MODE}" != "shell" ]]; then
    if [[ ! -f "${WANDB_NETRC}" ]] || ! python3 -c 'import netrc; assert netrc.netrc().authenticators("api.wandb.ai")'; then
        echo "W&B authentication is required. Run 'wandb login' on the host." >&2
        exit 1
    fi
fi

if [[ ! -f "${HF_MODELS_DIR}/llama3-8b/tokenizer.json" ]]; then
    echo "Missing tokenizer assets in ${HF_MODELS_DIR}/llama3-8b" >&2
    exit 1
fi

if [[ ! -f "${ALTO_DIR}/3rdparty/torchtitan/pyproject.toml" ]]; then
    git -C "${ALTO_DIR}" submodule update --init --recursive
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
    -e WANDB_RUN_NAME \
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
