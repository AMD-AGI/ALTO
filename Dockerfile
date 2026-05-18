FROM rocm/pytorch-nightly:20260429082157-rocm7.2.2

RUN apt-get update && apt-get install -y \
    git-lfs \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN update-pciids

RUN pip install --no-cache-dir huggingface_hub "datasets>=3.6.0" \
    transformers tabulate wandb fsspec tyro "tokenizers>=0.15.0" safetensors \
    tensorboard pre-commit yapf pybind11 meson-python torchdata pytablewriter \
    "antlr4-python3-runtime==4.11.0" sympy math_verify more_itertools peft \
    accelerate pillow "numpy<2" opencv-python-headless scipy \
    numba huggingface-hub[cli,hf_transfer] "packaging>=24.2" \
    "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8" \
    protobuf-protoc-bin fmt && \
    pip install --no-cache-dir /opt/rocm/share/amd_smi

RUN cd /var/lib/jenkins && \
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness && \
    cd lm-evaluation-harness && \
    pip install -e .

COPY . /var/lib/jenkins/alto

RUN cd /var/lib/jenkins/alto/3rdparty/torchtitan && \
    pip install --no-build-isolation -e . && \
    cd /var/lib/jenkins/alto && \
    pip install -e .
