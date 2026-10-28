FROM rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/alto

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Keep the image's ROCm PyTorch build. Later pip installs are constrained to it.
RUN python3 - <<'PY' > /tmp/rocm-constraints.txt
import importlib.metadata as metadata

for distribution in ("torch", "triton", "torchvision", "torchaudio"):
    try:
        print(f"{distribution}=={metadata.version(distribution)}")
    except metadata.PackageNotFoundError:
        pass
PY

COPY 3rdparty/torchtitan/requirements.txt /tmp/torchtitan-requirements.txt

RUN pip install -c /tmp/rocm-constraints.txt -r /tmp/torchtitan-requirements.txt \
    && pip install -c /tmp/rocm-constraints.txt \
        "poetry-core>=1.0.0" \
        "poetry-dynamic-versioning>=1.0.0,<2.0.0" \
        torchao \
        safetensors \
        compressed_tensors \
        pydantic \
        yapf \
        lm_eval \
        easydict \
        loguru \
    && rm /tmp/torchtitan-requirements.txt

COPY . /opt/alto

RUN pip install --no-build-isolation --no-deps -e /opt/alto/3rdparty/torchtitan \
    && pip install --no-build-isolation --no-deps -e /opt/alto

WORKDIR /opt/alto
CMD ["/bin/bash"]
