# ALTO: Advanced Low-precision Training and Optimization

ALTO is a Python library for low-precision model training and optimization, built on top of the [TorchTitan fork](https://github.com/AMD-AGI/torchtitan-amd/tree/dev/alto). It ships Triton-backed low-precision kernels (MXFP4, NVFP4, block-scaled FP8, and related utilities) and a configurable stack of **modifiers**—low-precision training (LPT)—wired into TorchTitan through a model-converter pipeline.

## Contents

- [Features](#features)
- [Supported models](#supported-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Export and evaluation](#export-and-evaluation)

## Features

### Low-precision training (LPT)

Training-oriented kernels and schemes include:

- **[Blockwise FP8](alto/kernels/blockwise_fp8)** — linear, grouped GEMM, and FlashAttention.
- **[MXFP4](alto/kernels/fp4/mxfp4)** — linear, grouped GEMM, and FlashAttention.
- **[NVFP4](alto/kernels/fp4/nvfp4)** — linear and grouped GEMM, using an E4M3 inner-block scale with an optional two-level (tensorwise) outer scale.

Techniques used to narrow the gap versus BF16 include:

- 2D block quantization
- Randomized Hadamard Transform (RHT)
- Stochastic Rounding (SR)
- Differential Gradient Estimation (DGE)

### Modifiers

Recipes can combine multiple stages under `alto/modifiers/`, e.g. Low-Precision Training (LPT) Modifier.

Recipe YAML follows the same general shape as [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/bede809f388aaeb1438a4d692d2d79109f9357dc); use the configs under `alto/models/*/configs/` as concrete templates.

## Supported models

| Model | Integration notes |
|--------|-------------------|
| **[Llama 3](alto/models/llama3)** | * Extended [state dict adapter](alto/models/llama3/state_dict_adapter.py) for Hugging Face Safetensors with observer/modifier state; <br/>* [patcher](alto/models/patcher.py) keeps query/key projections in the Transformers layout for RoPE; <br/> * Config registry hooks for TorchTitan. |
| **[DeepSeek V3](alto/models/deepseek_v3)** | Config registry hooks for TorchTitan. |
| **[GPT-OSS](alto/models/gpt_oss)** | Config registry hooks for TorchTitan. |

## Requirements

- **Python** 3.9+
- **PyTorch** 2.9+ (see `pyproject.toml` for the full dependency set: `torchao`, `safetensors`, `compressed_tensors`, `lm_eval`, etc.)
- **GPU** — training paths expect ROCm/CUDA-capable hardware; see TorchTitan documentation for parallel layout details.

## Installation

Clone the repository **with submodules** so the vendored TorchTitan tree is present:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/ALTO.git
cd ALTO
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the TorchTitan tree shipped under `3rdparty/torchtitan`, then install ALTO in editable mode:

```bash
pip install --no-build-isolation -e 3rdparty/torchtitan
pip install -e .
```

## Usage

### Wire a recipe into TorchTitan

1. Author or copy a **recipe YAML** (see existing files under `alto/models/<model>/configs/`).
2. Register a TorchTitan config that attaches ALTO’s converter to `model_converters`.

Example (Llama 3 registry pattern):

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer
from alto.components.converter import ModelOptConverter

config.model_converters = ModelConvertersContainer.Config(
    converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml"),
    ],
)
```

See [`alto/models/llama3/config_registry.py`](alto/models/llama3/config_registry.py) for full trainer configs.

### Launch training

From the repository root, the shared launcher wraps `torchrun` and `python -m alto.train`:

```bash
NGPU=8 MODULE=llama3 CONFIG=your_config_name ./examples/run.sh
```

Environment variables (see [`examples/run.sh`](examples/run.sh)):

| Variable | Role |
|----------|------|
| `NGPU` | Processes per node (default `8`). |
| `MODULE` | TorchTitan module name (`llama3`, `gpt_oss`, …). |
| `CONFIG` | Registered config function name. |
| `TRAIN_FILE` | Python module for training entrypoint (default `alto.train`). |
| `COMM_MODE` | Optional: `fake_backend` or `local_tensor` for config checks / single-GPU debugging. |

## Examples

### GPT-OSS 20B — MXFP4 Training

- Recipe: [`alto/models/gpt_oss/configs/lpt_recipe.yaml`](alto/models/gpt_oss/configs/lpt_recipe.yaml)

  uses `LowPrecisionTrainingModifier` with `scheme: "mxfp4"`.

- Config: [`gpt_oss_20b_lpt`](alto/models/gpt_oss/config_registry.py) in the GPT-OSS registry.

- Run:

  ```bash
  NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_lpt ./examples/run.sh
  ```

Illustrative recipe fragment:

```yaml
training_stage:
  lpt_modifiers:
    LowPrecisionTrainingModifier:
      scheme: "mxfp4"          # also supports "nvfp4" (plus "mxfp8_e4m3" / "mxfp8_e5m2")
      targets: ["Linear", "GptOssGroupedExperts"]
      ignore: ["output", "re:.*\\.router\\.gate"]
      use_2dblock_x: false
      use_2dblock_w: true
      use_hadamard: true
      use_sr_grad: true
      use_dge: false
      two_level_scaling: none  # use "tensorwise" to enable NVFP4's outer scale
```

To train the same model with NVFP4 instead, set `scheme: "nvfp4"` and `two_level_scaling: "tensorwise"`. NVFP4 kernels require the environment variable `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` at launch.

## Export and evaluation

TorchTitan typically saves checkpoints in PyTorch DCP format; you can convert to Hugging Face Safetensors and run lm-eval tasks with the bundled export utility:

```bash
python ./alto/utils/exportation/export.py \
  llama3 llama3_1b_opt \
  --tasks wikitext
```

## Project links

- **Homepage** [github.com/AMD-AGI/ALTO](https://github.com/AMD-AGI/ALTO)
- **TorchTitan submodule:** [github.com/AMD-AGI/torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd/tree/dev/alto) (`3rdparty/torchtitan`)

## Contact
For questions, issues, or contributions, please reach out to the maintainers:
- Guanchen Li — [@guanchenl](https://github.com/guanchenl) · GuanChen.Li@amd.com
- Han Wang — [@hann-wang](https://github.com/hann-wang) · Han.Wang@amd.com
- Yue Sun — [@ysa2215](https://github.com/ysa2215) · Yue.Sun2@amd.com
- Zhitao Wang — [@zhitwang17](https://github.com/zhitwang17) · Zhitao.Wang@amd.com

See [CODEOWNERS](.github/CODEOWNERS) for the full ownership list.
