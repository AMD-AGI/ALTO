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
- **[MXFP8](alto/kernels/mxfp8)** — linear and grouped GEMM (block-scaled E4M3, with E5M2 reserved for gradients).
- **[NVFP4](alto/kernels/fp4/nvfp4)** — linear and grouped GEMM, using an E4M3 inner-block scale with an optional two-level (tensorwise) outer scale.

Techniques used to narrow the gap versus BF16 include:

- 2D block quantization
- Randomized Hadamard Transform (RHT)
- Stochastic Rounding (SR)
- Differential Gradient Estimation (DGE)
- Two-Level Scaling
  - blockwise (128-dim macro-block for MXFP4)
  - tensorwise (for NVFP4)
- Weight De-Oscillation
  - see [PR #25](https://github.com/AMD-AGI/ALTO/pull/25)

### Modifiers

Recipes can combine multiple stages under `alto/modifiers/`, e.g. Low-Precision Training (LPT) Modifier.

Recipe YAML follows the same general shape as [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/bede809f388aaeb1438a4d692d2d79109f9357dc); use the configs under `alto/models/*/configs/` as concrete templates.

## Supported models

| Model | Integration notes |
|--------|-------------------|
| **[Llama 3](alto/models/llama3)** | * Extended [state dict adapter](alto/models/llama3/state_dict_adapter.py) for observer and modifier state; <br/> * Config registry hooks for TorchTitan. |
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
from torchtitan.components.model_converter import ModelConvertersContainer
from alto.components.converter import ModelOptConverter

config.model_converters = ModelConvertersContainer.Config(
    converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml"),
    ],
)
```

Each model's [`config_registry.py`](alto/models/llama3/config_registry.py) starts from the upstream recipe when one exists, then restores the ALTO dataset, batch, checkpoint, and modifier settings. `examples/run.sh` selects ALTO's trainer through `TRAIN_FILE` (default `alto.train`).

### Launch training

From the repository root, the shared launcher wraps `torchrun` and `python -m alto.train`:

```bash
NGPU=8 MODULE=llama3 CONFIG=your_config_name ./examples/run.sh
```

Extra arguments are forwarded to `alto.train`. Besides the TorchTitan loader options (`--override`, `--output-dir`, `--resume-step`, ...), dotted config field overrides are accepted, e.g. `--training.steps 10 --checkpointer.folder ckpt`. Overrides are applied after the config function runs, so fields baked into the model config (such as the sequence length) must be changed in the config function instead.

Environment variables (see [`examples/run.sh`](examples/run.sh)):

| Variable | Role |
|----------|------|
| `NGPU` | Processes per node (default `8`). |
| `MODULE` | Model name (`llama3`, `gpt_oss`, …) resolving to `alto.models.<name>.config_registry`, or a fully qualified module. |
| `CONFIG` | Registered config function name. |
| `TRAIN_FILE` | Python module for training entrypoint (default `alto.train`). |
| `COMM_MODE` | Optional: `fake_backend` for config checks with fake process groups. |

## Examples

### GPT-OSS — MXFP4 Training

- Recipe: [`alto/models/gpt_oss/configs/lpt_recipe.yaml`](alto/models/gpt_oss/configs/lpt_recipe.yaml)

  uses `LowPrecisionTrainingModifier` with `scheme: "mxfp4"`.

- Config: [`gpt_oss_20b_lpt`](alto/models/gpt_oss/config_registry.py). Upstream does not publish a 20B recipe, so this config builds the `20b` flavor and keeps ALTO's pretraining batch, learning rate, and Megatron data settings. [`gpt_oss_debugmodel_lpt`](alto/models/gpt_oss/config_registry.py) remains available for a small upstream debug run.

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
      targets: ["Linear", "GptOssGroupedLinear"]
      ignore: ["lm_head", "re:.*\\.router\\.gate"]
      use_2dblock_x: false
      use_2dblock_w: true
      use_hadamard: true
      use_sr_grad: true
      use_dge: false

      # Use "tensorwise" to enable NVFP4's outer scale
      two_level_scaling: none  

      # Step to enable weight de-oscillation. 0 means disabled.
      # If 0, weight de-oscillation is disabled.
      deosc_step: 2000
```

To train the same model with NVFP4 instead, set `scheme: "nvfp4"` and `two_level_scaling: "tensorwise"`.

## Export and evaluation

TorchTitan typically saves checkpoints in PyTorch DCP format; you can convert to Hugging Face Safetensors and run lm-eval tasks with the bundled export utility:

```bash
python ./alto/utils/exportation/export.py \
  alto.models.llama3.config_registry llama3_1b_opt \
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
