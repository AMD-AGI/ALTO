# AMD Distributed Model-Optimizer

## Observers
Observe and calculate statistics of module weights/inputs/outputs.

* Quantization
  * minmax
* Sparsification
  * per_channel_norm
  * hessian

## Modifiers

* Quantization
  * QuantizationModifier
* Sparsification
  * WandaPruningModifier
  * SparseGPTModifier

## Installation

Install the modified torchtitan:

```bash
pip install --no-build-isolation -e 3rdparty/torchtitan
```

Install this project:

```
pip install -e .
```

## Configuration

Create a recipe file following the same settings as [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/bede809f388aaeb1438a4d692d2d79109f9357dc).

Include the recipe in torchtitan config:

```toml
[model]
converters = ["modeloptimizer"]

[model_optimizer]
recipe = "./modeloptimizer/models/llama3/configs/recipe.yaml"
```

## Run calibration

For post-training calibration, set training steps to 1 and adjust calibration steps through `global_batch_size` and `local_batch_size`.

```toml
[training]
local_batch_size = 1
global_batch_size = 10
steps = 1
```

Start calibration by:

```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 CONFIG_FILE=./modeloptimizer/models/llama3/configs/llama3_1b.toml ./run.sh
```

## Export

By default torchtitan saves state dict in torch dcp format and converts it to hf safetensors offline.

We have patched the state_dict_adapter to save the observer/modifier states in hf safetensors. See [llama3 state_dict_adapter](modeloptimizer/models/llama3/state_dict_adapter.py) for example.

## TODO

* observers
  * [ ] mse
  * [ ] hessian
* modifiers
  * quantization
    * [ ] GPTQ
    * [ ] more quant settings: dtype, granularity, etc.
  * sparsification
    * [ ] Magnitude
  * [ ] pruning
  * [ ] transform
* [ ] models
* checkpointing
  * [ ] compressed tensors
  * [ ] permutation of Q, K scale/zero_point
    * The weight in torchtitan has a different layout
