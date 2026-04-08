# AMD Distributed Model-Optimizer


## Features

### Observers
Observe and calculate statistics of module weights/inputs/outputs.

* Quantization
  * minmax
* Sparsification
  * per_channel_norm
  * hessian

### Modifiers

* Quantization
  * QuantizationModifier
* Sparsification
  * WandaPruningModifier
  * SparseGPTModifier
* Low-Precision Training
  * MXFP4

### Models

* Llama3
  * Patched [state_dict_adapter](modeloptimizer/models/llama3/state_dict_adapter.py) to save the observer/modifier states in hf safetensors.
  * Patched [RoPE](modeloptimizer/models/patcher.py) to keep wq, wk in the transformers layout.
* DeepSeek
* GPT-OSS

## Installation

Install the modified torchtitan:

```bash
pip install --no-build-isolation -e 3rdparty/torchtitan
```

Install this project:

```
pip install -e .
```


## General Usage

Create a recipe file following similar settings as [llm-compressor](https://github.com/vllm-project/llm-compressor/tree/bede809f388aaeb1438a4d692d2d79109f9357dc).

Include the recipe in the torchtitan config registry:

(See [Llama3](modeloptimizer/models/llama3/config_registry.py) for example)
```python
config.model_converters = ModelConvertersContainer.Config(converters=[
  ModelOptConverter.Config(recipe="./modeloptimizer/models/llama3/configs/recipe.yaml",),
],)
```

Then start training or post-training optimization.

## Low-Precision Training

* Create a recipe with `LowPrecisionTrainingModifier`:

  See [GPT-OSS LPT recipe](modeloptimizer/models/gpt_oss/configs/lpt_recipe.yaml) for example.
  ```yaml
  training_stage:
    lpt_modifiers:
      LowPrecisionTrainingModifier:
        scheme: "mxfp4"
        targets: ["Linear", "GptOssGroupedExperts"]
        ignore: ["output", "re:.*\\.router\\.gate"]
        use_2dblock_x: false
        use_2dblock_w: true
        use_hadamard: true
        use_sr_grad: true
        use_dge: false
  ```
* Include the recipe in the config registry

  See [gpt_oss_20b_lpt](modeloptimizer/models/gpt_oss/config_registry.py) for example.

* Start training with
  ```bash
  NGPU=8 MODULE=gpt_oss CONFIG=gpt_oss_20b_lpt ./examples/run.sh
  ```

## Post-Training Optimization

### Run calibration

For post-training calibration, set training steps to 1 and adjust calibration steps through `global_batch_size` and `local_batch_size`.

```python
config.training.local_batch_size = 1
config.training.global_batch_size = 10
config.training.steps = 1
```

Start calibration by:

```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 MODULE=llama CONFIG=llama3_1b_opt ./examples/run.sh
```

### Export

By default torchtitan saves state dict in torch dcp format and converts it to hf safetensors offline.

We have prepared a script for checkpoint conversion, compression and evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python ./modeloptimizer/utils/exportation/export.py \
  llama3 llama3_1b_opt \
  --tasks wikitext
```


## TODO

* observers
  * [ ] mse
  * [x] hessian
* modifiers
  * quantization
    * [ ] GPTQ
    * [ ] more quant settings: dtype, granularity, etc.
  * sparsification
    * [x] SparseGPT
    * [x] Magnitude
  * [x] pruning
  * low-precision training
    * [x] MXFP4
    * [ ] MXFP8
    * [ ] blockwise FP8
  * [ ] transform
* models
  * [x] llama3
  * [x] deepseek
  * [x] gpt-oss
  * [ ] flux
* checkpointing
  * [x] compressed tensors for quantization
  * [x] compressed tensors for sparsification
  * [x] layer name mapping in the ignore list
  * [x] include model optimizer states in fqn_to_index_mapping
  * [x] permutation of Q, K scale/zero_point
    * The weight in torchtitan has a different layout
    * We have patched torchtitan llama3 models to use the transformers layout
* parallelization
  * [x] PP
  * [ ] other parallelization dims
