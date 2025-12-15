# Model-Optimizer

```bash
├── configs                        # yaml cfg for algorithm / calib / eval / save
│   ├── distillation
│   ├── pruning
│   ├── quanitzation
│   ├── recipe
│   ├── sparsification
│   │   ├── llama-magnitude.yml
│   │   ├── llama-wanda-demo.yml
│   │   └── llama-wanda.yml
│   └── speculative-decoding
├── examples                       # entrance
│   ├── calib_dataset_examples.md
│   └── prune_example.sh
├── README.md
├── requirements.txt
└── src
    ├── data                       # dataset management (one for calib / eval)
    │   ├── base_dataset.py
    │   ├── __init__.py
    │   └── preproc_factory.py
    ├── eval                       # eval code
    │   ├── eval_utils.py
    │   ├── __init__.py
    │   ├── lmeval_utils.py
    │   └── ppl_eval.py
    ├── __main__.py                # code entrance
    ├── models                     # model management
    │   ├── base_model.py
    │   ├── __init__.py
    │   ├── instella.py
    │   ├── llama.py
    │   └── qwen.py
    ├── optimization               # algorithms
    │   ├── blockwise_optimization.py
    │   ├── distillation
    │   ├── __init__.py
    │   ├── pruning
    │   ├── quantization
    │   ├── sparsification
    │   │   ├── blockwise_sparsification.py
    │   │   ├── __init__.py
    │   │   └── wanda.py
    │   └── speculative-decoding
    └── utils                     # utils
        ├── __init__.py
        ├── registry_factory.py
        └── utils.py
```

## Examples
```bash
# Use xcdoss mi250 system to run without changing configuration
bash examples/prune_example.sh
```
