# Calibration Dataset Examples

### C4
```yaml
calib:
  dataset:
    name: allenai/c4
    subset_name: default
    data_files:
      train: "en/c4-train.00000-of-01024.json.gz"
    split: train
    download: True
    download_mode: reuse_dataset_if_exists
    cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/c4
    revision: 607bd4c8450a42878aa9ddc051a65a055450ef87
    hf_context_key: text
  processing:
    n_samples: CUSTOMIZABLE
    bs: CUSTOMIZABLE
    seq_len: CUSTOMIZABLE
    preproc: CUSTOMIZABLE
    seed: CUSTOMIZABLE
```

### PTB
```yaml
calib:
  dataset:
    name: ptb_text_only
    subset_name: penn_treebank
    split: train
    download: True
    download_mode: reuse_dataset_if_exists
    cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/ptb
    hf_context_key: sentence
  processing:
    n_samples: CUSTOMIZABLE
    bs: CUSTOMIZABLE
    seq_len: CUSTOMIZABLE
    preproc: CUSTOMIZABLE
    seed: CUSTOMIZABLE
```

### Wikitest2
```yaml
calib:
  dataset:
    name: wikitext
    subset_name: wikitext-2-raw-v1
    split: train
    download: True
    download_mode: reuse_dataset_if_exists
    cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/wikitext2
    hf_context_key: text
  processing:
    n_samples: CUSTOMIZABLE
    bs: CUSTOMIZABLE
    seq_len: CUSTOMIZABLE
    preproc: CUSTOMIZABLE
    seed: CUSTOMIZABLE
```

### Fineweb-edu
```yaml
calib:
  dataset:
    name: HuggingFaceFW/fineweb-edu
    subset_name: sample-10BT
    data_files:
        train: "sample/10BT/000_00000.parquet"
    split: train
    download: True
    download_mode: reuse_dataset_if_exists
    cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/fineweb_edu
    hf_context_key: text
  processing:
    n_samples: CUSTOMIZABLE
    bs: CUSTOMIZABLE
    seq_len: CUSTOMIZABLE
    preproc: CUSTOMIZABLE
    seed: CUSTOMIZABLE
```

### Neural Magic Calibration Dataset
```yaml
calib:
  dataset:
    name: neuralmagic/calibration
    subset_name: LLM
    split: train
    download: True
    download_mode: reuse_dataset_if_exists
    cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/neuralmagic
    hf_context_key: text
  processing:
    n_samples: CUSTOMIZABLE
    bs: CUSTOMIZABLE
    seq_len: CUSTOMIZABLE
    preproc: CUSTOMIZABLE
    seed: CUSTOMIZABLE
```