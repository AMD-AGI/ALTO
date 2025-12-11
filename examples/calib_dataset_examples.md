```yaml
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
```

```yaml
name: wikitext
subset_name: wikitext-2-raw-v1
split: train
download: True
download_mode: reuse_dataset_if_exists
cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/wikitext2
hf_context_key: text
```

```yaml
name: HuggingFaceFW/fineweb-edu
subset_name: sample-10BT
data_files:
    train: "sample/10BT/000_00000.parquet"
split: train
download: True
download_mode: reuse_dataset_if_exists
cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/fineweb_edu
hf_context_key: text
```

```yaml
name: neuralmagic/calibration
subset_name: LLM
split: train
download: True
download_mode: reuse_dataset_if_exists
cache_dir: /group/ossdphi_algo_scratch_13/guanchen/datasets/calibration/neuralmagic
hf_context_key: text
```