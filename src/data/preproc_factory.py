import json
import os
import random

import torch

from src.utils import PREPROC_REGISTRY


@PREPROC_REGISTRY
def calib_truncated_jointdoc_random(calib_dataset, tokenizer, n_samples, seq_len, sep='\n\n', hf_context_key='text', *args, **kwargs):
    encoded = tokenizer(sep.join(calib_dataset[hf_context_key]), return_tensors='pt')
    samples = []
    for _ in range(n_samples):
        i = random.randint(0, encoded.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def calib_truncated_long_text_random(calib_dataset, tokenizer, n_samples, seq_len, hf_context_key='text', *args, **kwargs):
    samples = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(calib_dataset) - 1)
            encoded = tokenizer(calib_dataset[i][hf_context_key], return_tensors='pt')
            if encoded.input_ids.shape[1] >= seq_len:
                break
        i = random.randint(0, encoded.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = encoded.input_ids[:, i:j]
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def calib_truncated_short_text_random(calib_dataset, tokenizer, n_samples, seq_len, hf_context_key='text', *args, **kwargs):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data[hf_context_key]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = torch.cat(samples, dim=1)    # (1, L_total)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_split)]
    return samples    # [(1, seq_len), (1, seq_len), ...]


@PREPROC_REGISTRY
def calib_truncated_long_text_leftalign(calib_dataset, tokenizer, n_samples, seq_len, hf_context_key='text', *args, **kwargs):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data[hf_context_key]
        encoded = tokenizer(
            line, return_tensors='pt', max_length=seq_len, truncation=True
        )
        line_encoded = encoded.input_ids
        samples.append(line_encoded)
        n_run += 1
        if n_run == n_samples:
            break
    return samples


@PREPROC_REGISTRY
def calib_truncated_messages_ragged(calib_dataset, tokenizer, n_samples, seq_len, hf_context_key='text', *args, **kwargs):
    calib_dataset = calib_dataset.shuffle(seed=42).select(range(n_samples))
    texts = []
    samples = []
    for example in calib_dataset:
        text = tokenizer.apply_chat_template(
            example[hf_context_key],
            tokenize=False,
        )
        texts.append(text)

    for i in range(n_samples):
        encoded = tokenizer(
            texts[i],
            padding=False,
            max_length=seq_len,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        inp = encoded.input_ids
        samples.append(inp)
    return samples


@PREPROC_REGISTRY
def test_jointdoc(calib_dataset, tokenizer, sep='\n\n', hf_context_key='text', *args, **kwargs):
    encoded = tokenizer(sep.join(calib_dataset[hf_context_key]), return_tensors='pt')
    return encoded