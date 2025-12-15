# Dataset Preprocessing 

## Calibration

### calib_truncated_jointdoc_random
This preprocessing routine first concatenates all documents in the calibration dataset (using `sep`) and tokenizes the resulting joint document into a single sequence. It then generates n_samples training examples by repeatedly selecting a random contiguous window of length `seq_len` from the joint document. Each sampled subsequence is returned as a fixed-length tensor segment suitable for calibration.

> **Typical usage: GPTQ / SparseGPT / OminiQuant**; 
> 
> `wikitext2(sep='\n\n', hf_context_key='text')`
> 
> `ptb(sep=' ', hf_context_key='sentence')`
>
> `pileval(sep='\n\n', hf_context_key='text')`

```python
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
```

### calib_truncated_long_text_random
This routine constructs calibration samples by repeatedly drawing a random document from the dataset and tokenizing it, rejecting draws whose tokenized length is shorter than `seq_len`. For each accepted document, it uniformly selects a random contiguous span of length `seq_len`. Repeating this process `n_samples` times yields a list of fixed-length token subsequences sampled from sufficiently long texts.

> **Typical usage: GPTQ / SparseGPT**; 
> 
> `c4(hf_context_key='text')`


```python
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
```

### calib_truncated_short_text_random
This routine first shuffles the dataset and scans examples sequentially, keeping only “short” texts whose tokenized length does not exceed `seq_len`. It collects up to `n_samples` such token sequences, concatenates them into one long 1D token stream, and then splits the stream into as many non-overlapping contiguous segments of length `seq_len` as possible. The output is a list of fixed-length token blocks suitable for calibration.

> **Typical usage: AWQ**; 
> 
> `pileval(hf_context_key='text')`

```python
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
```

### calib_truncated_long_text_leftalign
This routine shuffles the calibration dataset and then iterates through examples, tokenizing each text with truncation to a maximum length of `seq_len`. For each selected example, it keeps the left-aligned prefix (i.e., the first `seq_len` tokens, or fewer if shorter) as a tensor. It returns the first `n_samples` such truncated sequences for calibration.

> **Typical usage: SmoothQuant**; 
> 
> `pileval(hf_context_key='text')`
```python
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
```

### calib_truncated_messages_leftalign
This routine first shuffles the dataset and selects the first `n_samples` entries. For each entry, it renders the message list into a single chat-formatted string via `apply_chat_template`. It then tokenizes each rendered conversation independently with truncation to `seq_len`, and returns a leftalign list of per-sample tensors (each of length ≤ `seq_len`).
> **Typical usage: ultrachat**; 
```python
@PREPROC_REGISTRY
def calib_truncated_messages_leftalign(calib_dataset, tokenizer, n_samples, seq_len, hf_context_key='text', *args, **kwargs):
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
```

## Test

### test_jointdoc
This helper function concatenates all texts in the dataset into a single joint document using `sep`, then tokenizes the resulting string into a single batched tensor. It returns the full tokenized encoding for inspection or debugging.
> **Typical usage: ppl evaluation**; 

```python
@PREPROC_REGISTRY
def test_jointdoc(calib_dataset, tokenizer, sep='\n\n', hf_context_key='text', *args, **kwargs):
    encoded = tokenizer(sep.join(calib_dataset[hf_context_key]), return_tensors='pt')
    return encoded
```