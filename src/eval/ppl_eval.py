import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm, trange

from src.utils import to_device


@torch.no_grad()
def prefill_perplexity_offload(model, testenc, seq_len, bs, device="cuda"):
    layers = model.get_blocks()
    nsamples = testenc.input_ids.numel() // seq_len
    
    first_block_inputs = []
    for i in trange(nsamples, desc='prefill_perplexity_offload: collecting first block input'):
        batch = [{
            k: v[:, i*seq_len: (i+1)*seq_len] 
            for k, v in testenc.items() if torch.is_tensor(v)
        },]
        model.collect_first_block_input(batch)
        first_block_inputs.append(model.get_first_block_input())
        model.first_block_input = {}
    torch.cuda.empty_cache()

    block_hidden_state = to_device(torch.stack([d["data"][0][0] for d in first_block_inputs], 0), device)
    block_kwargs = first_block_inputs[0]['kwargs'][0]
    for i in trange(len(layers), desc='prefill_perplexity_offload: processing layers'):
        layer = to_device(layers[i], device)
        for j in range(nsamples):
            block_hidden_state[j] = layer(block_hidden_state[j].unsqueeze(0), **block_kwargs)[0]
        layers[i] = to_device(layer, 'cpu')
        torch.cuda.empty_cache()
    
    after_block_modules = to_device(model.get_layers_after_blocks(), device)
    testenc = to_device(testenc.input_ids, device)
    nlls = []
    for i in range(nsamples):
        hidden_states = block_hidden_state[i].unsqueeze(0)
        for module in after_block_modules:
            hidden_states = module(hidden_states)
        shift_logits = hidden_states[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seq_len) : ((i + 1) * seq_len)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    
    testenc = to_device(testenc, 'cpu')
    torch.cuda.empty_cache()
    
    return ppl.item()


@torch.no_grad()
def prefill_perplexity(model, testenc, seq_len, bs, device="cuda"):
    testenc = testenc.input_ids
    to_device(model.model, device)
    nsamples = testenc.numel() // seq_len

    nlls = []
    for i in trange(0, nsamples, bs, desc='prefill_perplexity: processing samples'):
        j = min(i + bs, nsamples)
        inputs = to_device(testenc[:, (i * seq_len): (j * seq_len)], device)
        inputs = inputs.reshape(j - i, seq_len)
        lm_logits = model.model(inputs).logits
        model.reset_kv()
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        neg_log_likelihood = loss.float() * seq_len * (j - i)
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))

    to_device(testenc, 'cpu')
    torch.cuda.empty_cache()

    return ppl.item()


@torch.no_grad()
def decode_perplexity_offload(model, testenc, seq_len, bs, num_samples, max_num_eval_tokens):
    if max_num_eval_tokens == -1:
        max_num_eval_tokens = 65536
    num_eval_tokens = 0
    model.model.to('cuda')
    num_samples = 1 if num_samples is None else num_samples
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    nlls = []

    for text in testenc[: num_samples]:
        logger.info(text)
        encodings = testenc.input_ids
        seq_len = encodings.size(1)
        logger.info(f'seq_len: {seq_len}')
        pbar = tqdm(range(0, seq_len - 1), desc=f"total tokens {max_num_eval_tokens}")

        for idx in pbar:
            input_ids = encodings[:, idx:idx + 1].cuda()
            with torch.no_grad():
                outputs = model.model(
                    input_ids,
                )
                logits = outputs.logits.view(-1, model.model.config.vocab_size)
                label = encodings[:, idx + 1:idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            num_eval_tokens += 1
            if num_eval_tokens is not None and num_eval_tokens >= max_num_eval_tokens:
                break
        if num_eval_tokens is not None and num_eval_tokens >= max_num_eval_tokens:
            break
    model.reset_kv()
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


@torch.no_grad()
def decode_perplexity(model, testenc, seq_len, bs, num_samples, max_num_eval_tokens):
    if max_num_eval_tokens == -1:
        max_num_eval_tokens = 65536
    num_eval_tokens = 0
    model.model.to('cuda')
    num_samples = 1 if num_samples is None else num_samples
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    nlls = []

    for text in testenc[: num_samples]:
        logger.info(text)
        encodings = testenc.input_ids
        seq_len = encodings.size(1)
        logger.info(f'seq_len: {seq_len}')
        pbar = tqdm(range(0, seq_len - 1), desc=f"total tokens {max_num_eval_tokens}")

        for idx in pbar:
            input_ids = encodings[:, idx:idx + 1].cuda()
            with torch.no_grad():
                outputs = model.model(
                    input_ids,
                )
                logits = outputs.logits.view(-1, model.model.config.vocab_size)
                label = encodings[:, idx + 1:idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            num_eval_tokens += 1
            if num_eval_tokens is not None and num_eval_tokens >= max_num_eval_tokens:
                break
        if num_eval_tokens is not None and num_eval_tokens >= max_num_eval_tokens:
            break
    model.reset_kv()
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
