import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm, trange

@torch.no_grad()
def prefill_perplexity(model, testenc, seq_len, bs):
    testenc = testenc.input_ids
    model.model.to('cuda')
    nsamples = testenc.numel() // seq_len

    nlls = []

    for i in trange(0, nsamples, bs):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * seq_len): (j * seq_len)].cuda()
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

    testenc.cpu()
    torch.cuda.empty_cache()

    return ppl.item()


@torch.no_grad()
def decode_perplexity(model, testenc, seq_len, bs, num_samples, tokenizer, num_eval_tokens):
    num_eval_tokens = 0
    num_samples = 1 if num_samples is None else num_samples
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    nlls = []

    for text in testenc[: num_samples]:
        logger.info(text)
        encodings = tokenizer(text, return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
        logger.info(f'seq_len: {seq_len}')
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx:idx + 1].cuda()
            with torch.no_grad():
                outputs = model.model(
                    input_ids,
                )
                logits = outputs.logits.view(-1, model.model.config.vocab_size)
                label = encodings.input_ids[:, idx + 1:idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            num_eval_tokens += 1
            if num_eval_tokens is not None and num_eval_tokens >= num_eval_tokens:
                break
        if num_eval_tokens is not None and num_eval_tokens >= num_eval_tokens:
            break
    model.reset_kv()
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
