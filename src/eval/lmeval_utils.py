import os
from tqdm import tqdm, trange
import torch.nn.functional as F
import types
import functools
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.api.model import TemplateLM
from transformers import StoppingCriteria, StoppingCriteriaList


class EvalModel(TemplateLM):
    def __init__(self, model, batch_size=1, device="cuda", **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.model = model.model.to(device)
        self.dtype = model.torch_dtype
        self.tokenizer = model.tokenizer
        self.device_map = model.device_map
        self.device = device

    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False, **kwargs)

    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        results = []
        for (context, continuation), context_enc, continuation_enc in tqdm(requests, disable=disable_tqdm):
            input_ids = torch.tensor([context_enc + continuation_enc]).to(self.device)
            target_ids = torch.tensor([continuation_enc]).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, len(context_enc)-1:-1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum().item()
            results.append((total_log_prob, False))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        results = []

        for req in tqdm(requests, disable=disable_tqdm):
            if hasattr(req, "args"):
                string = req.args[0]
            elif isinstance(req, (tuple, list)):
                string = req[0]
            else:
                string = req

            tokens = self.tok_encode(string)
            rolling_window = self.max_length
            loglik_chunks = []

            for i in range(0, len(tokens), rolling_window):
                context_tokens = tokens[:i]
                continuation_tokens = tokens[i:i + rolling_window]
                if len(continuation_tokens) == 0:
                    continue

                input_tokens = context_tokens + continuation_tokens
                input_ids = torch.tensor([input_tokens], device=self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits

                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  
                labels = input_ids[:, 1:]                           

                C = len(context_tokens)
                K = len(continuation_tokens)
                t_start = max(C - 1, 0)
                t_end = C + K - 2 

                if t_end < t_start:
                    continue

                positions = torch.arange(t_start, t_end + 1, device=self.device)
                selected_log_probs = log_probs[:, positions, :].gather(
                    -1, labels[:, positions].unsqueeze(-1)
                ).squeeze(-1) 
                loglik = selected_log_probs.sum().item()  
                loglik_chunks.append(loglik)
            results.append(sum(loglik_chunks))

        return results

    def generate_until(self, requests):
        results = []
        for request in tqdm(requests, desc='running generate until'):
            context = request.args[0]
            gen_kwargs = request.args[1]
            max_length = gen_kwargs.get('max_length', self.max_gen_toks)
            stop_sequences = gen_kwargs.get('stop_sequences', [])

            inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
            stopping_criteria = self._get_stopping_criteria(stop_sequences)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    eos_token_id=self.eot_token_id,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    early_stopping=True,
                    stopping_criteria=stopping_criteria,
                )
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            for stop_seq in stop_sequences:
                index = generated.find(stop_seq)
                if index != -1:
                    generated = generated[:index]
                    break
            results.append(generated)
        return results

    def _get_stopping_criteria(self, stop_sequences):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopSequencesCriteria(StoppingCriteria):
            def __init__(self, stop_sequences, tokenizer):
                self.stop_sequences = stop_sequences
                self.tokenizer = tokenizer

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated:
                        return True
                return False

        return StoppingCriteriaList([StopSequencesCriteria(stop_sequences, self.tokenizer)])

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256 