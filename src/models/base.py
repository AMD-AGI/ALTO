import gc
import inspect
import json
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config, device_map=None, use_cache=False):
        self.config = config
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.trust_remote_code = self.config.model.get('trust_remote_code', True)
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
        self.weight_offload = self.config.model.get('weight_offload', False)
        torch_dtype = self.config.model.torch_dtype
        self.torch_dtype = torch_dtype if torch_dtype == 'auto' else eval(torch_dtype)
        self.device_map = device_map
        self.use_cache = use_cache
        self.vision_model = None
        self.vision_projector = None
        self.modality = 'language'
        self.kvcache_buffer = []
        self.build_tokenizer()
        self.build_model()
        self.model.eval()
        self.update_key_info()

    @abstractmethod
    def find_blocks(self):
        pass

    @abstractmethod
    def find_embed_layers(self):
        pass

    @abstractmethod
    def get_embed_layers(self):
        pass

    @abstractmethod
    def get_layers_except_blocks(self):
        pass

    @abstractmethod
    def get_subsets_in_block(self, block):
        pass

    @abstractmethod
    def skip_layer_name(self):
        pass

    @abstractmethod
    def has_bias(self):
        pass

    def get_modality(self):
        assert self.modality in ['vision', 'language']
        return self.modality
    
    def set_modality(self, modality='language'):
        assert modality in ['vision', 'language']
        self.modality = modality
        self.update_key_info()

    def update_key_info(self):
        self.find_blocks()
        self.find_embed_layers()
        self.find_block_name()

    def reset_kv(self):
        for kvcache in self.kvcache_buffer:
            kvcache._reset_states()

    def get_matmul_in_block(self):
        return {}

    def get_act_fn_in_block(self):
        return {}

    def get_softmax_in_block(self):
        return {}
    
    def find_block_name(self):
        pass

    def get_model(self):
        return self.model

    def get_extra_rot_module_besides_embed_layers(self):
        return []

    def get_blocks(self):
        return self.blocks

    def build_tokenizer(self):
        assert self.tokenizer_mode in ['fast', 'slow']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        return self.tokenizer

    def get_num_attention_heads(self):
        return self.model_config.num_attention_heads

    def apply_chat_template(self, prompt):
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def batch_process(self, samples, apply_chat_template=False, return_inputs=True):
        texts = []
        for idx in range(len(samples)):
            question = samples[idx]['question']
            if apply_chat_template:
                question = self.apply_chat_template(question)
            texts.append(question)
        if not return_inputs:
            return texts
        model_inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        return inputs

    def get_catcher(self, first_block_input):
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                if 'output_router_logits' in kwargs:
                    assert kwargs['output_router_logits'] is False
                    kwargs.pop('output_router_logits')
                first_block_input['kwargs'].append(kwargs)
                raise ValueError
        return Catcher

    def __str__(self):
        return f'\nConfig: \n{str(self.model_config)} \nModel: \n{str(self.model)}'

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False
        logger.info(f'self.model_config : {self.model_config}')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.model_config,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        logger.info(f'self.model : {self.model}')

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)

        Catcher = self.get_catcher(first_block_input)

        if not self.weight_offload:
            self.move_embed_to_device('cuda')
            self.blocks[0] = self.blocks[0].cuda()
        self.blocks[0] = Catcher(self.blocks[0])

        for data in calib_data:
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            try:
                self.model(**data)
            except ValueError:
                pass
        self.first_block_input = first_block_input
        assert len(self.first_block_input) > 0, 'Catch input data failed.'
        if padding_mask:
            for idx in range(len(self.first_block_input['data'])):
                token_num = self.first_block_input['data'][idx].shape[1]
                if token_num != padding_mask[idx].shape[1]:
                    padding_mask[idx] = F.pad(
                        padding_mask[idx],
                        self.get_one_pad_setting(
                            self.tokenizer.padding_side,
                            token_num - padding_mask[idx].shape[1]
                        ),
                        value=1
                    )
        self.padding_mask = padding_mask

        self.first_block_input = {
            k: v.cpu() if torch.is_tensor(v) else
            [t.cpu() for t in v] if isinstance(v, list) and all(torch.is_tensor(t) for t in v)
            else v
            for k, v in self.first_block_input.items()
        }

        if not self.weight_offload:
            self.blocks[0] = self.blocks[0].cpu()
            self.move_embed_to_device('cpu')
        self.blocks[0] = self.blocks[0].module

    def get_one_pad_setting(self, padding_side, length):
        if padding_side == 'left':
            return [0, length]
        elif padding_side == 'right':
            return [length, 0]
        else:
            raise Exception(f'Not support padding_side: {padding_side}.')

    def get_first_block_input(self):
        return self.first_block_input

    def get_padding_mask(self):
        return self.padding_mask

    def get_model_config(self):
        return self.model_config

    def move_embed_to_device(self, device):
        for embed_layer in self.get_embed_layers():
            embed_layer.to(device)
        for attention_rotary_layer in self.get_attention_rotary_layers():
            attention_rotary_layer.to(device)

    def get_block_linears(self, block):
        return {
            name: m
            for name, m in block.named_modules()
            if isinstance(m, nn.Linear)
        }

    def get_all_linears(self, module):
        return {
            name: m
            for name, m in module.named_modules()
            if isinstance(m, nn.Linear)
        }

    def get_extra_modules(self, block):
        return {}

    def get_moe_gate(self, block):
        return None
