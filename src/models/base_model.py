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

from src.optimization.quantization.modules import LINEAR_TYPES

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
        self.kvcache_buffer = []
        self.build_tokenizer()
        self.build_model()
        self.model.eval()
        self.update_key_info()

    def build_tokenizer(self):
        assert self.tokenizer_mode in ['fast', 'slow']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=self.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
    def get_layers_before_blocks(self):
        pass

    @abstractmethod
    def get_layers_after_blocks(self):
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

    @abstractmethod
    def get_query_projection_name(self):
        return 'q_proj'
    
    @abstractmethod
    def get_key_projection_name(self):
        return 'k_proj'
    
    @abstractmethod
    def get_value_projection_name(self):
        return 'v_proj'
    
    @abstractmethod
    def get_out_projection_name(self):
        return 'o_proj'
    
    @abstractmethod
    def get_up_projection_name(self):
        return 'up_proj'
    
    @abstractmethod
    def get_gate_projection_name(self):
        return 'gate_proj'
    
    @abstractmethod
    def get_down_projection_name(self):
        return 'down_proj'

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

    @torch.no_grad()
    def collect_first_block_input(self, calib_data):
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

    def get_first_block_input(self):
        return self.first_block_input

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

    def replace_module_block(self, module, block, block_idx, params_dict):
        self.replace_module_subset(module,
                                    block,
                                    {'layers': self.get_block_linears(block)},
                                    block_idx,
                                    params_dict)


    def replace_module_subset(self, module, block, subset, block_idx, params_dict):
        if module in LINEAR_TYPES:
            layers_dict = {
                name: layer for name, layer in subset['layers'].items()
                if isinstance(layer, tuple(LINEAR_TYPES))
            }
        else:
            layers_dict = subset['layers']
        for name, m in layers_dict.items():
            if hasattr(m, 'no_quant') and m.no_quant:
                continue

            M = module.new(m, **params_dict)

            name_tmp = name.rsplit('.', 1)
            if len(name_tmp) == 2:
                parent_name = name_tmp[0]
                parent = block.get_submodule(parent_name)
                child_name = name_tmp[1]
            elif len(name_tmp) == 1:
                parent = block
                child_name = name_tmp[0]

            setattr(parent, child_name, M)
            del M

            logger.info(f'replace >>> {name} in {block_idx}-th block')

        del layers_dict
        gc.collect()
        torch.cuda.empty_cache()

    def replace_module_all(self, module, params_dict, keep_device=False):
        for block_idx in range(len(self.blocks)):
            logger.info(f'Replace block index: {block_idx}/{len(self.blocks)}')
            if keep_device:
                self.replace_module_block(module, self.blocks[block_idx], block_idx, params_dict)
            else:
                self.blocks[block_idx].cuda()
                self.replace_module_block(module, self.blocks[block_idx], block_idx, params_dict)
                self.blocks[block_idx].cpu()
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(f'The Replaced model: {self.model}')