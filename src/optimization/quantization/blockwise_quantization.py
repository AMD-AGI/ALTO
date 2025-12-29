import copy
import functools
import gc
import json
import os
import re
from collections import defaultdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from .modules import *
from ..blockwise_optimization import BlockwiseOptimizer
from .core import FloatQuantizer, IntegerQuantizer


class BlockwiseQuantization(BlockwiseOptimizer):
    def __init__(self, model, quant_config, global_config, input):
        super().__init__(model, quant_config, global_config, input)
        self.quant_config = self.optimization_config
        self.error_accumulation = self.quant_config.get('error_accumulation', False)
        # weight
        quant_dtype = self.quant_config['weight'].get('quant_dtype', 'int')
        if quant_dtype == 'int':
            self.weight_quantizer_module = IntegerQuantizer
        elif quant_dtype == 'float':
            self.weight_quantizer_module = FloatQuantizer
        self.weight_quantizer = self.weight_quantizer_module(**self.quant_config['weight'])
        # act
        if 'act' in self.quant_config:
            self.weight_only = False
            quant_dtype = self.quant_config['act'].get('quant_dtype', 'int')
            if quant_dtype == 'int':
                self.act_quantizer_module = IntegerQuantizer
            elif quant_dtype == 'float':
                self.act_quantizer_module = FloatQuantizer
            self.act_quantizer = self.act_quantizer_module(**self.quant_config['act'])
        else:
            self.weight_only = True
            self.act_quantizer = None

    def w_qdq(self, module, weight_quantizer):
        return weight_quantizer.fake_quant_weight_dynamic(module.weight, None)

    def a_qdq(self, act, module, act_quantizer, input_index=0):
        return act_quantizer.fake_quant_act_dynamic(act)

    def get_replacement_params(self, mode='fake_quant', weight_only=False, name=None):
        params_dict = {}
        if mode in ['fake_quant', 'fake_quant_wo_kv']:
            params_dict['a_qdq'] = (
                partial(self.a_qdq, act_quantizer=self.act_quantizer)
                if not weight_only
                else None
            )
            params_dict['w_qdq'] = partial(self.w_qdq, weight_quantizer=self.weight_quantizer)
        return params_dict

    def block_forward(self, block, input_data=None):
        output = []
        if input_data is None:
            input_data = self.input['data']
        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            for k in self.input['kwargs'][i]:
                if torch.is_tensor(self.input['kwargs'][i][k]):
                    self.input['kwargs'][i][k] = self.input['kwargs'][i][k].to(
                        device=next(block.parameters()).device
                    )
                if isinstance(self.input['kwargs'][i][k], tuple):
                    self.input['kwargs'][i][k] = tuple(
                        tmp.to(device=next(block.parameters()).device)
                        for tmp in self.input['kwargs'][i][k]
                    )
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])
                if isinstance(out, tuple):
                    out = out[0]
                output.append(out)
        return output

    def optimize_block(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)
        input_feat_modules = {k: v for d in [named_linears, extra_modules] for k, v in d.items()}
        logger.info(f'input_feat_modules: {input_feat_modules}')
        input_feat = defaultdict(list)
        handles = self.register_hooks(input_feat_modules, input_feat)
        if not self.data_free:
            if self.error_accumulation:
                self.block_forward(block)
            else:
                self.input['data'] = self.block_forward(block)
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()
            self.optimize_block_subsets(block, input_feat, self.input['kwargs'])
        else:
            self.optimize_block_subsets(block, None, None)
        self.model.replace_module_block(
            FakeQuantLinear,
            block,
            self.block_idx,
            self.get_replacement_params(
                mode='fake_quant', weight_only=self.weight_only, name=None
            ),
        )
        self.set_non_linear_mode('fake_quant', block, False)
        if not self.data_free and self.error_accumulation:
            self.input['data'] = self.block_forward(block)
        block = block.cpu()
        del input_feat, block
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def set_non_linear_mode(self, quant_format, module, mode):
        assert mode in [True, False]
        if quant_format != 'fake_quant':
            return
        for name, m in module.named_modules():
            if getattr(m, 'calib', None) is not None:
                m.calib = mode
    
    def optimize_block_subsets(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = self.model.get_subsets_in_block(block)
        self.set_non_linear_mode('fake_quant', block, False)
        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            inspect_module = subset['inspect']
            input_name = subset['input'][0]
            inspect_has_kwargs = subset['has_kwargs']
            if inspect_has_kwargs:
                if 'sub_keys' in subset:
                    subset_kwargs = []
                    for i in range(len(block_kwargs)):
                        for k, v in subset['sub_keys'].items():
                            subset_kwargs.append({k: block_kwargs[i][v]})
                else:
                    subset_kwargs = block_kwargs
            else:
                subset_kwargs = {}
            self.optimize_subset(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs
            )
        self.set_non_linear_mode('fake_quant', block, True)
        logger.info(f'End transform the {self.block_idx}-th block')

    def optimize_subset(self, layers_dict, input_feat, prev_op, input_name, inspect_module, subset_kwargs):
        pass

