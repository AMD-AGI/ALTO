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

from ..blockwise_optimization import BlockwiseOptimizer
from .core import FloatQuantizer, IntegerQuantizer


class BlockwiseQuantization(BlockwiseOptimizer):
    def __init__(self, model, quant_config, config, input):
        super().__init__(model, quant_config, config, input)
        self.quant_out = self.quant_config.get('quant_out', False)
        # weight
        quant_type = self.quant_config['weight'].get('quant_type', 'int-quant')
        if quant_type == 'int-quant':
            self.weight_quant_module = IntegerQuantizer
        elif quant_type == 'float-quant':
            self.weight_quant_module = FloatQuantizer
        logger.info(f'The used Weight Quant Module is {self.weight_quant_module}')
        self.wquantizer = self.weight_quant_module(**self.quant_config['weight'])
        # act
        if 'act' in self.quant_config:
            self.w_only = False
            quant_type = self.quant_config['act'].get('quant_type', 'int-quant')
            if quant_type == 'int-quant':
                self.act_quant_module = IntegerQuantizer
            elif quant_type == 'float-quant':
                self.act_quant_module = FloatQuantizer
            self.aquantizer = self.act_quant_module(**self.quant_config['act'])
            self.act_static = self.quant_config['act'].get('static', False)
            if self.act_static:
                assert (
                    self.quant_config['act']['granularity'] == 'per_tensor'
                ), 'Only support per_tensor static quant'
        else:
            self.w_only = True
            self.aquantizer = None
            self.act_static = False
            self.quant_attn = False
            self.quant_softmax = False
            self.quant_act_fn = False


    def w_qdq(self, module, wquantizer):
        tmp_weight = module.weight
        tmp_weight = wquantizer.fake_quant_weight_dynamic(tmp_weight, None)
        return tmp_weight

    def a_qdq(self, act, module, aquantizer, input_index=0):
        if self.act_static:
            args = {
                'scales': (getattr(module, f'buf_act_scales_{input_index}', None)),
                'zeros': (getattr(module, f'buf_act_zeros_{input_index}', None)),
                'qmax': (getattr(module, f'buf_act_qmax_{input_index}', None)),
                'qmin': (getattr(module, f'buf_act_qmin_{input_index}', None)),
            }
            return aquantizer.fake_quant_act_static(act, args)
        else:
            return aquantizer.fake_quant_act_dynamic(act)

    def get_replacement_params(self, mode='fake_quant', w_only=False, name=None):
        params_dict = {}
        if mode in ['fake_quant', 'fake_quant_wo_kv']:
            params_dict['a_qdq'] = (
                partial(self.a_qdq, aquantizer=self.aquantizer)
                if not w_only
                else None
            )
            params_dict['w_qdq'] = partial(self.w_qdq, wquantizer=self.wquantizer)
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

    def optimize(self, block):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        extra_modules = self.model.get_extra_modules(block)

        input_feat_modules = {
            k: v for d in [named_linears, extra_modules] for k, v in d.items()
        }
        logger.info(f'input_feat_modules: {input_feat_modules}')
        input_feat = defaultdict(list)

        handles = self.register_hooks(input_feat_modules, input_feat)

        self.block_init(block)

        self.run(block, input_feat, handles)

        block = block.cpu()
        del input_feat, block
        gc.collect()
        torch.cuda.empty_cache()

    def register_hooks(self, input_feat_modules, input_feat):
        handles = []
        if not self.data_free:
            for name in input_feat_modules:
                handles.append(
                    input_feat_modules[name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, name=name, feat_dict=input_feat
                        )
                    )
                )
        return handles

    def run(self, block, input_feat, handles):
        if not self.data_free:
            if self.quant_out:
                self.block_forward(block)
            else:
                self.input['data'] = self.block_forward(block)

            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

            self.block_transform(block, input_feat, self.input['kwargs'])
        else:
            self.block_transform(block)

        if not self.data_free and self.quant_out:
            self.model.replace_module_block(
                FakeQuantLinear,
                block,
                self.block_idx,
                self.get_replacement_params(
                    mode='fake_quant', w_only=self.w_only, name=None
                ),
            )
            self.set_non_linear_mode('fake_quant', block, False)
            self.input['data'] = self.block_forward(block)
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx}-th block')
        subsets = self.model.get_subsets_in_block(block)

        self.set_non_linear_mode('fake_quant', block, False)

        for index, subset in enumerate(subsets):
            logger.info(f'subset: {subset}')
            layers_dict = subset['layers']
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
            self.subset_transform(
                subset,
                input_feat,
                subset_kwargs,
            )


        self.set_non_linear_mode('fake_quant', block, True)
        logger.info(f'End transform the {self.block_idx}-th block')

