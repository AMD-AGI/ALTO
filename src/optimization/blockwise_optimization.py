import os
from abc import ABCMeta, abstractmethod
import torch
from loguru import logger


class BlockwiseOptimizer(metaclass=ABCMeta):
    def __init__(self, model, optimization_config, input):
        self.model = model
        self.blocks = model.get_blocks()
        self.optimization_config = optimization_config
        self.input = input
        self.data_free = False if self.input else True
        self.block_idx = None
        self.num_blocks = len(self.blocks)
        if self.input:
            for i in range(len(input['kwargs'])):
                if 'use_cache' in input['kwargs'][i]:
                    input['kwargs'][i].pop('use_cache')
            for i in range(len(input['kwargs'])):
                if 'past_key_value' in input['kwargs'][i]:
                    input['kwargs'][i]['past_key_value'] = None
            self.n_samples = 0
            for i in range(len(input['data'])):
                self.n_samples += input['data'][i].shape[0]

    def optimize(self):
        for i in range(len(self.blocks)):
            self.block_idx = i
            logger.info(
                f'\nblock index: {self.block_idx}/{len(self.blocks)} '
                f'\nblock: {self.blocks[self.block_idx]}'
            )
            self.optimize_block(self.blocks[self.block_idx])

    def cache_input_hook(self, module, inputs, outputs, name, feat_dict):
        module_inputs = [i.detach().cpu() for i in inputs]
        if len(module_inputs) == 1:
            inp = module_inputs[0]
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            feat_dict[name].append(inp)
        else:
            feat_dict[name].append(tuple(module_inputs))

    @abstractmethod
    def optimize_block(self, block):
        pass

    def layer_init(self, layer):
        pass

    def subset_init(self, subset):
        pass

    def block_init(self, block):
        pass