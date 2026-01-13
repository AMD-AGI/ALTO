import functools
import gc
import os
import json
from collections import defaultdict

import torch
from loguru import logger

from src.utils import module_device, to_device

from ..blockwise_optimization import BlockwiseOptimizer


class BlockwiseSparsification(BlockwiseOptimizer):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)
        self.sparsity_config = self.optimization_config
        self.error_accumulation = self.sparsity_config.get('error_accumulation', False)
        logger.info(f'use error_accumulation {self.error_accumulation}')
        self.W_mask = {}
        self.sparsity = self.sparsity_config['weight']['sparsity']
        self.sparsity_dict = None
        sparsity_dict_path = self.sparsity_config['weight'].get('sparsity_dict_path', None)
        if sparsity_dict_path:
            with open(sparsity_dict_path, "r", encoding="utf-8") as f:
                self.sparsity_dict = json.load(f)
        self.N = self.sparsity_config['weight'].get('N', -1)
        self.M = self.sparsity_config['weight'].get('M', -1)
        self.block_sparsity_config = self.sparsity_config['weight'].get('block_sparsity', False)
        if self.N > 0 or self.M > 0:
            assert self.sparsity_dict is None and not isinstance(self.sparsity, list), 'Can not use non-uniform sparsity.'
            assert isinstance(self.N, int) and isinstance(self.M, int) and self.N > 0 and self.M > 0, \
                'Invalid N:M sparsity: N and M must be positive integers.'
            assert self.N < self.M, 'Invalid N:M sparsity: N must be strictly less than M.'
            if abs(self.N / self.M - self.sparsity) > 1e-3:
                logger.warning('N:M pattern determines sparsity; ignoring configured sparsity.')
            if self.block_sparsity_config:
                raise ValueError('N:M sparsity and block sparsity cannot be enabled at the same time.')
        if self.block_sparsity_config:
            assert isinstance(self.block_sparsity_config, dict) and \
                {'block_height', 'block_width'} <= self.block_sparsity_config.keys(), \
                'block_sparsity must be a dict with block_height and block_width.'
            self.block_height = self.block_sparsity_config.get('block_height', 1)
            self.block_width = self.block_sparsity_config.get('block_width', 1)
            self.block_saliency_metric = self.block_sparsity_config.get('block_saliency_metric', 'absmean')
            assert self.block_saliency_metric in ('absmean', 'absmax')

    def block_forward(self, block, input_data=None):
        output = []
        if input_data is None:
            input_data = self.input['data']
            input_kwargs = self.input['kwargs']
        for i in range(len(input_data)):
            block_device = module_device(block)
            input_data[i] = to_device(input_data[i], block_device)
            input_kwargs[i] = to_device(input_kwargs[i], block_device)
            with torch.no_grad():
                out = block(input_data[i], **self.input['kwargs'][i])[0]
                output.append(out)
        return output

    def optimize_block(self, block):
        to_device(block, torch.device('cuda'))
        if not self.data_free:
            named_linears = self.model.get_block_linears(block)
            logger.info(f'named_linears: {named_linears}')
            input_feat = defaultdict(list)
            output_feat = defaultdict(list)
            handles = self.register_hooks(named_linears, input_feat, output_feat)
            if not self.error_accumulation:
                self.input['data'] = self.block_forward(block)
            else:
                self.block_forward(block)
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()
            self.optimize_block_subsets(block, input_feat, output_feat, self.input['kwargs'])
            if self.error_accumulation:
                self.input['data'] = self.block_forward(block)
            block = block.cpu()
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.optimize_block_subsets(block, None, None)

    def optimize_block_subsets(self, block, input_feat, output_feat, block_kwargs):
        logger.info(f'Start transform the {self.block_idx+1}-th block')
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_module = subset['inspect']
            inspect_has_kwargs = subset['has_kwargs']
            subset_kwargs = block_kwargs if inspect_has_kwargs else {}
            self.optimize_subset(
                layers_dict,
                input_feat,
                output_feat,
                prev_op,
                input_name,
                inspect_module,
                self.block_idx,
                subset_kwargs,
            )
        logger.info(f'End transform the {self.block_idx+1}-th block')

    def optimize_subset(self, layers_dict, input_feat, output_feat, prev_op, input_name, inspect_module, block_idx, subset_kwargs):
        pass

    def save_optimization_metadata(self):
        sparse_mask_save_dir = self.global_config.save.get('save_optimization_metadata_path', None)
        if sparse_mask_save_dir:
            if self.optimized:
                os.makedirs(sparse_mask_save_dir, exist_ok=False)
                torch.save(
                    {k: v.detach().cpu() for k, v in self.W_mask.items()},
                    os.path.join(sparse_mask_save_dir, "sparse_mask.pt")
                )
                logger.info(f'Sparse mask saved to {sparse_mask_save_dir}.')
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Optimization metadata did not saved.')

    def save_transformed_model(self):
        transformed_model_save_dir = self.global_config.save.get('save_transformed_path', None)
        if transformed_model_save_dir:
            if self.optimized:
                os.makedirs(transformed_model_save_dir, exist_ok=False)
                self.model.model.save_pretrained(transformed_model_save_dir)
                self.model.tokenizer.save_pretrained(transformed_model_save_dir)
                logger.info(f"Transformed model & tokenizer saved to {transformed_model_save_dir}.")
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Transformed model did not saved.')

    def save_optimized_model(self):
        optimized_model_save_dir = self.global_config.save.get('save_optimized_path', None)
        if optimized_model_save_dir:
            if self.optimized:
                os.makedirs(optimized_model_save_dir, exist_ok=False)
                self.model.tokenizer.save_pretrained(optimized_model_save_dir)
                from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained
                modify_save_pretrained(self.model.model) 
                self.model.model.save_pretrained(
                    save_directory=optimized_model_save_dir, 
                    save_compressed=True,
                    skip_sparsity_compression_stats=False,
                )
                logger.info(f"Optimized model & tokenizer saved to {optimized_model_save_dir}.")
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Optimized model did not saved.')
