import functools
import gc
import os
from collections import defaultdict

import torch
from loguru import logger

from src.utils import module_device, to_device

from ..blockwise_optimization import BlockwiseOptimizer


class BlockwisePruning(BlockwiseOptimizer):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)
        self.pruning_config = self.optimization_config
        self.error_accumulation = self.pruning_config.get('error_accumulation', False)
        logger.info(f'use error_accumulation {self.error_accumulation}')
        self.sparsity = self.pruning_config['weight']['sparsity']
        self.W_mask = {}
        # pruning dimensions
        self.prune_attn      = self.pruning_config['weight'].get('prune_attn', False)
        self.prune_mlp       = self.pruning_config['weight'].get('prune_mlp', False)
        self.prune_layer     = self.pruning_config['weight'].get('prune_layer', False)
        self.prune_sublayer  = self.pruning_config['weight'].get('prune_sublayer', False)
        self.prune_embedding = self.pruning_config['weight'].get('prune_embedding', False)

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
            handles = self.register_hooks(named_linears, input_feat)
            if not self.error_accumulation:
                self.input['data'] = self.block_forward(block)
            else:
                self.block_forward(block)
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()
            self.optimize_block_subsets(block, input_feat, self.input['kwargs'])
            if self.error_accumulation:
                self.input['data'] = self.block_forward(block)
            block = block.cpu()
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        else:
            self.optimize_block_subsets(block, None, None)

    def optimize_block_subsets(self, block, input_feat, block_kwargs):
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
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs
            )
        logger.info(f'End transform the {self.block_idx+1}-th block')

    def optimize_subset(self, layers_dict, input_feat, prev_op, input_name, inspect_module, subset_kwargs):
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
        pass
