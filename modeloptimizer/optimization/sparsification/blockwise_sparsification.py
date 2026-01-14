import gc
import os
from typing import TYPE_CHECKING, Optional, Iterable
from collections import defaultdict

import torch
from torchtitan.tools.logging import logger

from modeloptimizer.optimization.model_optimizer import BlockwiseOptimizer
if TYPE_CHECKING:
    from modeloptimizer.config import JobConfig, SparsificationConfig


class BlockwiseSparsification(BlockwiseOptimizer):

    def __init__(
        self,
        job_config: "JobConfig",
        *args,
        **kwargs,
    ):
        super().__init__(job_config, *args, **kwargs)

        self.sparsity_config: "SparsificationConfig" = job_config.sparsification
        self.error_accumulation = self.sparsity_config.error_accumulation
        logger.info(f'sparsity_config: {self.sparsity_config}')

        self.W_mask = {}
        # TODO: support layer-wise sparsity configuration
        self.sparsity = self.sparsity_config.sparsity
        self.N = self.sparsity_config.N
        self.M = self.sparsity_config.M
        self.block_sparsity_config = self.sparsity_config.block_sparsity
        if self.N > 0 or self.M > 0:
            assert isinstance(self.N, int) and isinstance(self.M, int) and self.N > 0 and self.M > 0, \
                'Invalid N:M sparsity: N and M must be positive integers.'
            assert self.N < self.M, 'Invalid N:M sparsity: N must be strictly less than M.'
            if abs(self.N / self.M - self.sparsity) > 1e-3:
                logger.warning(
                    'N:M pattern determines sparsity; ignoring configured sparsity.'
                )
            if self.block_sparsity_config:
                raise ValueError(
                    'N:M sparsity and block sparsity cannot be enabled at the same time.'
                )
        if self.block_sparsity_config:
            self.block_height = self.sparsity_config.block_height
            self.block_width = self.sparsity_config.block_width
            self.block_saliency_metric = self.sparsity_config.block_saliency_metric

    def get_block_linears(self, block):
        return {
            name: m
            for name, m in block.named_modules()
            if isinstance(m, torch.nn.Linear)
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attention.wq': block.attention.wq,
                    'attention.wk': block.attention.wk,
                    'attention.wv': block.attention.wv,
                },
                'prev_op': [block.attention_norm],
                'input': ['attention.wq'],
                'inspect': block.attention,
            },
            {
                'layers': {
                    'attention.wo': block.attention.wo
                },
                'prev_op': [block.attention.wv],
                'input': ['attention.wo'],
                'inspect': block.attention.wo,
            },
            {
                'layers': {
                    'feed_forward.w1': block.feed_forward.w1,
                    'feed_forward.w3': block.feed_forward.w3,
                },
                'prev_op': [block.ffn_norm],
                'input': ['feed_forward.w1'],
                'inspect': block.feed_forward,
                'is_mlp': True,
            },
            {
                'layers': {
                    'feed_forward.w2': block.feed_forward.w2
                },
                'prev_op': [block.feed_forward.w3],
                'input': ['feed_forward.w2'],
                'inspect': block.feed_forward.w2,
                'is_mlp': True,
            },
        ]

    def optimize_block(self, block, block_idx: str):
        if not self.data_free:
            named_linears = self.get_block_linears(block)
            # logger.info(f'named_linears: {named_linears}')

            for input_dict in self.input_batch:
                input_feat = defaultdict(list)
                output_feat = defaultdict(list)
                handles = self.register_hooks(named_linears, input_feat,
                                              output_feat)
                # TODO: call model.eval()?
                with torch.no_grad():
                    self.forward_step(input_dict)

                # TODO: error_accumulation
                # if not self.error_accumulation:
                #     self.inputs['data'] = self.block_forward(block)
                # else:
                #     self.block_forward(block)
                for h in handles:
                    h.remove()
                with torch.no_grad():
                    self.optimize_block_subsets(block, block_idx, input_feat,
                                                output_feat)
                # if self.error_accumulation:
                #     self.inputs['data'] = self.block_forward(block)

                del input_feat, output_feat
        else:
            self.optimize_block_subsets(block, block_idx, None, None)

    def optimize_block_subsets(self, block, block_idx: str, input_feat,
                               output_feat):
        logger.info(f'Start transform the block #{block_idx}')
        subsets = self.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            prev_op = subset['prev_op']
            layers_dict = subset['layers']
            input_name = subset['input'][0]
            inspect_module = subset['inspect']
            self.optimize_subset(
                layers_dict,
                input_feat,
                output_feat,
                prev_op,
                input_name,
                inspect_module,
                block_idx,
            )
        logger.info(f'End transform the #{block_idx}')

    def optimize_subset(self, layers_dict, input_feat, output_feat, prev_op,
                        input_name, inspect_module, block_idx, subset_kwargs):
        pass

    def save_optimization_metadata(self):
        sparse_mask_save_dir = self.global_config.save.get(
            'save_optimization_metadata_path', None)
        if sparse_mask_save_dir:
            if self.optimized:
                os.makedirs(sparse_mask_save_dir, exist_ok=False)
                torch.save({
                    k: v.detach().cpu() for k, v in self.W_mask.items()
                }, os.path.join(sparse_mask_save_dir, "sparse_mask.pt"))
                logger.info(f'Sparse mask saved to {sparse_mask_save_dir}.')
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Optimization metadata did not saved.')

    def save_transformed_model(self):
        transformed_model_save_dir = self.global_config.save.get(
            'save_transformed_path', None)
        if transformed_model_save_dir:
            if self.optimized:
                os.makedirs(transformed_model_save_dir, exist_ok=False)
                self.model.model.save_pretrained(transformed_model_save_dir)
                self.model.tokenizer.save_pretrained(transformed_model_save_dir)
                logger.info(
                    f"Transformed model & tokenizer saved to {transformed_model_save_dir}."
                )
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Transformed model did not saved.')

    def save_optimized_model(self):
        optimized_model_save_dir = self.global_config.save.get(
            'save_optimized_path', None)
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
                logger.info(
                    f"Optimized model & tokenizer saved to {optimized_model_save_dir}."
                )
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Optimized model did not saved.')
