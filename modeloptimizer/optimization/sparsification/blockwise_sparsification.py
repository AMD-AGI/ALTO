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
        if block is None:
            return None
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

                for h in handles:
                    h.remove()
                with torch.no_grad():
                    self.optimize_block_subsets(block, block_idx, input_feat,
                                                output_feat)

                del input_feat, output_feat
                torch.cuda.empty_cache()
                gc.collect()
        else:
            self.optimize_block_subsets(block, block_idx, None, None)

    def optimize_block_subsets(self, block, block_idx: str, input_feat,
                               output_feat):
        if block is None:
            return
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
