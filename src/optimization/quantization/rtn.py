import torch
from loguru import logger

from src.utils import ALGO_REGISTRY

from .blockwise_quantization import BlockwiseQuantization


@ALGO_REGISTRY
class RTN(BlockwiseQuantization):
    def __init__(self, model, quant_config, global_config, input):
        super().__init__(model, quant_config, global_config, input)

    @torch.no_grad()
    def block_opt(self, block, *opt_kwargs):
        super().block_opt(block, *opt_kwargs)

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict, 
        input_feat, 
        output_feat, 
        prev_op, 
        input_name, 
        inspect_module, 
        subset_kwargs
    ):
        pass