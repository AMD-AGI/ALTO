import torch
import torch.nn as nn
from loguru import logger

from src.utils import ALGO_REGISTRY

from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class Magnitude(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, input):
        super().__init__(model, sparsity_config, input)

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs,
    ):
        layers = list(layers_dict.values())
        for layer in layers:
            W_metric = torch.abs(layer.weight.data)
            W_mask = (torch.zeros_like(W_metric) == 1)

            if self.N > 0:  # semi-structured n:m sparsity
                for input_idx in range(W_metric.shape[1]):
                    if input_idx % self.M == 0:
                        tmp = W_metric[:, input_idx: (input_idx + self.M)].float()
                        W_mask.scatter_(1, input_idx + torch.topk(tmp, self.N, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity)]
                W_mask.scatter_(1, indices, True)
            layer.weight.data[W_mask] = 0  