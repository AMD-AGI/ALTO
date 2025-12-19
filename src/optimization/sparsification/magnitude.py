import torch
import torch.nn as nn
from loguru import logger

from src.utils import ALGO_REGISTRY

from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class Magnitude(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)

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

            if self.N > 0:
                # semi-structured n:m sparsity
                W_height, W_width = W_metric.shape
                pad = (-W_width) % self.M  
                W_metric_paded = F.pad(W_metric.float(), (0, pad), value=0.0)  
                if pad:
                    W_metric_paded[:, W_width:] = float("inf") 
                Groups = W_metric_paded.shape[1] // self.M
                W_metric_paded = W_metric_paded.view(W_height, Groups, self.M)
                idx = W_metric_paded.topk(self.M - self.N, dim=2, largest=False).indices  
                mask = torch.zeros((W_height, Groups, self.M), dtype=torch.bool, device=W_metric.device)
                mask.scatter_(2, idx, True)
                W_mask = mask.view(W_height, Groups * self.M)[:, :W_width] 
            elif self.block_sparsity_config:
                W_height, W_width = W_metric.shape
                assert W_height % self.block_height == 0 and W_width % self.block_width == 0
                # (n_block_rows, n_block_columns, block_height, block_width)
                blocks = W_metric.reshape(W_height // self.block_height, self.block_height, W_width // self.block_width, self.block_width).permute(0, 2, 1, 3)
                score  = blocks.mean((2, 3)) if self.block_saliency_metric == "absmean" else blocks.amax((2, 3))
                k = int(score.numel() * self.sparsity)
                k = max(0, min(k, score.numel()))
                prune_blocks = torch.zeros_like(score, dtype=torch.bool)
                idx = score.flatten().topk(k, largest=False).indices
                prune_blocks.view(-1)[idx] = True
                W_mask = prune_blocks.repeat_interleave(self.block_height, 0).repeat_interleave(self.block_width, 1)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity)]
                W_mask.scatter_(1, indices, True)
            
            layer.weight.data[W_mask] = 0
            self.W_mask[layer_name] = W_mask