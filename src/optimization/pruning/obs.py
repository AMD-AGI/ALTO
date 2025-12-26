from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ALGO_REGISTRY
from .blockwise_pruning import BlockwisePruning


@ALGO_REGISTRY
class OBS(BlockwisePruning):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)

    @torch.no_grad()
    def get_row_scale(self, layer, act, scaler_row):
        if isinstance(act, list):
            act = torch.cat(act, dim=0)
        if len(act.shape) == 2:
            act = act.unsqueeze(0)
        if isinstance(layer, nn.Linear):
            if len(act.shape) == 3:
                act = act.reshape((-1, act.shape[-1]))
            act = act.t()

        act = act.type(torch.float32).to(scaler_row.device)
        scaler_row += torch.norm(act, p=2, dim=1) ** 2
        return scaler_row

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
        for layer_name, layer in layers_dict.items():
            columns = layer.weight.data.shape[1]
            scaler_row = torch.zeros((columns), device=layer.weight.device)
            nsamples = 0
            for batch_idx in range(len(input_feat[input_name])):
                scaler_row = self.get_row_scale(layer, input_feat[input_name][batch_idx], scaler_row)
                nsamples += input_feat[input_name][batch_idx].shape[0]
            scaler_row /= nsamples
            W_metric = torch.abs(layer.weight.data) * torch.sqrt(scaler_row.reshape((1, -1)))
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