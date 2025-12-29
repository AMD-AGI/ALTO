from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ALGO_REGISTRY
from .blockwise_pruning import BlockwisePruning


@ALGO_REGISTRY
class WandaStructured(BlockwisePruning):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)
        self.optimization_method_name = 'WandaStructured'
        self.applicability_error_message = 'Wanda (structured pruning) is only suitable for structured pruning of attn heads and mlp neurons.'
        assert self.prune_embedding == False, self.applicability_error_message
        assert self.prune_layer == False, self.applicability_error_message
        assert self.prune_sublayer == False, self.applicability_error_message

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
    def optimize_subset(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs,
    ):
        for layer_name, layer in layers_dict.items():        
            if layer_name not in (self.model.get_out_projection_name(), self.model.get_down_projection_name(),):
                continue
            else:
                columns = layer.weight.data.shape[1]
                scaler_row = torch.zeros((columns), device=layer.weight.device)
                nsamples = 0
                for batch_idx in range(len(input_feat[input_name])):
                    scaler_row = self.get_row_scale(layer, input_feat[input_name][batch_idx], scaler_row)
                    nsamples += input_feat[input_name][batch_idx].shape[0]
                scaler_row /= nsamples
                W_metric = torch.abs(layer.weight.data) * torch.sqrt(scaler_row.reshape((1, -1)))

                if layer_name == self.model.get_out_projection_name():
                    if self.prune_attn == False:
                        continue
                    num_attention_heads = getattr(self.model.model.config, "num_attention_heads", None)
                    num_key_value_heads = getattr(self.model.model.config, "num_key_value_heads", None)
                    W_mask = (torch.zeros(num_attention_heads, device=layer.weight.device) == 1)
                    hidden_dim = layer.weight.data.shape[1] // num_attention_heads
                    W_metric = W_metric.view(-1, num_attention_heads, hidden_dim).mean(dim=(0, 2))
                    group_size = num_attention_heads // num_key_value_heads
                    W_metric_group = W_metric.view(num_key_value_heads, group_size).mean(dim=1) 
                    group_idx = W_metric_group.topk(k=int(self.sparsity * num_key_value_heads), largest=False).indices
                    for g in group_idx:
                        start = g * group_size
                        end = min(start + group_size, num_attention_heads)
                        W_mask[start:end] = True
                    idx = W_mask.nonzero(as_tuple=True)[0]
                    channels = idx[:, None] * hidden_dim + torch.arange(hidden_dim, device=idx.device)
                    layer.weight.data[:, channels.reshape(-1)] = 0
                else:
                    if self.prune_mlp == False:
                        continue
                    W_mask = (torch.zeros(W_metric[1].shape, device=layer.weight.device) == 1)
                    W_metric = W_metric.mean(dim=0)
                    idx = W_metric.topk(int(self.sparsity * W_metric.numel()), largest=False).indices                    
                    W_mask[idx] = True
                    layer.weight.data[:, idx] = 0
            
            self.W_mask[layer_name] = W_mask