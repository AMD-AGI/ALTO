import functools
import gc
import json
import os
from collections import defaultdict
 
import torch
import torch.nn as nn
from loguru import logger
 
from src.utils import module_device, to_device
 
from ..blockwise_optimization import BlockwiseOptimizer
from .structured_mask import ModelDims, StructuredPruningMask
 
 
class BlockwisePruning(BlockwiseOptimizer):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)
        self.pruning_config = self.optimization_config
        self.error_accumulation = self.pruning_config.get('error_accumulation', False)
        logger.info(f'use error_accumulation {self.error_accumulation}')
        self.sparsity = self.pruning_config['weight']['sparsity']
        self.sparsity_dict = None
        sparsity_dict_path = self.pruning_config['weight'].get('sparsity_dict_path', None)
        if sparsity_dict_path:
            with open(sparsity_dict_path, "r", encoding="utf-8") as f:
                self.sparsity_dict = json.load(f)
        cfg = self.model.model.config
        num_q_heads = int(cfg.num_attention_heads)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_q_heads))
        self.model_dims = ModelDims(
            num_layers=len(self.blocks),
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            intermediate_size=int(cfg.intermediate_size),
            hidden_size=int(cfg.hidden_size),
        )
        self.W_mask = StructuredPruningMask(self.model_dims)
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
                subset_kwargs
            )
        logger.info(f'End transform the {self.block_idx+1}-th block')
 
    def optimize_subset(self, layers_dict, input_feat, output_feat, prev_op, input_name, inspect_module, block_idx, subset_kwargs):
        pass
 
    def save_optimization_metadata(self):
        pruning_mask_save_dir = self.global_config.save.get('save_optimization_metadata_path', None)
        if pruning_mask_save_dir:
            if self.optimized:
                os.makedirs(pruning_mask_save_dir, exist_ok=False)
                save_path = os.path.join(pruning_mask_save_dir, "pruning_mask.pt")
                self.W_mask.save(save_path)
                logger.info(f'Pruning mask saved to {save_path}.')
                logger.info(f'Mask summary: {self.W_mask.summary()}')
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
                if self.prune_layer:
                    logger.info(f'Transformed model is already real-optimized, please set save_transformed_path.')
                elif self.prune_attn or self.prune_mlp:
                    self.save_attn_mlp_pruned_model(optimized_model_save_dir)
                else:
                    pass # todo
                logger.info(f"Optimized model & tokenizer saved to {optimized_model_save_dir}.")
            else:
                logger.warning('Please optimize your model first.')
        else:
            logger.warning('Optimized model did not saved.')
 
    @torch.no_grad()
    def prune_linear_out(self, linear: nn.Linear, idx: torch.Tensor):
        idx = idx.to(linear.weight.device)
        W = linear.weight.index_select(0, idx).contiguous()
        pruner_linear = nn.Linear(W.shape[1], W.shape[0], bias=(linear.bias is not None),
                        device=linear.weight.device, dtype=linear.weight.dtype)
        pruner_linear.weight.copy_(W)
        if linear.bias is not None:
            pruner_linear.bias.copy_(linear.bias.index_select(0, idx).contiguous())
        return pruner_linear
 
    @torch.no_grad()
    def prune_linear_in(self, linear: nn.Linear, idx: torch.Tensor):
        idx = idx.to(linear.weight.device)
        W = linear.weight.index_select(1, idx).contiguous()
        pruned_linear = nn.Linear(W.shape[1], W.shape[0], bias=(linear.bias is not None),
                        device=linear.weight.device, dtype=linear.weight.dtype)
        pruned_linear.weight.copy_(W)
        if linear.bias is not None:
            pruned_linear.bias.copy_(linear.bias)
        return pruned_linear
 
    def head_idx(self, head_ids, head_dim):
        # expand head ids -> feature indices in [0, num_heads*head_dim)
        return torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in head_ids.tolist()], dim=0).long()
 
    @torch.no_grad()
    def save_attn_mlp_pruned_model(self, optimized_model_save_dir):
        cfg = self.model.model.config
        head_dim = self.model_dims.head_dim
        kv_groups = self.model_dims.kv_groups
        layers = self.model.blocks

        new_num_kvheads, new_num_heads, new_inter_dimension = None, None, None
        for i, layer in enumerate(layers):
            if self.prune_attn:
                keep_kv = self.W_mask.get_kept_head_indices(i)
                if keep_kv is not None:
                    new_num_kvheads = int(keep_kv.numel())
                    new_num_heads = int(new_num_kvheads * kv_groups)
                    keep_q = self.W_mask.get_kept_q_head_indices(i)
                    q_idx = self.head_idx(keep_q, head_dim)
                    kv_idx = self.head_idx(keep_kv, head_dim)

                    attn = layer.self_attn
                    attn.q_proj = self.prune_linear_out(attn.q_proj, q_idx)
                    attn.k_proj = self.prune_linear_out(attn.k_proj, kv_idx)
                    attn.v_proj = self.prune_linear_out(attn.v_proj, kv_idx)
                    attn.o_proj = self.prune_linear_in(attn.o_proj, q_idx)
                    if hasattr(attn, "num_heads"):            attn.num_heads = new_num_heads
                    if hasattr(attn, "num_key_value_heads"):  attn.num_key_value_heads = new_num_kvheads
                    if hasattr(attn, "num_key_value_groups"): attn.num_key_value_groups = new_num_heads // new_num_kvheads
                    if hasattr(attn, "head_dim"):             attn.head_dim = head_dim

            if self.prune_mlp:
                keep_mlp = self.W_mask.get_kept_neuron_indices(i)
                if keep_mlp is not None:
                    new_inter_dimension = int(keep_mlp.numel())

                    mlp = layer.mlp
                    if hasattr(mlp, "gate_proj"): mlp.gate_proj = self.prune_linear_out(mlp.gate_proj, keep_mlp)
                    if hasattr(mlp, "up_proj"):   mlp.up_proj   = self.prune_linear_out(mlp.up_proj, keep_mlp)
                    mlp.down_proj = self.prune_linear_in(mlp.down_proj, keep_mlp)

        if self.prune_attn and new_num_heads is not None:
            cfg.num_attention_heads = new_num_heads
            if hasattr(cfg, "num_key_value_heads"):
                cfg.num_key_value_heads = new_num_kvheads
            setattr(cfg, "head_dim", head_dim)

        if self.prune_mlp and new_inter_dimension is not None:
            cfg.intermediate_size = new_inter_dimension

        self.model.model.save_pretrained(optimized_model_save_dir)
        self.model.tokenizer.save_pretrained(optimized_model_save_dir)