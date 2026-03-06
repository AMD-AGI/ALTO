# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/sparsification/wanda/base.py
# licensed under the Apache License 2.0

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers.pruning.base import PruningModifierBase
from modeloptimizer.observers.per_channel_norm import PRECISION, PerChannelNormObserver
from modeloptimizer.utils.pytorch.module import TransformerConv1D

__all__ = ["WandaStructuredModifier"]


class WandaStructuredModifier(PruningModifierBase):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

    Sample yaml:

    ```yaml
    pruning_modifiers:
        WandaStructuredModifier:
            sparsity: 0.5
            pruning_dimension: "mlp"
            targets: ['Linear']
            ignore: ['re:.*lm_head']
    ```

    Lifecycle:

    - on_initialize
        - register_hook(module, calibrate_module, "forward")
    - on_sequential_batch_end
        - prune_weight
    - on_finalize
        - remove_hooks()

    :param sparsity: Sparsity to prune model to
    :param pruning_dimension: Dimension to prune
    :param targets: list of layer names to prune during WandaStructured, or '__ALL__'
        to prune every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        prune even if they match a target. Defaults to empty list.
    """

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._observer_name = "per_channel_norm"
        assert self.pruning_dimension in ["mlp", "attn+mlp", "mlp+attn", "attn"], \
            "WandaStructuredModifier only supports mlp and/or mlp pruning."
        return super().on_initialize(model_parts, **kwargs)

    def compress_modules(self):
        """
        Prune modules which have been calibrated
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples
            
            logger.info(f"Pruning {name} using {num_samples.item()} samples")
            assert isinstance(observer, PerChannelNormObserver), \
                "WandaPruningModifier requires per_channel_norm observer"
            
            # Determine pruning method based on dimension and layer name
            prune_fn = None
            if 'mlp' in self.pruning_dimension and 'w3' in name:
                prune_fn = self._prune_mlp
            elif 'attn' in self.pruning_dimension and 'wo' in name:
                prune_fn = self._prune_attn
            
            if prune_fn:
                pruned_weight, pruned_mask = prune_fn(
                    module=module,
                    row_scalar=observer.stats,
                    sparsity=sparsity,
                )
                module.weight.data.copy_(pruned_weight)

    def _observe_preprocess(
        self, 
        module: Module, 
        row_scalar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """
        Compute Wanda importance metric: |W| * sqrt(S).
        
        :param module: module containing weight to analyze
        :param row_scalar: activation statistics from calibration
        :return: (W, W_metric, final_shape, final_dtype)
        """
        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.data.clone()
        
        # Handle special layer types
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)
        if TransformerConv1D and isinstance(module, TransformerConv1D):
            W = W.t()
        
        # Compute Wanda metric in higher precision
        W = W.to(dtype=PRECISION)
        W_metric = torch.abs(W) * torch.sqrt(row_scalar.reshape((1, -1)))
        
        return W, W_metric, final_shape, final_dtype

    def _prune_mlp(
        self,
        module: Module,
        row_scalar: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune MLP layer by removing entire output channels.

        :param module: module to prune
        :param row_scalar: row scalar from calibration
        :param sparsity: target sparsity (fraction of channels to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, W_metric, final_shape, final_dtype = self._observe_preprocess(module, row_scalar)
        
        # Average metric across output dimension to get per-channel importance
        W_metric = W_metric.mean(dim=0)
        
        # Select channels to prune (those with lowest importance)
        num_prune = int(sparsity * W_metric.numel())
        idx = W_metric.topk(num_prune, largest=False).indices
        
        # Create mask and apply pruning
        W_mask = torch.zeros(W_metric.shape, dtype=torch.bool, device=W.device)
        W_mask[idx] = True
        W[:, idx] = 0
        
        return W.reshape(final_shape).to(final_dtype), W_mask

    def _prune_attn(
        self,
        module: Module,
        row_scalar: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune attention layer by removing entire attention heads (grouped by KV heads).

        :param module: module to prune
        :param row_scalar: row scalar from calibration
        :param sparsity: target sparsity (fraction of KV heads to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, W_metric, final_shape, final_dtype = self._observe_preprocess(module, row_scalar)
        
        # Get model dimensions
        num_attention_heads = self._model_args.layer.attention.n_heads
        num_key_value_heads = getattr(self._model_args.layer.attention, "n_kv_heads", num_attention_heads)
        hidden_dim = W.shape[1] // num_attention_heads
        group_size = num_attention_heads // num_key_value_heads
        
        # Compute per-head importance, then average over groups
        W_metric = W_metric.view(-1, num_attention_heads, hidden_dim).mean(dim=(0, 2))
        W_metric_group = W_metric.view(num_key_value_heads, group_size).mean(dim=1)
        
        # Select KV head groups to prune
        num_prune = int(sparsity * num_key_value_heads)
        group_idx = W_metric_group.topk(num_prune, largest=False).indices
        
        # Create mask and apply pruning
        W_mask = torch.zeros(num_attention_heads, dtype=torch.bool, device=W.device)
        for g in group_idx:
            start = g * group_size
            end = min(start + group_size, num_attention_heads)
            W_mask[start:end] = True
        
        # Zero out pruned channels
        idx = W_mask.nonzero(as_tuple=True)[0]
        channels = idx[:, None] * hidden_dim + torch.arange(hidden_dim, device=idx.device)
        W[:, channels.reshape(-1)] = 0
        
        # Return per-KV-head mask
        W_mask = W_mask[::group_size]
        return W.reshape(final_shape).to(final_dtype), W_mask