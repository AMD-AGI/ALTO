# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/sparsification/wanda/base.py
# licensed under the Apache License 2.0

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers.pruning.base import PruningModifierBase
from modeloptimizer.observers.hessian import PRECISION, HessianObserver
from modeloptimizer.utils.pytorch.module import TransformerConv1D

__all__ = ["ObsModifier"]


class ObsModifier(PruningModifierBase):
    """
    Modifier for applying the one-shot OBS algorithm to a model

    Sample yaml:

    ```yaml
    pruning_modifiers:
        ObsModifier:
            sparsity: 0.5
            pruning_dimension: "mlp"
            mlp_pruning_granularity: 8
            attention_pruning_granularity: 1
            dampening_frac: 0.01
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
    :param targets: list of layer names to prune during Obs, or '__ALL__'
        to prune every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        prune even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    mlp_pruning_granularity: int = 16
    attention_pruning_granularity: int = 1
    dampening_frac: float | None = 0.01

    def on_initialize(self, model: Module, **kwargs) -> bool:
        self._observer_name = "hessian"
        assert self.pruning_dimension in ["mlp", "attn+mlp", "mlp+attn", "attn"], \
            "ObsModifier only supports mlp and/or mlp pruning."
        return super().on_initialize(model, **kwargs)

    def compress_modules(self):
        """
        Prune modules which have been calibrated (Hessian stats collected).
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples
            
            logger.info(f"Pruning {name} using {num_samples.item()} samples")
            assert isinstance(observer, HessianObserver), \
                "ObsModifier requires hessian observer"
            
            # Determine pruning method based on dimension and layer name
            prune_fn = None
            if 'mlp' in self.pruning_dimension and 'w3' in name:
                prune_fn = self._prune_mlp
            elif 'attn' in self.pruning_dimension and 'wo' in name:
                prune_fn = self._prune_attn
            
            if prune_fn:
                pruned_weight, pruned_mask = prune_fn(
                    module=module,
                    hessian=observer.stats,
                    sparsity=sparsity,
                )
                module.weight.data.copy_(pruned_weight)

    def _observe_preprocess(
        self, 
        module: Module, 
        hessian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """
        Preprocess weight and Hessian for OBS pruning.
        
        :param module: module containing weight
        :param hessian: Hessian matrix from calibration
        :return: (W, H, G, final_shape, final_dtype)
            - W: transposed weight in high precision
            - H: regularized Hessian matrix
            - G: gradient matrix H @ W
            - final_shape: original weight shape
            - final_dtype: original weight dtype
        """
        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.data.clone()
        
        # Handle special layer types
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)
        if TransformerConv1D and isinstance(module, TransformerConv1D):
            W = W.t()
        
        # Transpose and convert to higher precision for computation
        W = W.to(dtype=PRECISION).t()
        H = hessian
        
        # Zero out dead neurons (those with zero Hessian diagonal)
        dead = torch.diag(H) == 0
        W[dead, :] = 0
        
        # Add regularization to Hessian for numerical stability
        H += torch.eye(W.shape[0]).to(W.device) * torch.mean(torch.diag(H)) * self.dampening_frac
        
        # Compute gradient matrix
        G = H @ W
        
        return W, H, G, final_shape, final_dtype

    def _prune_mlp(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune MLP layer by removing entire output channels.

        :param module: module to prune
        :param hessian: Hessian matrix from calibration
        :param sparsity: target sparsity (fraction of channels to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, H, G, final_shape, final_dtype = self._observe_preprocess(module, hessian)
        
        # Prune entire channels
        num_total_groups = W.shape[0]
        num_groups_to_remain = int(num_total_groups * (1 - sparsity))
        W_pruned, error, pruned_mask = self._prune_linear(
            W, H, G, num_total_groups, num_groups_to_remain, update_iter=self.mlp_pruning_granularity
        )
        
        return W_pruned.t().to(final_dtype).reshape(final_shape), pruned_mask

    def _prune_attn(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune attention layer by removing entire attention heads (grouped by KV heads).

        :param module: module to prune
        :param hessian: Hessian matrix from calibration
        :param sparsity: target sparsity (fraction of KV heads to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, H, G, final_shape, final_dtype = self._observe_preprocess(module, hessian)
        
        # Get model dimensions for grouped attention head pruning
        num_attention_heads = self._model_args.n_heads
        num_key_value_heads = self._model_args.n_kv_heads
        num_total_groups = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        num_groups_to_remain = int(num_total_groups * (1 - sparsity))
        
        W_pruned, error, pruned_mask = self._prune_linear(
            W, H, G, num_total_groups, num_groups_to_remain, update_iter=self.attention_pruning_granularity
        )
        
        return W_pruned.t().to(final_dtype).reshape(final_shape), pruned_mask

    @torch.no_grad()
    def _prune_linear(self, W, H, G, num_total_groups: int, num_groups_to_remain: int, pruning_granularity: int):
        """
        Core OBS (Optimal Brain Surgeon) pruning algorithm for group-structured pruning.
        
        This implements iterative group pruning using second-order information (Hessian).
        Groups with the smallest impact on the loss are pruned first, and remaining weights
        are adjusted to compensate for the pruned groups.
        
        :param W: Weight matrix (transposed, shape [in_features, out_features])
        :param H: Hessian matrix (shape [in_features, in_features])
        :param G: Gradient matrix H @ W (shape [in_features, out_features])
        :param num_total_groups: Total number of groups in the layer
        :param num_groups_to_remain: Number of groups to keep after pruning
        :param pruning_granularity: Number of iterative pruning rounds (default 32)
        :return: (pruned_weight, pruning_loss, pruned_mask)
        """
        device = W.device
        cin, cout = W.shape
        group_size = int(cin / num_total_groups)
        # Full Hessian inverse; downdated as we prune (OBS uses H^{-1} for closed-form weight update)
        H_inv = torch.linalg.inv(H)
        # Per-group L1 norm of W; mark already-zero groups (e.g. dead neurons)
        W_g = W.reshape(num_total_groups, group_size, cout)
        group_abs_sum = torch.sum(torch.abs(W_g), dim=(1, 2))
        pruned_group_mask = (group_abs_sum <= 1e-12)
        num_already_zero = int(pruned_group_mask.sum().item())
        if num_already_zero > 0:
            # Zero rows/cols of H_inv for already-pruned groups; re-solve W = H_inv @ G
            zero_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in torch.nonzero(pruned_group_mask, as_tuple=False).flatten()
            ])
            H_inv[zero_idx, :] = 0
            H_inv[:, zero_idx] = 0
            W = H_inv @ G
            # If target is satisfied by pruning only already-zero groups, return
            if (num_total_groups - num_groups_to_remain - num_already_zero) <= 0:
                W[zero_idx, :] = 0
                return W, 0.0
        remaining_to_prune = int(num_total_groups - num_groups_to_remain - int(pruned_group_mask.sum().item()))
        if remaining_to_prune <= 0:
            # Nothing to prune: optimal W on kept indices only, then return
            kept_mask = (~torch.repeat_interleave(pruned_group_mask, group_size))
            W_kept = torch.zeros_like(W)
            try:
                W_kept[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
            except Exception:
                H_cpu = H[kept_mask][:, kept_mask].cpu()
                G_cpu = G[kept_mask, :].cpu()
                W_kept[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)
            prune_loss = torch.sum(-W_kept * G + 0.5 * W_kept * (H @ W_kept)).detach().item()
            return W_kept, prune_loss
        # Distribute remaining_to_prune across rounds (roughly equal per round)
        update_rounds = max(int(min(pruning_granularity, remaining_to_prune)), 1)
        base, extra = divmod(remaining_to_prune, update_rounds)
        groups_to_prune_each_round = torch.full((update_rounds,), base, dtype=torch.int, device=device)
        if extra > 0:
            groups_to_prune_each_round[:extra] += 1
        for round_id in range(update_rounds):
            # OBS saliency per group: (1/2) w_g^T H_gg^{-1} w_g; smallest = prune first
            if group_size > 1:
                obj_mat = torch.zeros_like(W)
                for g in range(num_total_groups):
                    if pruned_group_mask[g]:
                        continue
                    sl = slice(g * group_size, (g + 1) * group_size)
                    H_block = torch.linalg.inv(H_inv[sl, sl]) 
                    obj_mat[sl, :] = (H_block @ W[sl, :] / 2.0)
            else:
                # Scalar per row: (1/2) * w^2 / H_inv_ii
                diag_Hinv = torch.diag(H_inv)
                safe_den = (pruned_group_mask.to(W.dtype) + diag_Hinv).clamp_min(1e-12)
                obj_mat = (1.0 / safe_den)[:, None] * (W / 2.0)
            obj_val = (W * obj_mat).reshape(num_total_groups, group_size, cout).sum(dim=(1, 2))
            obj_val_masked = obj_val + 1e20 * pruned_group_mask.to(obj_val.dtype)
            sorted_groups = torch.argsort(obj_val_masked)
            k = int(groups_to_prune_each_round[round_id].item())
            pick_groups = sorted_groups[:k]
            pick_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in pick_groups
            ])
            # OBS update: remove pruned weights' contribution; downdate H_inv (block Sherman-Morrison)
            Hinv_block_inv = torch.linalg.inv(H_inv[pick_idx][:, pick_idx])  # ≈ H[pick_idx, pick_idx]
            W -= H_inv[:, pick_idx] @ Hinv_block_inv @ W[pick_idx, :]
            W[pick_idx, :] = 0
            H_inv -= H_inv[:, pick_idx] @ Hinv_block_inv @ H_inv[pick_idx, :]
            H_inv[pick_idx, :] = 0
            H_inv[:, pick_idx] = 0
            pruned_group_mask[pick_groups] = True
        # Final closed-form on kept support: W_kept = H_kept^{-1} @ G_kept
        W_pruned = torch.zeros_like(W)
        kept_mask = (~torch.repeat_interleave(pruned_group_mask, repeats=group_size))
        try:
            W_pruned[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
        except Exception:
            H_cpu = H[kept_mask][:, kept_mask].cpu()
            G_cpu = G[kept_mask, :].cpu()
            W_pruned[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)
        prune_loss = torch.sum(-W_pruned * G + 0.5 * W_pruned * (H @ W_pruned)).detach().item()
        return W_pruned, prune_loss, pruned_group_mask