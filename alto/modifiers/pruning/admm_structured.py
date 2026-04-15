# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/sparsification/wanda/base.py


import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers.pruning.base import PruningModifierBase
from alto.observers.hessian import PRECISION, HessianObserver
from alto.utils.pytorch.module import TransformerConv1D

__all__ = ["AdmmStructuredModifier"]


class AdmmStructuredModifier(PruningModifierBase):
    """
    Modifier for applying the one-shot ADMM algorithm to a model
    from the paper: https://arxiv.org/abs/2401.02938

    Sample yaml:
    ```yaml
    pruning_modifiers:
        AdmmStructuredModifier:
            sparsity: 0.5
            pruning_dimension: "mlp"
            admm_rho: 10
            admm_iterations: 16
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
    :param targets: list of layer names to prune during Admm, or '__ALL__'
        to prune every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        prune even if they match a target. Defaults to empty list.
    """
    # modifier arguments
    admm_rho: int = 10
    admm_iterations: int = 16
    dampening_frac: float | None = 0.01

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._observer_name = "hessian"
        assert self.pruning_dimension in ["mlp", "attn+mlp", "mlp+attn", "attn"], \
            "AdmmStructuredModifier only supports mlp and/or mlp pruning."
        return super().on_initialize(model_parts, **kwargs)

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
                "AdmmStructuredModifier requires hessian observer"
            
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        """
        Preprocess weight and Hessian for ADMM pruning.
        
        This method normalizes the Hessian matrix and weight, adds regularization,
        and prepares necessary matrices for ADMM optimization.
        
        :param module: module containing weight
        :param hessian: Hessian matrix from calibration
        :return: (W, H, G, Hinv, norm, final_shape, final_dtype)
            - W: normalized transposed weight in high precision
            - H: regularized normalized Hessian matrix
            - G: gradient matrix H @ W
            - Hinv: inverse of regularized Hessian
            - norm: normalization factor (sqrt of diagonal)
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
        
        W = W.to(dtype=PRECISION)
        H = hessian
        
        # Normalize Hessian and weight by diagonal sqrt
        norm = torch.diag(H).sqrt() + 1e-8
        H = H / norm
        H = (H.T / norm).T
        W = (W * norm).T
        
        # Add regularization to Hessian for numerical stability
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += self.dampening_frac * torch.diag(H).mean()

        # ADMM regularization parameter and gradient
        rho = self.admm_rho
        G = H.matmul(W)
        H[diag, diag] += rho
        
        # Compute Hessian inverse (expensive operation)
        torch.cuda.empty_cache()
        Hinv = torch.inverse(H)
        
        return W, H, G, Hinv, norm, final_shape, final_dtype

    def _prune_mlp(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune MLP layer by removing entire output channels using ADMM.

        :param module: module to prune
        :param hessian: Hessian matrix from calibration
        :param sparsity: target sparsity (fraction of channels to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, H, G, Hinv, norm, final_shape, final_dtype = self._observe_preprocess(module, hessian)
        
        # ADMM optimization: iteratively prune channels with lowest importance
        # U: dual variable in ADMM, Z: auxiliary variable with sparsity constraint
        rho = 10  # ADMM penalty parameter
        U = torch.zeros_like(W)
        
        for _ in range(self.admm_iterations):  # ADMM iterations
            # Compute channel importance using L2 norm per channel
            channel_importance = (W + U).abs().norm(dim=1)
            
            # Select channels to prune (lowest importance)
            num_channels_to_prune = int(channel_importance.size(0) * sparsity)
            prune_indices = torch.topk(channel_importance, k=num_channels_to_prune, largest=False).indices
            
            # Create mask (False for pruned channels)
            mask = torch.ones_like(W, dtype=torch.bool, device=W.device)
            mask[prune_indices, :] = False
            
            # ADMM update steps
            Z = (W + U) * mask              # Project onto sparsity constraint
            U = U + (W - Z)                  # Update dual variable
            W = Hinv.matmul(G + rho * (Z - U))  # Update primal variable with Hessian info
        
        # Final projection and denormalization
        Z = (W + U) * mask
        pruned_W = (Z.T / norm).reshape(final_shape).to(final_dtype)
        return pruned_W, mask

    def _prune_attn(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prune attention layer by removing entire attention heads using ADMM.

        :param module: module to prune
        :param hessian: Hessian matrix from calibration
        :param sparsity: target sparsity (fraction of heads to prune)
        :return: (pruned_weight, pruned_mask)
        """
        W, H, G, Hinv, norm, final_shape, final_dtype = self._observe_preprocess(module, hessian)
        
        # Get model dimensions for attention head pruning
        num_attention_heads = self._model_args.layer.attention.n_heads
        head_dim = W.shape[1] // num_attention_heads
        
        # ADMM optimization: iteratively prune attention heads with lowest importance
        # U: dual variable in ADMM, Z: auxiliary variable with sparsity constraint
        rho = self.admm_rho  # ADMM penalty parameter
        U = torch.zeros_like(W)
        
        for _ in range(self.admm_iterations):  # ADMM iterations
            # Compute head importance: reshape to [num_heads, head_dim, out_features], then compute norm
            WU_grouped = (W + U).abs().view(num_attention_heads, head_dim, W.shape[1])
            head_importance = WU_grouped.norm(dim=(1, 2))
            
            # Select heads to prune (lowest importance)
            num_heads_to_prune = int(num_attention_heads * sparsity)
            prune_head_indices = torch.topk(head_importance, k=num_heads_to_prune, largest=False).indices
            
            # Create mask (False for pruned heads)
            mask = torch.ones_like(W, dtype=torch.bool, device=W.device)
            mask_grouped = mask.view(num_attention_heads, head_dim, W.shape[1])
            mask_grouped[prune_head_indices, :, :] = False
            mask = mask_grouped.view(W.shape)
            
            # ADMM update steps
            Z = (W + U) * mask                  # Project onto sparsity constraint
            U = U + (W - Z)                     # Update dual variable
            W = Hinv.matmul(G + rho * (Z - U))  # Update primal variable with Hessian info
        
        # Final projection and denormalization
        Z = (W + U) * mask
        pruned_W = (Z.T / norm).reshape(final_shape).to(final_dtype)
        return pruned_W, mask