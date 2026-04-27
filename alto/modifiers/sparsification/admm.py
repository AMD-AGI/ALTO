# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/base.py
#
# ADMM-based one-shot pruning: minimizes (1/2)||H*W - G||^2 s.t. sparsity via
# alternating W-update (linear solve), Z = project(W+U onto mask), U = U + (W - Z).
# Uses Hessian H; supports progressive sparsity schedule and N:M or unstructured masks.

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers.sparsification.base import SparsityModifierBase
from alto.observers.hessian import PRECISION, HessianObserver
from alto.utils.pytorch.module import TransformerConv1D

__all__ = ["AdmmModifier"]


class AdmmModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot ADMM algorithm to a model
    from the paper: https://arxiv.org/abs/2401.02938

    Sample yaml:

    ```yaml
    sparsity_modifiers:
        AdmmModifier:
            sparsity: 0.5
            mask_structure: "2:4"
            dampening_frac: 0.01
            block_size: 128
            targets: ['Linear']
            ignore: ['re:.*lm_head']
    ```

    Lifecycle:

    - on_initialize
        - register_hook(module, calibrate_module, "forward")
    - on_sequential_batch_end
        - sparsify_weight
    - on_finalize
        - remove_hooks()

    :param sparsity: Sparsity to compress model
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param iterative_prune_granularity: Number of iterations to prune the model
    :param iterations: Number of iterations to run the ADMM algorithm
    :param targets: list of layer names to compress during ADMM, or '__ALL__'
        to compress every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    dampening_frac: float | None = 0.01
    admm_rho: int = 10
    admm_pruning_granularity: int = 32
    admm_iterations: int = 32

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        # ADMM uses Hessian of activations; requires HessianObserver for calibration
        self._observer_name = "hessian"
        return super().on_initialize(model_parts, **kwargs)

    def compress_modules(self):
        """
        Sparsify modules that have been calibrated (Hessian stats collected).
        Each layer is compressed via _sparsify_weight and weight is updated in-place.
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples

            logger.info(f"Sparsifying {name} using {num_samples.item()} samples")
            assert isinstance(observer, HessianObserver), "ADMMModifier requires hessian observer"
            sparsified_weight, W_mask = self._sparsify_weight(
                module=module,
                hessian=observer.stats,
                sparsity=sparsity,
                prune_n=self._prune_n,
                prune_m=self._prune_m,
            )
            module.weight.data.copy_(sparsified_weight)

    def _sparsify_weight(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
        prune_n: int,
        prune_m: int,
    ) -> tuple[float, torch.Tensor]:
        """
        Run pruning on the layer up to the target sparsity value.

        :param module: module with weight being sparsified
        :param hessian: Hessian matrix of the activations
        :param sparsity: target sparsity to reach for layer
        :param prune_n: N for N:M pruning
        :param prune_m: M for N:M pruning
        """
        # --- Setup: weight layout and Hessian ---
        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.clone()
        W = W.to(dtype=PRECISION)
        H = hessian
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)
        elif TransformerConv1D and isinstance(module, TransformerConv1D):
            W.transpose_(0, 1)
        num_columns = W.shape[1]

        # Symmetric normalization: H' = H / (sqrt(diag)*sqrt(diag)^T), W' = W * norm (column scaling)
        norm = torch.diag(H).sqrt() + 1e-8
        H = H / norm
        H = (H.T / norm).T
        W = (W * norm).T
        # Dampening for numerical stability
        rho0 = self.dampening_frac * torch.diag(H).mean()
        diag = torch.arange(num_columns, device=H.device)
        H[diag, diag] += rho0

        # ADMM: (H + rho*I) W = G + rho*(Z - U) => W = (H+rho*I)^{-1} (G + rho*(Z-U)); precompute Hinv
        rho = self.admm_rho
        G = H.matmul(W)
        H[diag, diag] += rho
        Hinv = torch.inverse(H)
        U = torch.zeros_like(W)  # dual variable

        for iter in range(self.admm_iterations):
            # Progressive sparsity: cur_sparsity increases from 0 toward target over admm_pruning_granularity iters (cubic schedule)
            if prune_n != 0:
                sparsity = 0.5
                cur_sparsity = sparsity - sparsity * (1 - (iter + 1) / self.admm_pruning_granularity)**3
                # N:M mask: reshape to blocks of size prune_m, keep prune_n largest (by |W+U|) per block
                WT = (W + U).T.reshape((W.shape[1] * W.shape[0] // prune_m, prune_m)).abs()
                W_mask = torch.zeros(WT.shape, dtype=torch.bool, device=W.device)
                sort_inds = WT.sort(dim=1)[1]
                for i in range(prune_m - prune_n, prune_m):
                    W_mask[torch.arange(WT.shape[0]), sort_inds[:, i]] = 1
                W_mask = W_mask.reshape(W.T.shape).T
            else:
                # Unstructured: initial mask = keep (1-sparsity) fraction smallest entries zeroed
                cur_sparsity = sparsity - sparsity * (1 - (iter + 1) / self.admm_pruning_granularity)**3
                topk = torch.topk((W + U).abs().flatten(), k=int(W.numel() * sparsity), largest=False)
                W_mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
                W_mask[topk.indices] = 0
                W_mask = W_mask.reshape(W.shape)
                del topk

            # Refine mask by kth-value threshold: keep entries with |W+U| >= kth largest so that exactly cur_sparsity fraction are nonzero
            local_Z = (W + U).abs()
            local_Z[W_mask] = torch.inf
            thres = local_Z.flatten().kthvalue(int(W.numel() * cur_sparsity) + 1)[0]
            W_mask = (local_Z >= thres)
            del thres

            # Z-update: project (W+U) onto current mask; U-update: dual step; W-update: linear solve
            Z = (W + U) * W_mask
            U = U + (W - Z)
            W = Hinv.matmul(G + rho * (Z - U))

        # Final Z on mask and denormalize (W is stored as W.T in normalized space)
        Z = (W + U) * W_mask
        return (Z.T / norm).reshape(final_shape).to(final_dtype), W_mask
