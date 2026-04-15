# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/base.py

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers.sparsification.base import SparsityModifierBase
from alto.observers.hessian import PRECISION, HessianSparseGPTObserver
from alto.utils.pytorch.module import TransformerConv1D

__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot SparseGPT algorithm to a model
    from the paper: https://arxiv.org/abs/2301.00774

    Sample yaml:

    ```yaml
    sparsity_modifiers:
        SparseGPTModifier:
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

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Whether or not to preserve the sparsity mask
        during when applying sparsegpt, this becomes useful when starting from a
        previously pruned model, defaults to False.
    :param targets: list of layer names to compress during SparseGPT, or '__ALL__'
        to compress every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        sparsify even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    block_size: int = 128
    dampening_frac: float | None = 0.01
    preserve_sparsity_mask: bool = False

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._observer_name = "hessian_sparsegpt"
        return super().on_initialize(model_parts, **kwargs)

    def compress_modules(self):
        """
        Sparsify modules which have been calibrated
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples

            logger.info(f"Sparsifying {name} using {num_samples.item()} samples")
            assert isinstance(observer, HessianSparseGPTObserver), \
                "SparseGPTModifier requires hessian_sparsegpt observer"
            loss, sparsified_weight = self._sparsify_weight(
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
        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.clone()
        H = hessian
        block_size = self.block_size
        dampening_frac = self.dampening_frac
        preserve_sparsity_mask = self.preserve_sparsity_mask

        # standardize shape and dtype
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)
        elif TransformerConv1D and isinstance(module, TransformerConv1D):
            W.transpose_(0, 1)
        W = W.to(dtype=PRECISION)
        num_rows = W.shape[0]
        num_columns = W.shape[1]

        # mask dead hessian values
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # compute inverse hessian in place to save memory
        try:
            damp = dampening_frac * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[0], device=H.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except torch._C._LinAlgError:
            logger.warning("Failed to invert hessian due to numerical instability. Consider "
                           "increasing SparseGPTModifier.dampening_frac, increasing the number "
                           "of calibration samples, or shuffling the calibration dataset")
            Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

        # sparsity mask
        # TODO: consider computing sparsity mask in the same way and place as gptq
        mask = None
        if preserve_sparsity_mask:
            # compute existing sparsity mask
            mask = torch.where(
                W == 0,
                torch.tensor(1, dtype=torch.bool),
                torch.tensor(0, dtype=torch.bool),
            )
            current_sparsity = mask.sum() / W.numel()
            if current_sparsity > sparsity:
                raise ValueError("The target sparsity is lower than the sparsity "
                                 "of the base model. Please retry "
                                 "after turning preserve_sparsity_mask=False")

        losses = torch.zeros(num_rows, device=module.weight.device)

        # See section 3.4 of https://arxiv.org/abs/2203.07259
        for i1 in range(0, num_columns, block_size):
            i2 = min(i1 + block_size, num_columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                    if int(W1.numel() * sparsity) > mask1.sum():
                        # target sparsity is higher than base sparsity, extend mask1
                        tmp = ((~mask[:, i1:i2]) * W1**2 / (torch.diag(Hinv1).reshape((1, -1)))**2)
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1)))**2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = (W1[:, i:(i + prune_m)]**2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1)))**2)
                    if mask is not None:
                        tmp = tmp * (~mask[:, i:(i + prune_m)])

                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            losses += torch.sum(Losses1, 1) / 2

            if preserve_sparsity_mask:
                # respect the sparsity of other groups
                # really not needed, but kept for explicitness
                W[:, i2:] -= (~mask[:, i2:]) * Err1.matmul(Hinv[i1:i2, i2:])
            else:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if TransformerConv1D and isinstance(module, TransformerConv1D):
            W.transpose_(0, 1)
        W = W.reshape(final_shape).to(final_dtype)

        loss = torch.sum(losses).item()
        return loss, W
