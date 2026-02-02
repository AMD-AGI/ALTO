# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/base.py
# licensed under the Apache License 2.0

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers.sparsification.base import SparsityModifierBase
from modeloptimizer.observers.hessian import PRECISION, HessianObserver
from modeloptimizer.utils.pytorch.module import TransformerConv1D

__all__ = ["AdmmModifier"]


class AdmmModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot ADMM algorithm to a model

    Sample yaml:

    ```yaml
    test_stage:
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
    :param targets: list of layer names to compress during SparseGPT, or '__ALL__'
        to compress every layer in the model. Alias for `sequential_targets`
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    dampening_frac: float | None = 0.01
    iterative_prune_granularity: int = 32
    iterations: int = 32

    def on_initialize(self, model: Module, **kwargs) -> bool:
        self._observer_name = "hessian"
        return super().on_initialize(model, **kwargs)

    def compress_modules(self):
        """
        Sparsify modules which have been calibrated
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples

            logger.info(f"Sparsifying {name} using {num_samples.item()} samples")
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

        norm = torch.diag(H).sqrt() + 1e-8
        H = H / norm
        H = (H.T / norm).T
        W = (W * norm).T
        rho0 = self.dampening_frac * torch.diag(H).mean()
        diag = torch.arange(num_columns, device=H.device)
        H[diag,diag] += rho0

        rho = 1
        G = H.matmul(W) 
        H[diag,diag] += rho
        Hinv = torch.inverse(H)
        U = torch.zeros_like(W)

        for iter in range(self.iterations):
            if prune_n != -1:
                sparsity = 0.5
                cur_sparsity = sparsity - sparsity * (1 - (iter + 1) / self.iterative_prune_granularity) ** 3
                WT = (W + U).T.reshape((W.shape[1] * W.shape[0] // prune_m, prune_m)).abs()
                W_mask = torch.zeros(WT.shape, dtype=torch.bool, device=W.device)
                sort_inds = WT.sort(dim=1)[1]
                for i in range(prune_m - prune_n, prune_m):
                    W_mask[torch.arange(WT.shape[0]), sort_inds[:,i]] = 1
                W_mask = W_mask.reshape(W.T.shape).T
            else:
                cur_sparsity = sparsity - sparsity * (1 - (iter + 1) / self.iterative_prune_granularity) ** 3
                topk = torch.topk((W + U).abs().flatten(), k=int(W.numel() * sparsity), largest=False)
                W_mask = torch.ones(W.numel(), dtype=torch.bool, device=W.device)
                W_mask[topk.indices] = 0
                W_mask = W_mask.reshape(W.shape)
                del topk
            
            local_Z = (W + U).abs()
            local_Z[W_mask] = torch.inf
            thres = local_Z.flatten().kthvalue(int(W.numel() * cur_sparsity) + 1)[0]
            W_mask = (local_Z >= thres)
            del thres

            Z = (W + U) * W_mask
            U = U + (W - Z)
            W = Hinv.matmul(G + rho * (Z - U))

        Z = (W + U) * W_mask
        return (Z.T / norm).reshape(final_shape).to(final_dtype), W_mask