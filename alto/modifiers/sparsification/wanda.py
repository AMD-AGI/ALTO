# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/wanda/base.py
# licensed under the Apache License 2.0

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers.sparsification.base import SparsityModifierBase
from alto.observers.per_channel_norm import PRECISION, PerChannelNormObserver
from alto.utils.pytorch.module import TransformerConv1D

__all__ = ["WandaModifier"]


class WandaModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

    Sample yaml:

    ```yaml
    sparsity_modifiers:
        WandaModifier:
            sparsity: 0.5
            mask_structure: "2:4"
    ```

    Lifecycle:

    - on_initialize
        - register_hook(module, calibrate_module, "forward")
        - run_sequential / run_basic
            - make_empty_row_scalars
            - accumulate_row_scalars
    - on_sequential_batch_end
        - sparsify_weight
    - on_finalize
        - remove_hooks()

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param targets: list of layer names to compress during Wanda, or '__ALL__'
        to compress every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        sparsify even if they match a target. Defaults to empty list.
    """

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._observer_name = "per_channel_norm"
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
            assert isinstance(observer, PerChannelNormObserver), \
                "WandaModifier requires per_channel_norm observer"
            sparsified_weight, _ = self._sparsify_weight(
                module=module,
                row_scalar=observer.stats,
                sparsity=sparsity,
                prune_n=self._prune_n,
                prune_m=self._prune_m,
            )
            module.weight.data.copy_(sparsified_weight)

    def _sparsify_weight(
        self,
        module: Module,
        row_scalar: torch.Tensor,
        sparsity: float,
        prune_n: int,
        prune_m: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run pruning on the layer up to the target sparsity value.

        :param module: module to sparsify
        :param row_scalar: row scalar to use for sparsification
        :param sparsity: target sparsity to reach for layer
        :param prune_n: N for N:M pruning
        :param prune_m: M for N:M pruning
        :return: sparsified weight
        """

        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.data.clone()
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)

        if TransformerConv1D and isinstance(module, TransformerConv1D):
            W = W.t()

        W = W.to(dtype=PRECISION)
        S = row_scalar

        W_metric = torch.abs(W) * torch.sqrt(S.reshape((1, -1)))

        # initialize a mask to be all False
        W_mask = torch.zeros_like(W_metric) == 1
        if prune_n != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                        True,
                    )
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)

        W[W_mask] = 0.0  # set weights to zero

        if TransformerConv1D and isinstance(module, TransformerConv1D):
            W = W.t()

        W = W.reshape(final_shape).to(final_dtype)

        return W, W_mask.reshape(final_shape)
