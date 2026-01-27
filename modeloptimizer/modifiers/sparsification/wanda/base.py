# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/wanda/base.py
# licensed under the Apache License 2.0
# modifications:
# - extract layer_index from layer_name (in case PP are used)
# - isolate the calibration from sparsification into a separate observer
# - unifies the observers used by Wanda and OWL

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers.sparsification.base import SparsityModifierBase
from modeloptimizer.observers.per_channel_norm import WANDA_PRECISION

__all__ = ["WandaPruningModifier"]


class WandaPruningModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

    Sample yaml:

    ```yaml
    test_stage:
      sparsity_modifiers:
        WandaPruningModifier:
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
    :param sparsity_profile: Can be set to 'owl' to use Outlier Weighed
        Layerwise Sparsity (OWL), more information can be found
        in the paper https://arxiv.org/pdf/2310.05175
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param owl_m: Number of outliers to use for OWL
    :param owl_lmbda: Lambda value to use for OWL
    :param sequential_targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model. Alias for `targets`
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model. Alias for `sequential_targets`
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target. Defaults to empty list.
    """

    def on_initialize(self, model: Module, **kwargs) -> bool:
        self._observer_name = "per_channel_norm"
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
            sparsified_weight = self._sparsify_weight(
                module=module,
                row_scalar=observer.norm,
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
    ) -> torch.Tensor:
        """
        Run pruning on the layer up to the target sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        """

        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.data.clone()
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)

        # TODO: add support for Conv1D
        # if isinstance(module, transformers.Conv1D):
        #     W = W.t()

        W = W.to(dtype=WANDA_PRECISION)
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

        # TODO: add support for Conv1D
        # if isinstance(module, transformers.Conv1D):
        #     W = W.t()

        W = W.reshape(final_shape).to(final_dtype)

        return W
