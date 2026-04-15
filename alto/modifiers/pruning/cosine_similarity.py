# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/sparsification/sparsegpt/base.py


import re

import torch
from torch.nn import Module
import torch.nn.functional as F
from pydantic import PrivateAttr
from torchtitan.tools.logging import logger

from alto.modifiers.pruning.base import PruningModifierBase
from alto.observers.input_only import InputOnlyObserver

__all__ = ["CosineSimilarityModifier"]


class CosineSimilarityModifier(PruningModifierBase):
    """
    Modifier for layer pruning using cosine similarity (Block Influence metric).
    Paper: https://arxiv.org/abs/2301.00774

    Uses cosine distance between hidden states at layer boundaries to measure
    each layer's importance. Layers (or consecutive blocks) with the smallest
    cosine distance (i.e. least influence on representations) are pruned.

    Sample yaml:

    ```yaml
    pruning_modifiers:
        CosineSimilarityModifier:
            sparsity: 0.25
            pruning_dimension: "layer"
            targets: ["re:.*\\.wq$"]
            ignore: []
    ```

    Lifecycle:

    - on_initialize
        - register InputOnlyObserver hooks on target layers (e.g. wq per block)
    - pre_step / forward / post_step
        - observers collect input activations
        - compress_modules computes BI matrix and removes layers
    - on_finalize
        - cleanup hooks and observers

    :param sparsity: Fraction of layers to prune (e.g. 0.25 = remove 25% of layers)
    :param pruning_dimension: Must be "layer" (sublayer not yet supported)
    :param targets: layer name patterns for observer placement. Use "re:.*\\.wq$"
        to capture the input hidden state at each TransformerBlock boundary.
    :param ignore: patterns to exclude
    """

    # private variables
    _model_parts: list | None = PrivateAttr(default=None)
    _layers_to_prune: list = PrivateAttr(default_factory=list)
    _pruning_complete: bool = PrivateAttr(default=False)

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._observer_name = "input_only"
        self._model_parts = model_parts
        assert self.pruning_dimension in ["layer"], \
            "CosineSimilarityModifier currently only supports 'layer' pruning."
        return super().on_initialize(model_parts, **kwargs)

    def compress_modules(self):
        """
        Compute Block Influence (BI) matrix from collected activations,
        find optimal layers to prune, and remove them from the model.

        This is called from post_step after each calibration forward pass.
        Pruning is performed only once; subsequent calls are no-ops.
        """
        if self._pruning_complete:
            return

        # Organize observers by layer index
        layer_activations = {}
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            num_samples = observer.num_samples
            logger.info(f"Collected {num_samples.item()} samples for {name}")
            assert isinstance(observer, InputOnlyObserver), \
                "CosineSimilarityModifier requires input_only observer"

            # Extract layer index from name like "layers.0.attention.wq"
            idx_match = re.search(r'\.(\d+)\.', name)
            if idx_match:
                layer_idx = int(idx_match.group(1))
                layer_activations[layer_idx] = observer.stats

        if not layer_activations:
            logger.warning("No layer activations collected, skipping pruning.")
            return

        total_layers = self._model_args.n_layers
        num_layers_to_prune = int(self.sparsity * total_layers)
        if num_layers_to_prune <= 0:
            logger.info("No layers to prune (sparsity too low).")
            self._pruning_complete = True
            return

        # Compute BI matrix
        bi_matrix = self._compute_bi_matrix(layer_activations, total_layers)

        # Find layers to prune using greedy search
        self._layers_to_prune = self._find_toprune_layers(
            bi_matrix, total_layers, num_layers_to_prune
        )
        logger.info(f"Layers to prune (indices): {self._layers_to_prune}")

        # Remove layers from each model part
        for model in self._model_parts:
            self._remove_layers(model, self._layers_to_prune)

        self._pruning_complete = True

    @torch.no_grad()
    def _compute_bi(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """
        Compute Block Influence (BI) between two hidden state tensors.
        BI = 1 - mean(cosine_similarity(feature1, feature2))

        :param feature1: hidden states [N, seq_len, hidden_dim]
        :param feature2: hidden states [N, seq_len, hidden_dim]
        :return: scalar BI value (0 = identical, higher = more different)
        """
        cosine_sim = F.cosine_similarity(feature1, feature2, dim=-1).cpu()
        return 1 - torch.mean(cosine_sim).item()

    @torch.no_grad()
    def _compute_bi_matrix(
        self,
        layer_activations: dict[int, torch.Tensor],
        total_layers: int,
    ) -> list[list[float]]:
        """
        Compute the Block Influence matrix.

        bi_matrix[i][j] measures the influence of removing layers i through j-1,
        computed as cosine distance between hidden states at positions i and j.

        :param layer_activations: dict mapping layer_idx -> activation tensor
        :param total_layers: total number of layers in the model
        :return: (total_layers+1) x (total_layers+1) BI matrix
        """
        bi_matrix = [[0.0] * (total_layers + 1) for _ in range(total_layers + 1)]
        sorted_indices = sorted(layer_activations.keys())

        for i_pos, i in enumerate(sorted_indices):
            for j_pos in range(i_pos + 1, len(sorted_indices)):
                j = sorted_indices[j_pos]
                bi_matrix[i][j] = self._compute_bi(
                    layer_activations[i], layer_activations[j]
                )
                logger.info(f"BI[{i}][{j}] = {bi_matrix[i][j]:.6f}")

        return bi_matrix

    @torch.no_grad()
    def _find_toprune_layers(
        self,
        bi_matrix: list[list[float]],
        total_layers: int,
        num_layers_to_prune: int,
    ) -> list[int]:
        """
        Greedily find layers to prune by selecting consecutive blocks with
        minimum Block Influence.

        At each iteration, searches for the consecutive block of unpruned
        layers whose removal has the least impact on representations (smallest
        BI value), and marks them for pruning.

        Layer 0 (first) and layer N-1 (last) are never pruned:
        - Layer 0: critical for initial representation
        - Layer N-1: bi_matrix[N-1][N] is undefined (no observation beyond
          the last layer), and the final layer is typically important for
          output quality

        For BI[i][i+k], the layers being evaluated for removal are i..i+k-1.
        E.g. BI[30][31] small means layer 30's input ≈ layer 31's input,
        so layer 30 is the one to prune.

        :param bi_matrix: Block Influence matrix
        :param total_layers: total number of layers
        :param num_layers_to_prune: number of layers to remove
        :return: sorted list of layer indices to prune
        """
        pruned_indices = set()
        result = []

        while len(result) < num_layers_to_prune:
            min_value = float('inf')
            best_prune = []

            for block_size in range(1, num_layers_to_prune - len(result) + 1):
                # i starts from 1 (protect layer 0)
                # i + block_size < total_layers (protect last layer, and
                # ensure bi_matrix[i][i+block_size] was actually computed)
                for i in range(1, total_layers - block_size):
                    candidate = list(range(i, i + block_size))
                    if all(idx not in pruned_indices for idx in candidate):
                        importance_value = bi_matrix[i][i + block_size]
                        if importance_value < min_value:
                            min_value = importance_value
                            best_prune = candidate

            if not best_prune:
                logger.warning(
                    f"Could not find more layers to prune, "
                    f"stopping at {len(result)} of {num_layers_to_prune}"
                )
                break

            result.extend(best_prune)
            pruned_indices.update(best_prune)

        return sorted(result)

    def _remove_layers(self, model: Module, layers_to_prune: list[int]):
        """
        Remove specified layers from a torchtitan Transformer model.

        Deletes the target TransformerBlocks from model.layers (ModuleDict),
        then re-indexes remaining layers to maintain contiguous string keys
        ("0", "1", ...) and updates model.n_layers / model.model_args.n_layers.

        :param model: Transformer model with model.layers (ModuleDict)
        :param layers_to_prune: list of layer indices to remove
        """
        if not hasattr(model, 'layers'):
            logger.warning(
                f"Model {type(model).__name__} has no 'layers' attribute, "
                f"skipping removal."
            )
            return

        last_layer_idx = max(int(k) for k in model.layers.keys())

        # Remove layers in reverse order to avoid index shifting issues
        for idx in sorted(layers_to_prune, reverse=True):
            key = str(idx)
            if idx == 0:
                logger.warning("Layer 0 is the first layer and cannot be pruned, skipping.")
                continue
            if idx == last_layer_idx:
                logger.warning(f"Layer {idx} is the last layer and cannot be pruned, skipping.")
                continue
            if key in model.layers:
                del model.layers[key]
                logger.info(f"Removed layer {key}")
            else:
                logger.warning(f"Layer {key} not found in model, skipping.")

        # Re-index remaining layers to maintain contiguous string keys
        remaining_keys = sorted(model.layers.keys(), key=int)
        new_layers = torch.nn.ModuleDict()
        for new_idx, old_key in enumerate(remaining_keys):
            new_layers[str(new_idx)] = model.layers[old_key]
        model.layers = new_layers

        # Update model config
        new_n_layers = len(new_layers)
        if hasattr(model, "config"):
            model.config.n_layers = new_n_layers
        if hasattr(model, "n_layers"):
            model.n_layers = new_n_layers
        logger.info(
            f"Model now has {new_n_layers} layers "
            f"(pruned {len(layers_to_prune)} layers)"
        )

        torch.cuda.empty_cache()
