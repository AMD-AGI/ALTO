import gc
from loguru import logger

import numpy as np
from tqdm import tqdm 

import torch
import torch.nn.functional as F

from src.utils import ALGO_REGISTRY, to_device
from .blockwise_pruning import BlockwisePruning


@ALGO_REGISTRY
class CosineSimilarity(BlockwisePruning):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)
        self.optimization_method_name = 'CosineSimilarity'
        self.applicability_error_message = 'Cosine Similarity is only suitable for (sub-)layer pruning).'
        assert self.prune_embedding == False, self.applicability_error_message
        assert self.prune_attn == False, self.applicability_error_message
        assert self.prune_mlp == False, self.applicability_error_message
        self.total_layers = len(self.model.model.model.layers)
        self.activations = []
        self.activations.append(to_device(torch.cat(input['data'], dim=0), 'cpu'))
        self.bi_matrix = np.zeros((self.total_layers + 1, self.total_layers + 1))
    
    @torch.no_grad()
    def optimize_block(self, block):
        to_device(block, torch.device('cuda'))
        self.input['data'] = self.block_forward(block)
        self.activations.append(to_device(torch.cat(self.input['data'], dim=0), 'cpu'))
        torch.cuda.empty_cache()
        block = block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_bi(self, feature1: torch.Tensor, feature2: torch.Tensor):
        cosine_sim = to_device(F.cosine_similarity(
            to_device(feature1, torch.device('cuda')), 
            to_device(feature2, torch.device('cuda')), 
            dim=-1
        ), 'cpu')
        return 1 - torch.mean(cosine_sim).item()
    
    @torch.no_grad()
    def compute_bi_matrix(self):
        for i in tqdm(range(self.total_layers + 1), desc="Computing BI score matrix..."):
            for j in range(i + 1, self.total_layers + 1):
                activation_i = self.activations[i]
                activation_j = self.activations[j]
                self.bi_matrix[i, j] = self.compute_bi(activation_i, activation_j)

    @torch.no_grad()
    def find_toprune_layers(self, num_layers_to_prune):
        pruned_indices = set()
        result = []
        while len(result) < num_layers_to_prune:
            min_value = float('inf')
            best_prune = []
            for layers_to_prune in range(1, num_layers_to_prune - len(result) + 1):
                for i in range(self.total_layers - layers_to_prune + 1):
                    if all(idx not in pruned_indices for idx in range(i, i + layers_to_prune)):
                        importance_value = self.bi_matrix[i][i + layers_to_prune]
                        if importance_value < min_value:
                            min_value = importance_value
                            best_prune = list(range(i, i + layers_to_prune))
            result.extend(best_prune)
            pruned_indices.update(best_prune)
        return sorted(result) 

    @torch.no_grad()
    def after_optimize_blocks(self):
        num_layers_to_prune = int(self.sparsity * self.total_layers)
        self.compute_bi_matrix()
        layers_to_prune = self.find_toprune_layers(num_layers_to_prune)
        self.W_mask = {f'prune_layer={self.prune_layer},prune_sublayer={self.prune_sublayer}': torch.tensor(layers_to_prune)}
        first_layer_to_prune = min(layers_to_prune)
        total_layers = len(self.bi_matrix) - 1
        logger.info(f"Layers to prune (indices): {layers_to_prune}")
        for prune_idx in sorted(layers_to_prune, reverse=True):
            if prune_idx == 0:
                logger.warning(f"Layer {prune_idx} is the first layer and cannot be pruned.")
                continue 
            del self.model.model.model.layers[prune_idx]
        for layer_idx in range(first_layer_to_prune, total_layers-num_layers_to_prune):
            self.model.model.model.layers[layer_idx].self_attn.layer_idx = layer_idx  
        self.model.model.config.num_hidden_layers -= num_layers_to_prune
        torch.cuda.empty_cache()