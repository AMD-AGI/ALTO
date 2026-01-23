import math
import torch
import torch.nn as nn
from loguru import logger

import transformers

from src.utils import ALGO_REGISTRY
from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class SparseGPT(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)
        self.optimization_method_name = 'SparseGPT'
        self.applicability_message = 'SparseGPT is only suitable for unstructured and N:M sparsity pattern.'
        assert self.block_sparsity_config == False, self.applicability_message
        self.percdamp = sparsity_config['method_kwargs']['percdamp']
        self.blocksize = sparsity_config['method_kwargs']['blocksize']

    @torch.no_grad()
    def add_batch(self, layer, inp, H, nsamples):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        minibatch_size = inp.shape[0]
        if isinstance(layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        H *= nsamples / (nsamples + minibatch_size)
        nsamples += minibatch_size
        inp = math.sqrt(2 / nsamples) * inp.float().to(H.device)
        H += inp.matmul(inp.t())
        return H, nsamples

    @torch.no_grad()
    def optimize_subset(
        self,
        layers_dict,
        input_feat,
        output_feat,
        prev_op,
        input_name,
        inspect_module,
        block_idx,
        subset_kwargs,
    ):
        for name, layer in layers_dict.items():

            global_layer_name = f'layers.{block_idx}.{name}'
            if self.sparsity_dict is not None:
                sparsity = self.sparsity_dict[global_layer_name]
            elif isinstance(self.sparsity, list):
                sparsity = self.sparsity[block_idx]
            else:
                sparsity = self.sparsity
            logger.info(f"Sparsity of {name} is {sparsity}.")

            W = layer.weight.data.clone()
            device = W.device
            if isinstance(layer, transformers.Conv1D):
                W = W.t()
            rows, columns = W.shape[0], W.shape[1]
            W = W.float()
            H = torch.zeros((columns, columns), device=device, dtype=torch.float)
            nsamples = 0
            for batch_idx in range(len(input_feat[input_name])):
                H, nsamples = self.add_batch(layer, input_feat[input_name][batch_idx], H, nsamples)

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(columns, device=device)
            H[diag, diag] += damp
            try:
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
            except Exception:
                H_cpu = H.cpu()
                H_cpu = torch.linalg.cholesky(H_cpu)
                H_cpu = torch.cholesky_inverse(H_cpu)
                H_cpu = torch.linalg.cholesky(H_cpu, upper=True)
                H = H_cpu.to(device)
            Hinv = H

            W_mask = torch.zeros_like(W, dtype=torch.bool)
            Losses = torch.zeros(rows, device=device)
            for left in range(0, columns, self.blocksize):
                right = min(left + self.blocksize, columns)
                count = right - left

                W_local = W[:, left:right].clone()
                Pruned_local = torch.zeros_like(W_local)
                Error_local = torch.zeros_like(W_local)
                Losses_local = torch.zeros_like(W_local)
                Hinv_local = Hinv[left:right, left:right]

                if self.N == -1: 
                    Metric_local = W_local ** 2 / (torch.diag(Hinv_local).reshape((1, -1))) ** 2
                    thresh = torch.sort(Metric_local.flatten())[0][int(Metric_local.numel() * sparsity)]
                    Mask_local = Metric_local <= thresh
                else:
                    Mask_local = torch.zeros_like(W_local) == 1

                for i in range(count):
                    w_local_vector = W_local[:, i]
                    hinv_local_vector = Hinv_local[i, i]

                    if self.N != -1 and i % self.M == 0:
                        Metric_local = W_local[:, i:(i + self.M)] ** 2 / (torch.diag(Hinv_local)[i:(i + self.M)].reshape((1, -1))) ** 2
                        Mask_local.scatter_(1, i + torch.topk(Metric_local, self.N, dim=1, largest=False)[1], True)

                    pruned_local_vector = w_local_vector.clone()
                    pruned_local_vector[Mask_local[:, i]] = 0
                    Pruned_local[:, i] = pruned_local_vector
                    Losses_local[:, i] = (w_local_vector - pruned_local_vector) ** 2 / hinv_local_vector ** 2
                    error_local_vector = (w_local_vector - pruned_local_vector) / hinv_local_vector
                    W_local[:, i:] -= error_local_vector.unsqueeze(1).matmul(Hinv_local[i, i:].unsqueeze(0))
                    Error_local[:, i] = error_local_vector

                W_mask[:, left:right] = Mask_local
                W[:, left:right] = Pruned_local
                W[:, right:] -= Error_local.matmul(Hinv[left:right, right:])
                Losses += torch.sum(Losses_local, 1) / 2

            torch.cuda.synchronize()
            logger.info(f'Pruning gap: {torch.sum(Losses).item()}')
            layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype)
            self.W_mask[global_layer_name] = W_mask