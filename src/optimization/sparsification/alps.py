import math
import torch
import torch.nn as nn
from loguru import logger

import transformers

from src.utils import ALGO_REGISTRY
from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class ALPS(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)
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
        nsamples += minibatch_size
        inp = inp.float().to(H.device)
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
            for i1 in range(0, columns, self.blocksize):
                i2 = min(i1 + self.blocksize, columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if self.N == -1: 
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.N != -1 and i % self.M == 0:
                        tmp = W1[:, i:(i + self.M)] ** 2 / (torch.diag(Hinv1)[i:(i + self.M)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, self.N, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, i]] = 0
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W_mask[:, i1:i2] = mask1
                W[:, i1:i2] = Q1
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
                Losses += torch.sum(Losses1, 1) / 2

            torch.cuda.synchronize()
            logger.info(f'Pruning gap: {torch.sum(Losses).item()}')
            layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype)
            self.W_mask[global_layer_name] = W_mask