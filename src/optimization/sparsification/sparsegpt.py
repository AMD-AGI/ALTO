import math
import torch
import torch.nn as nn
from loguru import logger

import transformers

from src.utils import ALGO_REGISTRY
from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class SparseGPT(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, config):
        super().__init__(model, sparsity_config, input, config)
        self.percdamp = sparsity_config['special']['percdamp']
        self.blocksize = sparsity_config['special']['blocksize']
        self.dev = torch.device('cuda')
        self.layers_cache = {}

    @torch.no_grad()
    def add_batch(self, layer, name, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.layers_cache[name]['H'] *= self.layers_cache[name]['nsamples'] / (
            self.layers_cache[name]['nsamples'] + tmp
        )
        self.layers_cache[name]['nsamples'] += tmp
        inp = math.sqrt(2 / self.layers_cache[name]['nsamples']) * inp.float().to(self.dev)
        self.layers_cache[name]['H'] += inp.matmul(inp.t())

    @torch.no_grad()
    def layer_init(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.layers_cache[name]['H'] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev, dtype=torch.float
        )
        self.layers_cache[name]['nsamples'] = 0
        self.layers_cache[name]['columns'] = W.shape[1]
        self.layers_cache[name]['rows'] = W.shape[0]

    @torch.no_grad()
    def block_init(self, block):
        self.layers_cache = {}
        self.named_layers = self.model.get_block_linears(block)
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs
    ):
        for name, layer in layers_dict.items():
            for batch_idx in range(len(input_feat[input_name])):
                self.add_batch(layer, name, input_feat[input_name][batch_idx])
            W = layer.weight.data.clone()
            if isinstance(layer, transformers.Conv1D):
                W = W.t()
            W = W.float()
            H = self.layers_cache[name]['H']
            del self.layers_cache[name]['H']

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.layers_cache[name]['columns'], device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None
            Losses = torch.zeros(self.layers_cache[name]['rows'], device=self.dev)
            for i1 in range(0, self.layers_cache[name]['columns'], self.blocksize):
                i2 = min(i1 + self.blocksize, self.layers_cache[name]['columns'])
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if self.sparser.prunen == 0: 
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * self.sparser.sparsity)]
                        mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.sparser.prunen != 0 and i % self.sparser.prunem == 0:
                        tmp = W1[:, i:(i + self.sparser.prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + self.sparser.prunem)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, self.sparser.prunen, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, i]] = 0

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()
            del self.layers_cache[name]
            logger.info(f'Pruning gap: {torch.sum(Losses).item()}')

            layer.weight.data = W.reshape(layer.weight.shape).to(layer.weight.data.dtype)