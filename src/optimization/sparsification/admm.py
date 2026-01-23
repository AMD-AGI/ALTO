import torch
from loguru import logger

import transformers

from src.utils import ALGO_REGISTRY
from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class Admm(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)
        self.optimization_method_name = 'Admm'
        self.applicability_message = 'Admm is only suitable for unstructured and N:M sparsity pattern.'
        assert self.block_sparsity_config == False, self.applicability_message
        self.percdamp = sparsity_config['method_kwargs']['percdamp']
        self.iterative_prune_granularity = sparsity_config['method_kwargs']['iterative_prune_granularity']
        self.iterations = sparsity_config['method_kwargs']['iterations']

    @torch.no_grad()
    def add_batch(self, inp, H):
        inp = inp.reshape(-1, inp.shape[-1]).float().to(H.device)
        H += inp.T.matmul(inp)
        return H

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
            for batch_idx in range(len(input_feat[input_name])):
                H = self.add_batch(input_feat[input_name][batch_idx], H)

            norm = torch.diag(H).sqrt() + 1e-8
            H = H / norm
            H = (H.T / norm).T
            W = (W.float() * norm).T
            rho0 = self.percdamp * torch.diag(H).mean()
            diag = torch.arange(columns, device=device)
            H[diag,diag] += rho0

            rho = 1
            G = H.matmul(W) 
            H[diag,diag] += rho
            Hinv = torch.inverse(H)
            U = torch.zeros_like(W)

            for iter in range(self.iterations):
                if self.N != -1:
                    sparsity = 0.5
                    cur_sparsity = sparsity - sparsity * (1 - (iter + 1) / self.iterative_prune_granularity) ** 3
                    WT = (W + U).T.reshape((W.shape[1] * W.shape[0] // self.M, self.M)).abs()
                    W_mask = torch.zeros(WT.shape, dtype=torch.bool, device=W.device)
                    sort_inds = WT.sort(dim=1)[1]
                    for i in range(self.M - self.N, self.M):
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
            layer.weight.data = (Z.T / norm).reshape(layer.weight.shape).to(layer.weight.data.dtype)
            self.W_mask[global_layer_name] = W_mask