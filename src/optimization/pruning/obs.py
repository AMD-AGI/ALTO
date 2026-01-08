from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ALGO_REGISTRY
from .blockwise_pruning import BlockwisePruning


@ALGO_REGISTRY
class OBS(BlockwisePruning):
    def __init__(self, model, pruning_config, global_config, input):
        super().__init__(model, pruning_config, global_config, input)
        self.optimization_method_name = 'OBS'
        self.applicability_error_message = 'OBS is only suitable for structured pruning of attn heads and mlp neurons.'
        assert self.prune_embedding == False, self.applicability_error_message
        assert self.prune_layer == False, self.applicability_error_message
        assert self.prune_sublayer == False, self.applicability_error_message

    @torch.no_grad()
    def compute_hessian(self, inp, H, nsamples):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_samples = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        inp = inp.float().to(H.device)
        nsamples += batch_samples
        H += ((inp).matmul(inp.t()))
        return H, nsamples

    @torch.no_grad()
    def optimize_subset(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        block_idx,
        subset_kwargs,
    ):
        for layer_name, layer in layers_dict.items():
            global_layer_name = f'layers.{block_idx}.{layer_name}'
            out_name  = self.model.get_out_projection_name()
            down_name = self.model.get_down_projection_name()
            allowed = (
                (layer_name == out_name  and self.prune_attn) or
                (layer_name == down_name and self.prune_mlp)
            )

            if allowed:
                if self.sparsity_dict is not None:
                    sparsity = self.sparsity_dict[global_layer_name]
                elif isinstance(self.sparsity, list):
                    sparsity = self.sparsity[block_idx]
                else:
                    sparsity = self.sparsity
                logger.info(f"Sparsity of {layer_name} is {sparsity}.")

                device, dtype = layer.weight.device, layer.weight.dtype
                columns = layer.weight.data.shape[1]
                H = torch.zeros((columns, columns), device=device)
                nsamples = 0
                for batch_idx in range(len(input_feat[input_name])):
                    H, nsamples = self.compute_hessian(input_feat[input_name][batch_idx], H, nsamples)
                W = layer.weight.data.clone().t().float()
                dead = torch.diag(H) == 0
                W[dead,:] = 0 
                H += torch.eye(W.shape[0]).to(device) * torch.mean(torch.diag(H)) * 1e-2
                G = H @ W

                if layer_name == self.model.get_out_projection_name():
                    if self.prune_attn == False:
                        continue
                    num_attention_heads = getattr(self.model.model.config, "num_attention_heads", None)
                    num_key_value_heads = getattr(self.model.model.config, "num_key_value_heads", None)
                    group = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
                    W_t, error, pruned_mask = self.local_prune_core(W, H, G, group, int(group * (1-sparsity)), 1)
                else:
                    if self.prune_mlp == False:
                        continue
                    W_t, error, pruned_mask = self.local_prune_core(W, H, G, W.shape[0], int(W.shape[0] * (1-sparsity)), 32)
                
                layer.weight.data = W_t.t().to(layer.weight.dtype)
                logger.info(f"prune {layer_name}; error is: {error}.")
            else:
                continue
            
            self.W_mask[global_layer_name] = pruned_mask

    @torch.no_grad()
    def local_prune_core(self, W, H, G, num_total_groups: int, num_groups_to_remain: int, update_iter: int = 32):
        device = W.device
        cin, cout = W.shape
        group_size = int(cin / num_total_groups)
        H_inv = torch.linalg.inv(H)
        W_g = W.reshape(num_total_groups, group_size, cout)
        group_abs_sum = torch.sum(torch.abs(W_g), dim=(1, 2))
        pruned_group_mask = (group_abs_sum <= 1e-12)
        num_already_zero = int(pruned_group_mask.sum().item())
        if num_already_zero > 0:
            zero_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in torch.nonzero(pruned_group_mask, as_tuple=False).flatten()
            ])
            H_inv[zero_idx, :] = 0
            H_inv[:, zero_idx] = 0
            W = H_inv @ G
            if (num_total_groups - num_groups_to_remain - num_already_zero) <= 0:
                W[zero_idx, :] = 0
                return W, 0.0
        remaining_to_prune = int(num_total_groups - num_groups_to_remain - int(pruned_group_mask.sum().item()))
        if remaining_to_prune <= 0:
            kept_mask = (~torch.repeat_interleave(pruned_group_mask, group_size))
            W_kept = torch.zeros_like(W)
            try:
                W_kept[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
            except Exception:
                H_cpu = H[kept_mask][:, kept_mask].cpu()
                G_cpu = G[kept_mask, :].cpu()
                W_kept[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)
            prune_loss = torch.sum(-W_kept * G + 0.5 * W_kept * (H @ W_kept)).detach().item()
            return W_kept, prune_loss
        update_rounds = max(int(min(update_iter, remaining_to_prune)), 1)
        base, extra = divmod(remaining_to_prune, update_rounds)
        groups_to_prune_each_round = torch.full((update_rounds,), base, dtype=torch.int, device=device)
        if extra > 0:
            groups_to_prune_each_round[:extra] += 1
        for round_id in range(update_rounds):
            if group_size > 1:
                obj_mat = torch.zeros_like(W)
                for g in range(num_total_groups):
                    if pruned_group_mask[g]:
                        continue
                    sl = slice(g * group_size, (g + 1) * group_size)
                    H_block = torch.linalg.inv(H_inv[sl, sl]) 
                    obj_mat[sl, :] = (H_block @ W[sl, :] / 2.0)
            else:
                diag_Hinv = torch.diag(H_inv)
                safe_den = (pruned_group_mask.to(W.dtype) + diag_Hinv).clamp_min(1e-12)
                obj_mat = (1.0 / safe_den)[:, None] * (W / 2.0)
            obj_val = (W * obj_mat).reshape(num_total_groups, group_size, cout).sum(dim=(1, 2))
            obj_val_masked = obj_val + 1e20 * pruned_group_mask.to(obj_val.dtype)
            sorted_groups = torch.argsort(obj_val_masked)
            k = int(groups_to_prune_each_round[round_id].item())
            pick_groups = sorted_groups[:k]
            pick_idx = torch.cat([
                torch.arange(g * group_size, (g + 1) * group_size, device=device)
                for g in pick_groups
            ])
            Hinv_block_inv = torch.linalg.inv(H_inv[pick_idx][:, pick_idx])  # ≈ H[pick_idx, pick_idx]
            W -= H_inv[:, pick_idx] @ Hinv_block_inv @ W[pick_idx, :]
            W[pick_idx, :] = 0
            H_inv -= H_inv[:, pick_idx] @ Hinv_block_inv @ H_inv[pick_idx, :]
            H_inv[pick_idx, :] = 0
            H_inv[:, pick_idx] = 0
            pruned_group_mask[pick_groups] = True
        W_pruned = torch.zeros_like(W)
        kept_mask = (~torch.repeat_interleave(pruned_group_mask, repeats=group_size))
        try:
            W_pruned[kept_mask, :] = torch.linalg.inv(H[kept_mask][:, kept_mask]) @ G[kept_mask, :]
        except Exception:
            H_cpu = H[kept_mask][:, kept_mask].cpu()
            G_cpu = G[kept_mask, :].cpu()
            W_pruned[kept_mask, :] = (torch.linalg.inv(H_cpu) @ G_cpu).to(device)
        prune_loss = torch.sum(-W_pruned * G + 0.5 * W_pruned * (H @ W_pruned)).detach().item()
        return W_pruned, prune_loss, pruned_group_mask