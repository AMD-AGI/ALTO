from unittest.mock import patch
from typing import Callable
from functools import partial
from contextlib import contextmanager

import torch
from torchtitan.models.llama3 import Llama3StateDictAdapter, get_train_spec
from torchtitan.models.llama3.model import model as llama3_model_module


class PatchedLlama3StateDictAdapter(Llama3StateDictAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: wq, wk scale/zero_point permutation
        # add quantization states

        # collect candidate layers
        self.candidate_layers = {
            k[:-len(".weight")]: v[:-len(".weight")]
            for k, v in self.from_hf_map.items()
            if k.endswith(".weight")
        }
        # construct extra mappings for model optimizer states
        bitmask_params = ["compressed", "bitmask", "shape"]
        self.extra_map = {}
        for layer_name, target_name in self.candidate_layers.items():
            for bitmask_param in bitmask_params:
                self.extra_map[
                    f"{layer_name}.{bitmask_param}"] = f"{target_name}.{bitmask_param}"
            base_names = [
                "weight", "input", "output", "sparsity", "sparsity_owl"
            ]
            if "q_proj" in layer_name:
                base_names.append("q")
            if "k_proj" in layer_name:
                base_names.append("k")
            if "v_proj" in layer_name:
                base_names.append("v")
            for base_name in base_names:
                self.extra_map[
                    f"{layer_name}.{base_name}_scale"] = f"{target_name}.{base_name}_scale"
                self.extra_map[
                    f"{layer_name}.{base_name}_zero_point"] = f"{target_name}.{base_name}_zero_point"
                if base_name.startswith("sparsity"):
                    self.extra_map[
                        f"{layer_name}.{base_name}_observer.stats"] = f"{target_name}.{base_name}_observer.stats"
                    self.extra_map[
                        f"{layer_name}.{base_name}_observer.num_samples"] = f"{target_name}.{base_name}_observer.num_samples"
                else:
                    self.extra_map[
                        f"{layer_name}.{base_name}_observer.quant_min"] = f"{target_name}.{base_name}_observer.quant_min"
                    self.extra_map[
                        f"{layer_name}.{base_name}_observer.quant_max"] = f"{target_name}.{base_name}_observer.quant_max"
        self.from_hf_map.update(self.extra_map)

        # TODO: update fqn_to_index_mapping
        assert self.fqn_to_index_mapping is None

    def _permute(self, w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return w

    def _reverse_permute(self, w: torch.Tensor, *args,
                         **kwargs) -> torch.Tensor:
        return w


def patched_get_train_spec():
    train_spec = get_train_spec()
    train_spec.state_dict_adapter = PatchedLlama3StateDictAdapter
    return train_spec


patcher = patch("torchtitan.models.llama3.get_train_spec",
                patched_get_train_spec)
patcher.__enter__()


def patched_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                   tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    xq = xq.reshape(
        *xq.shape[:-1],
        2,
        head_dim // 2,
    ).transpose(-1, -2).reshape(
        *xq.shape[:-1],
        head_dim,
    ).contiguous()
    xk = xk.reshape(
        *xk.shape[:-1],
        2,
        head_dim // 2,
    ).transpose(-1, -2).reshape(
        *xk.shape[:-1],
        head_dim,
    ).contiguous()
    return func(xq, xk, freqs_cis)


@contextmanager
def patch_apply_rotary_emb():
    original_apply_rotary_emb = llama3_model_module.apply_rotary_emb
    llama3_model_module.apply_rotary_emb = partial(patched_apply_rotary_emb, func=original_apply_rotary_emb)
    yield
    llama3_model_module.apply_rotary_emb = original_apply_rotary_emb


patch_apply_rotary_emb().__enter__()
