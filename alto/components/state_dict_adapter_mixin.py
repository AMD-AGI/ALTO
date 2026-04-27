# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import re

from torch import Tensor

__all__ = ["StateDictAdapterMixin"]


class StateDictAdapterMixin:

    def populate_extra_map(self):
        # collect candidate layers
        self.candidate_layers = {
            k[:-len(".weight")]: v[:-len(".weight")] for k, v in self.from_hf_map.items() if k.endswith(".weight")
        }

        self.sequential_layers = {}
        for k, v in self.candidate_layers.items():
            if "{}" in k:
                hf_name = k[:k.index("{}") + 2]
                if hf_name not in self.sequential_layers:
                    layer_name = v[:v.index("{}") + 2]
                    self.sequential_layers[hf_name] = layer_name

        self.candidate_layers.update(self.sequential_layers)

        # construct extra mappings for model optimizer states
        bitmask_params = ["compressed", "bitmask", "shape"]
        self.extra_map = {}
        for layer_name, target_name in self.candidate_layers.items():
            for bitmask_param in bitmask_params:
                self.extra_map[f"{layer_name}.{bitmask_param}"] = f"{target_name}.{bitmask_param}"
            base_names = ["weight", "input", "output", "sparsity", "student_output"]
            if "q_proj" in layer_name:
                base_names.append("q")
            if "k_proj" in layer_name:
                base_names.append("k")
            if "v_proj" in layer_name:
                base_names.append("v")
            for base_name in base_names:
                self.extra_map[f"{layer_name}.{base_name}_scale"] = f"{target_name}.{base_name}_scale"
                self.extra_map[f"{layer_name}.{base_name}_zero_point"] = f"{target_name}.{base_name}_zero_point"
                self.extra_map[
                    f"{layer_name}.{base_name}_observer.quant_min"] = f"{target_name}.{base_name}_observer.quant_min"
                self.extra_map[
                    f"{layer_name}.{base_name}_observer.quant_max"] = f"{target_name}.{base_name}_observer.quant_max"
                self.extra_map[f"{layer_name}.{base_name}_observer.stats"] = f"{target_name}.{base_name}_observer.stats"
                self.extra_map[
                    f"{layer_name}.{base_name}_observer.num_samples"] = f"{target_name}.{base_name}_observer.num_samples"
        self.from_hf_map.update(self.extra_map)

    def update_storage_plan(self, state_dict: dict[str, Tensor]):
        if self.fqn_to_index_mapping is None:
            return
        extra_indices = {}
        for key, idx in self.fqn_to_index_mapping.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
            else:
                abstract_key = key
                layer_num = None
            abstract_key_prefix = abstract_key[:-len("weight")]
            # collect extra states starts with abstract_key_prefix
            extra_keys = [
                k.format(layer_num) if layer_num is not None else k
                for k in self.extra_map.keys()
                if k.startswith(abstract_key_prefix)
            ]
            extra_indices.update({k: self.fqn_to_index_mapping.get(k, idx) for k in extra_keys if k in state_dict})

        self.fqn_to_index_mapping.update(extra_indices)

    def map_ignore_list_to_hf(self, ignore_list: list[str]) -> list[str]:
        reverse_candidate_layers = {v: k for k, v in self.candidate_layers.items()}
        new_ignore_list = []
        for key in ignore_list:
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = reverse_candidate_layers[abstract_key]
                new_key = new_key.format(layer_num)
                new_ignore_list.append(new_key)
            else:
                new_ignore_list.append(key)
        return new_ignore_list
