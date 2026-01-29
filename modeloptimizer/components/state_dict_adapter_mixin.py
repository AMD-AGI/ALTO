import re

__all__ = ["StateDictAdapterMixin"]


class StateDictAdapterMixin:

    def populate_extra_map(self):
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
