from unittest.mock import patch
from torchtitan.models.llama3 import Llama3StateDictAdapter, get_train_spec


class PatchedLlama3StateDictAdapter(Llama3StateDictAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: wq, wk scale/zero_point permutation
        # add quantization states
        extra_map = {}
        for k, v in self.from_hf_map.items():
            if k.endswith(".weight"):
                layer_name = k[:-len(".weight")]
                target_name = v[:-len(".weight")]
                base_names = ["weight", "input", "output", "sparsity", "sparsity_owl"]
                if "q_proj" in k:
                    base_names.append("q")
                if "k_proj" in k:
                    base_names.append("k")
                if "v_proj" in k:
                    base_names.append("v")
                for base_name in base_names:
                    extra_map[
                        f"{layer_name}.{base_name}_scale"] = f"{target_name}.{base_name}_scale"
                    extra_map[
                        f"{layer_name}.{base_name}_zero_point"] = f"{target_name}.{base_name}_zero_point"
                    if base_name.startswith("sparsity"):
                        extra_map[
                            f"{layer_name}.{base_name}_observer.stats"] = f"{target_name}.{base_name}_observer.stats"
                        extra_map[
                            f"{layer_name}.{base_name}_observer.num_samples"] = f"{target_name}.{base_name}_observer.num_samples"
                    else:
                        extra_map[
                            f"{layer_name}.{base_name}_observer.quant_min"] = f"{target_name}.{base_name}_observer.quant_min"
                        extra_map[
                            f"{layer_name}.{base_name}_observer.quant_max"] = f"{target_name}.{base_name}_observer.quant_max"
        self.from_hf_map.update(extra_map)


def patched_get_train_spec():
    train_spec = get_train_spec()
    train_spec.state_dict_adapter = PatchedLlama3StateDictAdapter
    return train_spec


patcher = patch("torchtitan.models.llama3.get_train_spec",
                patched_get_train_spec)
patcher.__enter__()
