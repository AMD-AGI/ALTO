import importlib
from unittest.mock import patch

SUPPORTED_MODELS = ["llama3"]
PATCH_MODULES = ["model", "state_dict_adapter"]


class ModelPatcher:
    _patched = False

    @classmethod
    def patch(cls):
        if cls._patched:
            return
        cls._patched = True
        for model_name in SUPPORTED_MODELS:
            model_module = importlib.import_module(
                f"torchtitan.models.{model_name}")
            for patch_module in PATCH_MODULES:
                source_module = importlib.import_module(
                    f"modeloptimizer.models.{model_name}.{patch_module}")
                target_module = importlib.import_module(
                    f"torchtitan.models.{model_name}.model.{patch_module}")
                for attr_name in source_module.__all__:
                    patched_attr = getattr(source_module, attr_name)
                    print(
                        f"Patching {attr_name} of {model_name}: {patched_attr}")
                    original_attr = getattr(target_module, attr_name)
                    print(
                        f"Original {attr_name} of {model_name}: {original_attr}"
                    )
                    setattr(target_module, attr_name, patched_attr)
                    patch(
                        f"torchtitan.models.{model_name}.model.{patch_module}.{attr_name}",
                        patched_attr,
                    ).__enter__()
                    if hasattr(model_module, attr_name):
                        setattr(model_module, attr_name, patched_attr)
