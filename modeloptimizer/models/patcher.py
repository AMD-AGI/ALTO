import importlib
from unittest.mock import patch

import torch

SUPPORTED_MODELS = ["llama3", "gpt_oss", "deepseek_v3"]
PATCH_MODULES = ["config_registry", "state_dict_adapter"]


class ModelPatcher:
    _patched = False

    @classmethod
    def patch(cls):
        if cls._patched:
            return
        cls._patched = True

        cls.patch_fake_quantize()
        cls.patch_apply_rotary_emb_complex()

        for model_name in SUPPORTED_MODELS:
            model_module = importlib.import_module(f"torchtitan.models.{model_name}")
            for patch_module in PATCH_MODULES:
                try:
                    source_module = importlib.import_module(f"modeloptimizer.models.{model_name}.{patch_module}")
                except ImportError:
                    continue
                target_module = importlib.import_module(f"torchtitan.models.{model_name}.{patch_module}")
                for attr_name in source_module.__all__:
                    patched_attr = getattr(source_module, attr_name)
                    # print(
                    #     f"Patching {attr_name} of {model_name}: {patched_attr}")
                    original_attr = getattr(target_module, attr_name, None)
                    # print(
                    #     f"Original {attr_name} of {model_name}: {original_attr}"
                    # )
                    setattr(target_module, attr_name, patched_attr)
                    patch(
                        f"torchtitan.models.{model_name}.{patch_module}.{attr_name}",
                        patched_attr,
                    ).__enter__()
                    if hasattr(model_module, attr_name):
                        setattr(model_module, attr_name, patched_attr)

    @classmethod
    def patch_fake_quantize(cls):
        from compressed_tensors.quantization.lifecycle import forward as forward_module

        original_fake_quantize = forward_module.fake_quantize

        class FakeQuantizeFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, scale, zero_point, args, g_idx, global_scale):
                return original_fake_quantize(x, scale, zero_point, args, g_idx, global_scale)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None, None, None, None

        def fake_quantize(x, scale, zero_point, args, g_idx, global_scale):
            return FakeQuantizeFunction.apply(
                x,
                scale,
                zero_point,
                args,
                g_idx,
                global_scale,
            )

        forward_module.fake_quantize = fake_quantize

    @classmethod
    def patch_apply_rotary_emb_complex(cls):
        from torchtitan.models.common import rope, attention
        original_apply_rotary_emb_complex = rope.apply_rotary_emb_complex

        def apply_rotary_emb_complex(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
            positions: torch.Tensor | None = None,
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
            return original_apply_rotary_emb_complex(xq, xk, freqs_cis, positions)

        rope.apply_rotary_emb_complex = apply_rotary_emb_complex
        attention.apply_rotary_emb_complex = apply_rotary_emb_complex
        patch("torchtitan.models.common.rope.apply_rotary_emb_complex", apply_rotary_emb_complex).__enter__()
