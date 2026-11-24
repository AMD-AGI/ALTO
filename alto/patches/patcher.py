# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import dataclasses
import importlib
import sys

import torch

SUPPORTED_MODELS = ["llama3", "gpt_oss", "deepseek_v3"]


class ModelPatcher:
    _patched = False

    @classmethod
    def patch(cls):
        if cls._patched:
            return
        cls._patched = True

        cls.patch_fake_quantize()

        from alto.models.build_record import install_build_recorder

        for model_name in SUPPORTED_MODELS:
            install_build_recorder(model_name)
            cls.patch_state_dict_adapter(model_name)

    @classmethod
    def patch_model_config(cls, model_config):
        """Use HF-layout CosSin RoPE for ALTO Llama models only."""
        from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
        from torchtitan.models.llama3.model import Llama3Model

        if not isinstance(model_config, Llama3Model.Config):
            return

        for _fqn, config, parent, attr in model_config.traverse(ComplexRoPE.Config):
            assert parent is not None and attr is not None
            if type(config) is ComplexRoPE.Config:
                setattr(
                    parent,
                    attr,
                    CosSinRoPE.Config(
                        **{
                            field.name: getattr(config, field.name)
                            for field in dataclasses.fields(config)
                            if field.init
                        }
                    ),
                )

    @classmethod
    def patch_state_dict_adapter(cls, model_name: str):
        """Replace a model's state dict adapter with ALTO's subclass of the same name.

        The trainer builds the adapter from the model class attribute
        ``state_dict_adapter_cls``; module attributes are patched as well for
        code that imports the adapter by name.
        """
        try:
            source_module = importlib.import_module(f"alto.models.{model_name}.state_dict_adapter")
        except ImportError:
            return
        model_module = importlib.import_module(f"torchtitan.models.{model_name}")
        target_module = importlib.import_module(f"torchtitan.models.{model_name}.state_dict_adapter")
        model_cls_module = importlib.import_module(f"torchtitan.models.{model_name}.model")
        model_classes = [
            obj for obj in vars(model_cls_module).values()
            if isinstance(obj, type) and "state_dict_adapter_cls" in vars(obj)
        ]
        for attr_name in source_module.__all__:
            patched_attr = getattr(source_module, attr_name)
            for model_cls in model_classes:
                if model_cls.state_dict_adapter_cls.__name__ == attr_name:
                    model_cls.state_dict_adapter_cls = patched_attr
            setattr(target_module, attr_name, patched_attr)
            if hasattr(model_module, attr_name):
                setattr(model_module, attr_name, patched_attr)

    @classmethod
    def patch_quantization_scales(cls):
        """Add a PoT scale policy to the existing INT8 qparam calculation.

        Dynamic range reduction and QDQ remain compressed-tensors operations.
        Patch its shared qparam helper plus the aliases used by ALTO observers
        so static weights and dynamic activations use the same scale policy.
        """
        from compressed_tensors.quantization import utils
        from compressed_tensors.quantization.utils import helpers
        from alto.modifiers.quantization.format_registry import validate_scale_type

        observer_base = sys.modules.get("alto.observers.base")
        original = helpers.calculate_qparams
        if getattr(original, "_alto_scale_type", False):
            # Observers can be imported after this patch was installed.
            if observer_base is not None:
                observer_base.calculate_qparams = original
            return

        @torch.no_grad()
        def calculate_qparams(min_vals, max_vals, quantization_args, global_scale=None):
            if getattr(quantization_args, "scale_type", "float") != "pot":
                return original(min_vals, max_vals, quantization_args, global_scale=global_scale)
            validate_scale_type(quantization_args)
            if global_scale is not None:
                raise ValueError("scale_type=pot does not support global_scale")
            # PoT uses the positive INT8 endpoint (127) as its scale divisor.
            # Derive this from the observed range to avoid rounding a scale twice.
            scale = torch.maximum(min_vals.abs(), max_vals.abs()) / 127
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            scale = torch.pow(2, torch.ceil(torch.log2(scale)))
            zero_point = torch.zeros_like(scale, dtype=quantization_args.zp_dtype)
            if scale.ndim == 0:
                scale, zero_point = scale.reshape(1), zero_point.reshape(1)
            return scale, zero_point

        calculate_qparams._alto_scale_type = True
        helpers.calculate_qparams = calculate_qparams
        utils.calculate_qparams = calculate_qparams
        if observer_base is not None:
            observer_base.calculate_qparams = calculate_qparams

    @classmethod
    def patch_fake_quantize(cls):
        from compressed_tensors.quantization.lifecycle import forward as forward_module

        cls.patch_quantization_scales()

        original_fake_quantize = forward_module.fake_quantize

        class FakeQuantizeFunction(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, scale, zero_point, args, g_idx, global_scale):
                target_dtype = getattr(args, "format", None)
                if target_dtype in ("mx6", "mx9"):
                    # Dispatches to the REAL packed Triton kernel (quantize ->
                    # bit-packed bytes -> dequantize), not the torch fake-quant
                    # emulation -- this is the intentional, validated design
                    # (commit 95b1114: LLaMA-3.2-1B wikitext loss 2.1141 == the
                    # fake-quant path's 2.1140). Requires a CUDA tensor (Triton
                    # kernel launch). num_bits / group_size are fixed by the
                    # format rather than honoured, so they are validated here
                    # instead of forwarded -- a recipe that disagrees with the
                    # format would otherwise be silently ignored.
                    from alto.kernels.mx import MX_QUANT_BIT, convert_to_mx, convert_from_mx

                    assert args.group_size in (None, 16), \
                        f"{target_dtype} packed kernel only supports group_size 16, got {args.group_size}"
                    assert args.num_bits == MX_QUANT_BIT[target_dtype], \
                        (f"{target_dtype} packed kernel is fixed at num_bits "
                         f"{MX_QUANT_BIT[target_dtype]}, got {args.num_bits}")
                    axis = args.block_axis
                    packed = convert_to_mx(x, target_dtype=target_dtype, axis=axis)
                    return convert_from_mx(packed, target_dtype, x.dtype, x.shape, axis=axis)
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
