# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""ALTO's apply_quantization_config, extended with per-tensor Conv2d quantization."""

from copy import deepcopy

from torch import nn
from compressed_tensors.offload import unwrap_offload_forward
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationMetadata,
    QuantizationStatus,
    apply_quantization_config as _apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import set_forward_quantized
from compressed_tensors.quantization.lifecycle.initialize import initialize_qparams
from compressed_tensors.utils import match_named_modules, match_targets


def apply_quantization_config_conv2d(model: nn.Module, config: QuantizationConfig) -> None:
    """Initialize Conv2d alongside the upstream Linear/Embedding path.

    compressed-tensors 0.18 skips Conv2d even when a recipe targets it. Only
    per-tensor qparams are supported here: Linear's other layouts assume a 2-D
    weight. MX block axes are handled separately by ALTO's format dispatch.
    """
    config = deepcopy(config)
    convs = []
    target_to_scheme = {target: scheme for scheme in config.config_groups.values() for target in scheme.targets}
    for name, module in match_named_modules(model, target_to_scheme, config.ignore):
        if not isinstance(module, nn.Conv2d):
            continue
        scheme = target_to_scheme[match_targets(name, module, target_to_scheme)[0]]
        fields = (
            ("input", scheme.input_activations),
            ("weight", scheme.weights),
            ("output", scheme.output_activations),
        )
        for base_name, qargs in fields:
            if qargs is not None and qargs.strategy != "tensor":
                raise ValueError(f"Conv2d supports per-tensor quantization only; got "
                                 f"strategy {qargs.strategy!r} for {name}.{base_name}")
        convs.append((module, scheme, fields))

    _apply_quantization_config(model, config)
    for module, scheme, fields in convs:
        QuantizationMetadata.clear_all_qparams(module)
        for base_name, qargs in fields:
            if qargs is not None:
                initialize_qparams(
                    module,
                    base_name,
                    qargs,
                    observed_shape=module.weight.shape,
                    observed_dtype=module.weight.dtype,
                    force_zero_point=config.quantization_status < QuantizationStatus.COMPRESSED,
                )
        with unwrap_offload_forward(module):
            set_forward_quantized(module)
        module.quantization_scheme = scheme
        module.quantization_status = config.quantization_status
