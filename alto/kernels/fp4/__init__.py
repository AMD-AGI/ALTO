# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Unified public entrypoint for FP4 kernels.

This module exposes the most commonly used public APIs for both FP4 families:

- shared E2M1 encode/decode primitives from ``fp4_common``
- MXFP4 quantize/dequantize entrypoints
- NVFP4 quantize/dequantize entrypoints

Format-specific internals remain in their own subpackages:

- ``alto.kernels.fp4.fp4_common``
- ``alto.kernels.fp4.mxfp4``
- ``alto.kernels.fp4.nvfp4``
"""

from importlib import import_module


_LAZY_ATTRS = {
    "fp4_common": (".fp4_common", None),
    "mxfp4": (".mxfp4", None),
    "nvfp4": (".nvfp4", None),
    "dequantize_e2m1": (".fp4_common", "dequantize_e2m1"),
    "generate_philox_randval_2x": (".fp4_common", "generate_philox_randval_2x"),
    "quantize_e2m1": (".fp4_common", "quantize_e2m1"),
    "MXFP4_BLOCK_SIZE_DEFAULT": (".mxfp4.mxfp_quantization", "BLOCK_SIZE_DEFAULT"),
    "convert_from_mxfp4": (".mxfp4.mxfp_quantization", "convert_from_mxfp4"),
    "convert_to_mxfp4": (".mxfp4.mxfp_quantization", "convert_to_mxfp4"),
    "NVFP4_BLOCK_SIZE_DEFAULT": (".nvfp4.nvfp_quantization", "BLOCK_SIZE_DEFAULT"),
    "compute_dynamic_outer_scale": (
        ".nvfp4.nvfp_quantization",
        "compute_dynamic_outer_scale",
    ),
    "convert_from_nvfp4": (".nvfp4.nvfp_quantization", "convert_from_nvfp4"),
    "convert_to_nvfp4": (".nvfp4.nvfp_quantization", "convert_to_nvfp4"),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = (
    "MXFP4_BLOCK_SIZE_DEFAULT",
    "NVFP4_BLOCK_SIZE_DEFAULT",
    "compute_dynamic_outer_scale",
    "convert_from_mxfp4",
    "convert_from_nvfp4",
    "convert_to_mxfp4",
    "convert_to_nvfp4",
    "dequantize_e2m1",
    "fp4_common",
    "generate_philox_randval_2x",
    "mxfp4",
    "nvfp4",
    "quantize_e2m1",
)
