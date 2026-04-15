# modified from https://github.com/pytorch/ao/blob/f4b3c29f0dc1fe2f30d27340da0682b14b8fe187/torchao/prototype/moe_training/conversion_utils.py#L28
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

from typing import Callable, Optional, Type

import torch
from torch import nn
from torchtitan.tools.logging import logger

from .config import TrainingOpConfig


def _get_tensor_cls_for_config(config: TrainingOpConfig) -> Type[torch.Tensor]:
    """
    Returns the appropriate tensor class for the given config.
    """
    from .tensor import MXFP4TrainingWeightWrapperTensor

    if config.precision == "mxfp4":
        return MXFP4TrainingWeightWrapperTensor
    else:
        raise ValueError(f"Unsupported training op config: {config}")


def swap_params(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[TrainingOpConfig] = None,
    target_parameter_name: Optional[str] = None,
    module_name: Optional[str] = None,
) -> nn.Module:
    """
    Recurses through the nn.Module, recursively swapping the data tensor of
    each nn.Parameter with a training tensor subclass. Only applies if the module
    passed the module_filter_fn, if specified.

    The tensor class used is determined by the config type.

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Parameter` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from .tensor import TrainingWeightWrapperBaseTensor

    if config is None:
        raise ValueError("training op config is required")

    tensor_cls = _get_tensor_cls_for_config(config)

    if isinstance(module, nn.Parameter) and (module_filter_fn is None or module_filter_fn(module, "")):
        if len(list(module.children())) > 0:
            raise AssertionError(f"Does not support a root nn.Parameter with children: {module}")
        if not isinstance(module.data, TrainingWeightWrapperBaseTensor):
            new_data = tensor_cls(module.data, config)
            return nn.Parameter(new_data, requires_grad=module.requires_grad)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)
        if module_filter_fn is None or module_filter_fn(module, cur_fqn):
            for param_name, param in module.named_parameters(recurse=False):
                if (target_parameter_name is not None and param_name != target_parameter_name):
                    continue
                if not isinstance(param.data, TrainingWeightWrapperBaseTensor):
                    new_param = nn.Parameter(
                        tensor_cls(param.data, config),
                        requires_grad=param.requires_grad,
                    )
                    setattr(module, param_name, new_param)
                    logger.info(
                        f"Swapped {module_name}{'.' if module_name else ''}{cur_fqn}{'.' if cur_fqn else ''}{param_name} to {tensor_cls.__name__}"
                    )

                    def get_name_func_new():
                        orignal_name = module.__class__.__name__
                        return f"{orignal_name}[{config}]"

                    module._get_name = get_name_func_new

    post_order_traversal(root_module)
    return root_module
