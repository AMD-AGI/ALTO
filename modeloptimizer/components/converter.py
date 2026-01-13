# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from functools import partial
from importlib.util import find_spec
from typing import Any, List
from inspect import signature

import torch
import torch.nn as nn

from torchtitan.components.quantization import (
    QuantizationConverter,)
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe.utils import set_token_group_alignment_size_m
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

from modeloptimizer.config import QuantConfig
from modeloptimizer.optimization.quantization import nn as qnn


def replace_to_light(
    model: nn.Module,
    name: str,
    config: QuantConfig,
    mapping: Dict[type, type],
):
    original_type = type(model)
    if original_type in mapping.keys():
        lmodule = new_lightmodule(original_type, mapping[original_type], model,
                                  name, config)
        return lmodule
    for n, mod in model.named_children():
        model.add_module(
            n,
            replace_to_light(
                mod,
                name + ("." if name else "") + n,
                config,
                mapping,
            ),
        )
    return model


def unpack(attrs, non_param_attrs, param_attrs):
    for k, v in attrs.items():
        if k in ["_modules"]:
            continue
        if k in ["_parameters", "_buffers"]:
            unpack(v, non_param_attrs, param_attrs)
        elif isinstance(v, nn.Parameter):
            param_attrs.append(k)
        else:
            non_param_attrs[k] = v if k != 'bias' else v is not None
    return non_param_attrs, param_attrs


def new_lightmodule(
    original_class: type,
    lightmodule_class: type,
    module: nn.Module,
    name: str,
    config: QuantConfig,
):
    quant_config = config.get_layer_config(name, original_class.__name__)
    non_param_attrs, param_attrs = unpack(vars(module), {}, [])
    init_lightargs_keys = signature(lightmodule_class).parameters.keys()
    init_lightargs = {
        k: v for k, v in non_param_attrs.items() if k in init_lightargs_keys
    }
    # init light module
    lightmodule = lightmodule_class(quant_config=quant_config, **init_lightargs)
    # load parameters
    for name in param_attrs + list(non_param_attrs.keys()):
        setattr(lightmodule, name, getattr(module, name))
    return lightmodule


class ModelOptConverter(QuantizationConverter):

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)

        self.quant_config: QuantConfig = job_config.quantization
        self.mapping = {nn.Linear: qnn.LLinear}

        # TODO: set gmm alignment to quantization blocksize
        # set_token_group_alignment_size_m(ALIGN_SIZE_M)
        # logger.info(
        #     f"Setting token group alignment size to {ALIGN_SIZE_M}")

    def convert(self, model: nn.Module):
        replace_to_light(model, "", self.quant_config, self.mapping)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        return

    def post_initialization(self, model: nn.Module | list[nn.Module]):
        return


register_model_converter(ModelOptConverter, "modeloptimizer")
