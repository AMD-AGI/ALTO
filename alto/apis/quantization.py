# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Build, load, and apply post-training quantization recipes."""

import copy
import os
import re
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
from torchtitan.tools.logging import logger

from alto.config import Recipe
from alto.kernels.mx import MX_QUANT_BIT
from alto.modifiers.quantization import QuantizationModifier

__all__ = [
    "apply_quantization",
    "create_quantization_recipe",
    "load_quantization_recipe",
]

SCHEMES = (*MX_QUANT_BIT, "int8_dynamic", "int8_static")

# MX blocks follow the input features for Linear and input channels for Conv2d.
_BLOCK_AXIS = {"Linear": -1, "Conv2d": -3}

# The token strategy keeps dims 0 and 1, so on a conv input it would scale each
# input channel, the axis the conv reduces over.
_INT8_DYNAMIC_STRATEGY = {"Linear": "token", "Conv2d": "tensor"}


def create_quantization_recipe(
        scheme: str,
        targets: Sequence[str] = ("Linear",),
        ignore: Sequence[str] = (),
) -> Recipe:
    """Build a weight + input-activation recipe, one group per module type.

    :param scheme: fully dynamic ``"mx6"``/``"mx9"``; ``"int8_dynamic"``, with
        static per-tensor weights and activations scaled at run time, per token
        on Linear and per tensor on Conv2d; or static per-tensor
        ``"int8_static"`` with minmax activation calibration
    :param targets: module types, out of ``Linear`` and ``Conv2d``. Conv2d
        requires the caller to install its quantization initialization first
    :param ignore: module names covering their subtree, or ``re:`` patterns
    :return: a recipe for :func:`apply_quantization`
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported scheme {scheme!r}; choose one of: {', '.join(SCHEMES)}")
    targets = list(dict.fromkeys([targets] if isinstance(targets, str) else targets))
    if not targets or any(target not in _BLOCK_AXIS for target in targets):
        raise ValueError(f"targets must be drawn from {', '.join(_BLOCK_AXIS)}; got {targets}")
    ignore = [ignore] if isinstance(ignore, str) else list(ignore)

    config_groups = {}
    for target in targets:
        weights = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "tensor"}
        if scheme in MX_QUANT_BIT:
            weights.update(num_bits=MX_QUANT_BIT[scheme], dynamic=True,
                           format=scheme, block_axis=_BLOCK_AXIS[target])
        inputs = dict(weights)
        if scheme == "int8_dynamic":
            inputs.update(strategy=_INT8_DYNAMIC_STRATEGY[target], dynamic=True)
        elif scheme == "int8_static":
            inputs["observer"] = "minmax"
        config_groups[f"group_{target.lower()}"] = {
            "targets": [target],
            "weights": weights,
            "input_activations": inputs,
        }

    patterns = [name if name.startswith("re:") else f"re:{re.escape(name)}($|\\.)" for name in ignore]
    modifier = QuantizationModifier(config_groups=config_groups, ignore=patterns, sequential=False)
    return Recipe.from_modifiers(modifier, modifier_group_name="quantization")


def load_quantization_recipe(path: Union[str, os.PathLike]) -> Recipe:
    """Read a YAML or JSON file using the existing Recipe parser.

    Entries in ``ignore`` retain compressed_tensors matching semantics:
    exact module names, class names, or ``re:`` patterns.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no recipe file at {path}")
    return Recipe.create_instance(path)


def apply_quantization(
    model: nn.Module,
    recipe: Union[Recipe, str, os.PathLike],
    calibrate: Optional[Callable[[], object]] = None,
) -> nn.Module:
    """Quantize a model in place through the QuantizationModifier lifecycle.

    Leave ``nn.MultiheadAttention.out_proj`` in ``ignore``: the attention passes
    its weight to ``F.multi_head_attention_forward`` without calling it, so
    neither its quantized forward nor its input observer ever runs. A static
    recipe then fails in ``post_step`` on the empty observer, and a dynamic one
    leaves it in full precision.

    :param model: the module to quantize
    :param recipe: a Recipe or recipe file path; copied before use
    :param calibrate: runs the model over calibration data; required when the
        recipe needs that data, and skipped otherwise
    :return: ``model``
    """
    if not isinstance(recipe, Recipe):
        recipe = load_quantization_recipe(recipe)
    recipe = copy.deepcopy(recipe)
    if not recipe.modifiers:
        raise ValueError("the recipe holds no modifiers")
    for modifier in recipe.modifiers:
        if type(modifier) is not QuantizationModifier:
            raise ValueError(f"{type(modifier).__name__} is not supported: apply_quantization runs "
                             f"QuantizationModifier only")
        if modifier.sequential:
            raise ValueError("sequential=True is not supported: it finds blocks through model.layers")
        if not modifier.has_config():
            raise ValueError("a QuantizationModifier in the recipe carries no quantization config")

    needs_calibration = any(modifier.resolved_config.requires_calibration_data() for modifier in recipe.modifiers)
    if needs_calibration and calibrate is None:
        raise ValueError("the recipe needs calibration data: pass calibrate, a callable that runs the model over it")
    if not needs_calibration and calibrate is not None:
        logger.warning("the recipe needs no calibration data; calibrate will not be called")

    parts = [model]
    for modifier in recipe.modifiers:
        # Not convert(): it also writes torchtitan's _param_init, which a plain nn.Module lacks.
        modifier.initialize_quantization(model)
        modifier.initialize(parts)
    for modifier in recipe.modifiers:
        modifier.pre_step(parts)
    if needs_calibration:
        with torch.no_grad():
            calibrate()
    for modifier in recipe.modifiers:
        modifier.post_step(parts)
    for modifier in recipe.modifiers:
        modifier.finalize(parts)
    return model
