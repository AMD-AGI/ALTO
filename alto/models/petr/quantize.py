# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Fully-dynamic MX6 / MX9 recipes.

Model-agnostic: the part of a model to leave alone is named by pattern, not by
an attribute name a particular architecture happens to use, so the same entry
point works on a detector, an LLM, or anything else built out of ``nn.Conv2d``
and ``nn.Linear``.

Mixed precision comes from ``ignore``. A model already runs in some dtype, and
the modules a recipe skips keep it, so "MX6 here, BF16 there" is one recipe with
the BF16 half excluded.

Scales come from the tensor being quantized on each forward pass, so no
calibration data, observers or persisted scales are involved.

The public surface is :func:`apply_mx_quantization`; building a recipe and
driving its modifiers through ``convert -> initialize -> pre_step -> post_step
-> finalize`` is internal.
"""

import re
import torch.nn as nn
from compressed_tensors.utils import match_named_modules
from alto.config import Recipe

_MX_FORMAT_BITS = {"mx6": 5, "mx9": 8}

# Axis the MX blocks run along, per module type. A Conv2d is blocked along its
# input channels;
_MX_BLOCK_AXIS = {nn.Conv2d: 1, nn.Linear: -1}


def _mx_quant_args(mx_format: str, block_axis: int) -> dict:
    """Fully-dynamic MX quantization args.

    ``num_bits`` is required by ``compressed_tensors`` for bookkeeping and
    follows the format's fixed physical layout; the Triton codec still owns the
    actual packed representation.
    """
    try:
        num_bits = _MX_FORMAT_BITS[mx_format]
    except KeyError as exc:
        supported = ", ".join(sorted(_MX_FORMAT_BITS))
        raise ValueError(
            f"unsupported MX format {mx_format!r}; choose one of: {supported}"
        ) from exc

    return {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": True,
        "format": mx_format,
        "block_axis": block_axis,
    }


def _create_mx_recipe(mx_format: str, ignore_patterns=()) -> Recipe:
    """Build an MX W+A recipe for every Conv2d and Linear.

    Args:
        mx_format: ``"mx6"`` or ``"mx9"``.
        ignore_patterns: ``re:`` patterns naming the modules left out of the
            recipe, so they keep the model's dtype.
    """
    config_groups = {
        # Split per module type because MX blocks run along a different axis for
        # each.
        f"group_{cls.__name__.lower()}": {
            "targets": [cls.__name__],
            "weights": dict(_mx_quant_args(mx_format, axis)),
            "input_activations": dict(_mx_quant_args(mx_format, axis)),
        }
        for cls, axis in _MX_BLOCK_AXIS.items()
    }
    # ignore sits on the modifier, not on a scheme: compressed_tensors applies it
    # across every config group.
    modifier = {"sequential": False, "config_groups": config_groups}
    if ignore_patterns:
        modifier["ignore"] = list(ignore_patterns)
    return Recipe.from_dict(
        {
            "quantization_stage": {
                "quantization_modifiers": {"QuantizationModifier": modifier}
            }
        }
    )


def _apply_recipe(model, recipe):
    """Drive a recipe's modifiers through their lifecycle against ``model``."""
    modifiers = recipe.modifiers
    if not modifiers:
        raise ValueError(f"recipe {recipe} produced no modifiers")

    model_parts = [model]
    for modifier in modifiers:
        modifier.convert(model)
    for modifier in modifiers:
        modifier.initialize(model_parts)
    for modifier in modifiers:
        modifier.pre_step(model_parts)
    for modifier in modifiers:
        modifier.post_step(model_parts)
    for modifier in modifiers:
        modifier.finalize(model_parts)
    return model


def apply_mx_quantization(model, mx_format: str, ignore=()):
    """Apply an MX6 or MX9 W+A recipe in place.

    Every Conv2d and Linear is quantized unless ``ignore`` excludes it; excluded
    modules keep the dtype the model already runs in.

    Raises:
        ValueError: if the recipe would reach nothing, or if ``ignore`` was given
            but excludes nothing. Neither fails on its own -- a pattern that
            matches no module quantizes the whole model quietly -- and both mean
            the run will not measure what was asked for. Both are checked before
            the model is touched, so a rejected call leaves it unmodified.
    """
    # A bare name is matched against the module name exactly, which would select
    # only the container -- not a Conv2d or Linear, and so exclude nothing. Widen
    # it to the subtree; re: patterns pass through.
    patterns = [
        p if p.startswith("re:") else f"re:{re.escape(p)}($|\\.)" for p in ignore
    ]
    recipe = _create_mx_recipe(mx_format, patterns)

    # Match_named_modules is the matcher apply_quantization_config itself uses,
    # so the scope is checked exactly, before the model is touched.
    targets = [cls.__name__ for cls in _MX_BLOCK_AXIS]
    total = [name for name, _ in match_named_modules(model, targets)]
    kept = [name for name, _ in match_named_modules(model, targets, patterns)]
    if not kept:
        raise ValueError(
            f"the {mx_format} recipe matched no {'/'.join(targets)} in "
            f"{type(model).__name__}")
    if ignore and len(kept) == len(total):
        raise ValueError(
            f"ignore={list(ignore)} excluded nothing: all {len(total)} module(s) "
            f"would be quantized. A pattern is a module name, covering that "
            f"module and everything under it, or 're:' plus a regex; names look "
            f"like {total[0]!r}")

    return _apply_recipe(model, recipe)
