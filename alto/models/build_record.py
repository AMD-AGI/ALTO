# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Remember the arguments of ``build_model_config`` and replay them.

Expert-parallel placement and RoPE length are fixed when a recipe builds its
model. CLI overrides arrive later, so ALTO calls the same function again with
the recorded flavor and attention backend and the final parallelism and
sequence length.
"""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Any

_BUILD_ARGS: dict[int, tuple[Any, dict[str, Any]]] = {}


def install_build_recorder(model_name: str) -> None:
    """Record ``build_model_config`` calls for one torchtitan model package."""
    package = importlib.import_module(f"torchtitan.models.{model_name}")
    flavors = importlib.import_module(f"torchtitan.models.{model_name}.flavors")
    original = getattr(flavors, "build_model_config", None)
    if original is None or getattr(original, "_alto_records_build", False):
        return

    signature = inspect.signature(original)

    @functools.wraps(original)
    def build_model_config(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        config = original(*args, **kwargs)
        _BUILD_ARGS[id(config)] = (original, dict(bound.arguments))
        return config

    build_model_config._alto_records_build = True  # type: ignore[attr-defined]
    flavors.build_model_config = build_model_config
    if getattr(package, "build_model_config", None) is original:
        package.build_model_config = build_model_config


def rebuild_recorded_model(config: Any) -> None:
    """Call the model's ``build_model_config`` with the final trainer settings."""
    from torchtitan.models.common.config_utils import decoder_vocab_size

    from alto.models.patcher import ModelPatcher

    model = getattr(config, "model", None)
    recorded = None if model is None else _BUILD_ARGS.get(id(model))
    if recorded is None:
        return
    build, arguments = recorded
    arguments = dict(arguments)
    arguments["parallelism"] = config.parallelism
    arguments["seq_len"] = config.training.max_context_length
    rebuilt = build(**arguments)
    _BUILD_ARGS[id(rebuilt)] = (build, arguments)
    _BUILD_ARGS.pop(id(model), None)
    config.model = rebuilt
    ModelPatcher.patch_model_config(config.model)
    loss_fn = getattr(getattr(config, "loss", None), "loss_fn", None)
    if loss_fn is not None and hasattr(loss_fn, "global_vocab_size"):
        loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    # The stock expert-parallel dispatcher synchronizes with the CPU.
    if config.parallelism.expert_parallel_degree > 1:
        config.training.disable_cuda_graphs = True
