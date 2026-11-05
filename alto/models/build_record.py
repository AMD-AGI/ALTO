# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Remember ``build_model_config`` and rebuild it with the final trainer settings.

Expert-parallel placement and RoPE length are fixed when a recipe builds its
model. CLI overrides arrive later, so ALTO calls the same function again with
the recorded flavor and attention backend and the final parallelism and
sequence length.

That second call returns a new config. Edits the recipe made after
``build_model_config`` (for example replacing ``inner_attention``) are copied
onto it. ``--model.*`` overrides are applied by the caller after this returns,
so they land on the rebuilt config.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
import inspect
from typing import Any

_BUILD_ARGS: dict[int, tuple[Any, dict[str, Any], Any]] = {}


class _Replace:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _Fields:
    __slots__ = ("updates",)

    def __init__(self, updates: dict[str, Any]) -> None:
        self.updates = updates


class _Items:
    __slots__ = ("updates", "ctor")

    def __init__(self, updates: list[Any], ctor: type) -> None:
        self.updates = updates
        self.ctor = ctor


class _Keys:
    __slots__ = ("updates",)

    def __init__(self, updates: dict[Any, Any]) -> None:
        self.updates = updates


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
        # Snapshot before the recipe mutates the returned config in place.
        _BUILD_ARGS[id(config)] = (
            original,
            dict(bound.arguments),
            copy.deepcopy(config),
        )
        return config

    build_model_config._alto_records_build = True  # type: ignore[attr-defined]
    flavors.build_model_config = build_model_config
    if getattr(package, "build_model_config", None) is original:
        package.build_model_config = build_model_config


def rebuild_recorded_model(config: Any) -> None:
    """Rebuild the recorded model and replay edits made after the first build."""
    from torchtitan.models.common.config_utils import decoder_vocab_size

    from alto.patches.patcher import ModelPatcher

    model = getattr(config, "model", None)
    recorded = None if model is None else _BUILD_ARGS.get(id(model))
    if recorded is None:
        return
    build, arguments, baseline = recorded
    # Root type stays the rebuilt config. Nested type changes (a recipe
    # swapping inner attention) are copied across.
    deltas = _config_delta(baseline, model, replace_type=False)
    arguments = dict(arguments)
    arguments["parallelism"] = config.parallelism
    arguments["seq_len"] = config.training.max_context_length
    rebuilt = build(**arguments)
    # Baseline for a later rebuild is this call's output, before recipe edits
    # are copied back on. Those edits are still on ``rebuilt`` and are found
    # again by diffing against this snapshot.
    pristine = copy.deepcopy(rebuilt)
    if deltas is not None:
        _apply_config_delta(rebuilt, deltas)
    new_id = id(rebuilt)
    _BUILD_ARGS[new_id] = (build, arguments, pristine)
    if new_id != id(model):
        _BUILD_ARGS.pop(id(model), None)
    config.model = rebuilt
    ModelPatcher.patch_model_config(config.model)
    loss_fn = getattr(getattr(config, "loss", None), "loss_fn", None)
    if loss_fn is not None and hasattr(loss_fn, "global_vocab_size"):
        loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    # The stock expert-parallel dispatcher synchronizes with the CPU.
    if config.parallelism.expert_parallel_degree > 1:
        config.training.disable_accelerator_graphs = True


def _is_rope_patch(baseline: Any, current: Any) -> bool:
    """ALTO rewrites Llama Complex RoPE to CosSin RoPE after the build.

    The class change is reapplied by ``ModelPatcher`` on the rebuilt config.
    Treating it as a wholesale replacement would keep the previous RoPE length.
    """
    from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE

    return type(baseline) is ComplexRoPE.Config and type(current) is CosSinRoPE.Config


def _values_equal(baseline: Any, current: Any) -> bool:
    # ``functools.partial`` copies compare unequal even when func/args match.
    if isinstance(baseline, functools.partial) and isinstance(current, functools.partial):
        return (
            baseline.func == current.func
            and baseline.args == current.args
            and (baseline.keywords or {}) == (current.keywords or {})
        )
    try:
        return bool(baseline == current)
    except Exception:
        return False


def _fields_delta(baseline: Any, current: Any) -> _Fields | None:
    updates: dict[str, Any] = {}
    for field in dataclasses.fields(baseline):
        if not hasattr(current, field.name):
            continue
        sub = _config_delta(getattr(baseline, field.name), getattr(current, field.name))
        if sub is not None:
            updates[field.name] = sub
    if not updates:
        return None
    return _Fields(updates)


def _config_delta(baseline: Any, current: Any, *, replace_type: bool = True) -> Any:
    if baseline is current:
        return None
    if _is_rope_patch(baseline, current):
        return _fields_delta(baseline, current)
    if type(baseline) is not type(current):
        if replace_type:
            return _Replace(current)
        if dataclasses.is_dataclass(baseline) and not isinstance(baseline, type):
            return _fields_delta(baseline, current)
        return _Replace(current)
    if dataclasses.is_dataclass(baseline) and not isinstance(baseline, type):
        return _fields_delta(baseline, current)
    if isinstance(baseline, (list, tuple)) and isinstance(current, type(baseline)):
        if len(baseline) != len(current):
            return _Replace(current)
        updates = [_config_delta(old, new) for old, new in zip(baseline, current)]
        if all(item is None for item in updates):
            return None
        return _Items(updates, type(current))
    if isinstance(baseline, dict) and isinstance(current, dict):
        if baseline.keys() != current.keys():
            return _Replace(current)
        updates = {}
        for key, value in baseline.items():
            sub = _config_delta(value, current[key])
            if sub is not None:
                updates[key] = sub
        if not updates:
            return None
        return _Keys(updates)
    if _values_equal(baseline, current):
        return None
    return _Replace(current)


def _apply_config_delta(node: Any, delta: Any) -> Any:
    if isinstance(delta, _Replace):
        return delta.value
    if isinstance(delta, _Fields):
        for name, sub in delta.updates.items():
            setattr(node, name, _apply_config_delta(getattr(node, name), sub))
        return node
    if isinstance(delta, _Items):
        return delta.ctor(
            node[index] if sub is None else _apply_config_delta(node[index], sub)
            for index, sub in enumerate(delta.updates)
        )
    if isinstance(delta, _Keys):
        for key, sub in delta.updates.items():
            node[key] = _apply_config_delta(node[key], sub)
        return node
    raise TypeError(f"unknown model-config delta {type(delta)!r}")
