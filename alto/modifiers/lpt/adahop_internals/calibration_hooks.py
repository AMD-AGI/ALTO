# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Pure-Python hook factories + JSON I/O for ``AdaHOPModifier``.

Split out of ``adahop.py`` so it can be unit-tested without dragging in
ALTO's triton-backed kernel imports.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict


def make_forward_callback(modifier: Any, fqn: str, detect: Callable) -> Callable:
    """Build a closure that writes ``(x, w)`` patterns for ``fqn`` into the
    modifier's current per-step bucket."""

    def _cb(x, w) -> None:
        if not modifier._per_step_patterns:
            return
        per_step = modifier._per_step_patterns[-1]
        per_layer = per_step.setdefault(fqn, {})
        per_layer["x"] = detect(x)
        per_layer["w"] = detect(w)

    return _cb


def make_forward_pre_hook(modifier: Any, fqn: str, detect: Callable) -> Callable:
    """Forward pre-hook on the nn.Linear module. Captures (x, weight._data) directly
    from the module — avoids the tensor-subclass dispatch path entirely, which is
    fragile under FSDP because all the rewrap paths (__torch_dispatch__,
    fsdp_post_all_gather, and any intervening detach/view) drop wrapper instance
    state. By the time __torch_function__("linear") fires the canonical wrapper
    is gone; the module reference is stable across all of that."""

    def _pre(module, args):
        if not modifier._per_step_patterns:
            return
        if not args:
            return
        x = args[0]
        weight = module.weight
        # Under FSDP weight is a DTensor wrapping our subclass; under no-FSDP
        # it's the subclass directly. Either way ._data lives on the subclass.
        w = weight.data
        # Unwrap DTensor if present.
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(w, DTensor):
                w = w._local_tensor
        except Exception:
            pass
        w_data = getattr(w, "_data", w)
        per_step = modifier._per_step_patterns[-1]
        per_layer = per_step.setdefault(fqn, {})
        per_layer["x"] = detect(x.detach())
        per_layer["w"] = detect(w_data.detach())

    return _pre


def make_backward_hook(modifier: Any, fqn: str, detect: Callable) -> Callable:
    """Build a backward hook that writes ``grad_output`` pattern for ``fqn``."""

    def _hook(_module, _grad_input, grad_output) -> None:
        if not modifier._per_step_patterns:
            return
        go = grad_output[0] if isinstance(grad_output, (list, tuple)) else grad_output
        if go is None:
            return
        per_step = modifier._per_step_patterns[-1]
        per_layer = per_step.setdefault(fqn, {})
        per_layer["grad_output"] = detect(go.detach())

    return _hook


def load_modes_from_json(path: str) -> Dict[str, Dict[str, str]]:
    """Load per-layer modes from JSON, tolerating both schemas:

    * Bare ``{fqn: {slot: mode}}`` — what users hand-author or extract.
    * Wrapped ``{"aggregated_patterns": {...}, "per_layer_modes": {...}}``
      — what :func:`write_modes_json` produces.
    """
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "per_layer_modes" in blob and isinstance(blob["per_layer_modes"], dict):
        return blob["per_layer_modes"]
    return blob


def write_modes_json(
    path: str,
    aggregated: Dict[str, Dict[str, str]],
    modes: Dict[str, Dict[str, str]],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {"aggregated_patterns": aggregated, "per_layer_modes": modes}
    with open(path, "w") as f:
        json.dump(blob, f, indent=2, sort_keys=True)
