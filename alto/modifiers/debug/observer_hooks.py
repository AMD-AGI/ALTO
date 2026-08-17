# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Pure-Python hook factories for DebugObserverModifier.

No triton imports — safe to run on a CPU-only box.

Each factory returns a closure that:
- Checks the per-fqn ``active`` gate before doing anything.
- Detaches and moves tensors to CPU before storing them.
- Writes into ``captures[fqn][step_idx][key]``.

The captures dict has the shape::

    {
        fqn: {
            "active": bool,
            step_idx: {
                "input": Tensor,
                "weight": Tensor,
                "grad_output": Tensor,
                "grad_weight": Tensor,       # nn.Linear
                "grad_mlp1_weight": Tensor,  # GptOssGroupedExperts
                "grad_mlp2_weight": Tensor,  # GptOssGroupedExperts
            },
        }
    }
"""

from __future__ import annotations

from typing import Any


def _unwrap_weight(weight: Any) -> Any:
    """Strip DTensor and TrainingWeightWrapperTensor wrappers to reach the
    underlying storage tensor, mirroring the pattern in
    alto/modifiers/lpt/adahop_internals/calibration_hooks.py."""
    w = weight.data
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(w, DTensor):
            w = w._local_tensor
    except Exception:
        pass
    # TrainingWeightWrapperTensor stores the raw data in ._data
    w = getattr(w, "_data", w)
    return w


def _store(captures: dict, fqn: str, step_idx: int, key: str, tensor: Any) -> None:
    """Detach, move to CPU, and store tensor in captures."""
    captures[fqn].setdefault(step_idx, {})[key] = tensor.detach().cpu()


# ---------------------------------------------------------------------------
# nn.Linear hooks
# ---------------------------------------------------------------------------

def make_linear_fwd_pre_hook(
    captures: dict,
    fqn: str,
    step_ref: list,
    capture_input: bool,
    capture_weight: bool,
):
    """Forward pre-hook for nn.Linear.

    Captures the raw (pre-quant) activation and unwrapped weight storage.
    step_ref is a one-element list so the closure always sees the current step.
    """

    def _hook(module, args):
        if not captures[fqn]["active"]:
            return
        step_idx = step_ref[0]
        if capture_input and args:
            _store(captures, fqn, step_idx, "input", args[0])
        if capture_weight:
            try:
                w = _unwrap_weight(module.weight)
                _store(captures, fqn, step_idx, "weight", w)
            except Exception:
                pass

    return _hook


def make_linear_bwd_hook(
    captures: dict, fqn: str, step_ref: list, capture_grad_output: bool, capture_grad_weight: bool = False
):
    """Full-backward hook for nn.Linear.

    grad_input = (grad_x, grad_weight, grad_bias) — exactly what
    MXFP4LinearFunction.backward returns, so grad_weight here is the
    post-clip value, not the value accumulated into param.grad later.
    grad_output = upstream gradient flowing into this layer's output.
    """

    def _hook(module, grad_input, grad_output):
        if not captures[fqn]["active"]:
            return
        step_idx = step_ref[0]
        if capture_grad_output:
            go = grad_output[0] if isinstance(grad_output, (list, tuple)) else grad_output
            if go is not None:
                _store(captures, fqn, step_idx, "grad_output", go)
        if capture_grad_weight and isinstance(grad_input, (list, tuple)) and len(grad_input) > 1:
            gw = grad_input[1]
            if gw is not None:
                _store(captures, fqn, step_idx, "grad_weight", gw)

    return _hook


# ---------------------------------------------------------------------------
# GptOssGroupedExperts hooks
# ---------------------------------------------------------------------------

def make_grouped_experts_fwd_pre_hook(
    captures: dict,
    fqn: str,
    step_ref: list,
    capture_input: bool,
    capture_weight: bool,
):
    """Forward pre-hook for GptOssGroupedExperts.

    Captures the activation entering mlp1 (args[0]) and the mlp1_weight storage.
    The mlp2 activation is internal to _run_experts_grouped_mm and cannot be
    captured at the module-hook level (v1 limitation, noted in _meta).
    """

    def _hook(module, args):
        if not captures[fqn]["active"]:
            return
        step_idx = step_ref[0]
        if capture_input and args:
            _store(captures, fqn, step_idx, "input", args[0])
        if capture_weight:
            for attr in ("mlp1_weight", "mlp2_weight"):
                param = getattr(module, attr, None)
                if param is not None:
                    try:
                        w = _unwrap_weight(param)
                        _store(captures, fqn, step_idx, attr, w)
                    except Exception:
                        pass

    return _hook


def make_grouped_experts_bwd_hook(
    captures: dict, fqn: str, step_ref: list, capture_grad_output: bool
):
    """Full-backward hook for GptOssGroupedExperts."""

    def _hook(module, grad_input, grad_output):
        if not captures[fqn]["active"]:
            return
        if not capture_grad_output:
            return
        go = grad_output[0] if isinstance(grad_output, (list, tuple)) else grad_output
        if go is None:
            return
        _store(captures, fqn, step_ref[0], "grad_output", go)

    return _hook


def make_grouped_experts_grad_weight_hook(
    captures: dict,
    fqn: str,
    step_ref: list,
    key: str,
    capture_grad_weight: bool,
):
    """Parameter grad hook for mlp1_weight or mlp2_weight of GptOssGroupedExperts.

    key should be "grad_mlp1_weight" or "grad_mlp2_weight".
    """

    def _hook(grad):
        if not captures[fqn]["active"]:
            return
        if not capture_grad_weight:
            return
        _store(captures, fqn, step_ref[0], key, grad)

    return _hook
