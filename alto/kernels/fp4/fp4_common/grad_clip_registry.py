# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from .grad_clip_config import GradClipConfig

# Thread-local dict: id(module) -> GradClipConfig
_registry: dict[int, GradClipConfig] = {}


def register(module_id: int, cfg: GradClipConfig) -> None:
    _registry[module_id] = cfg


def deregister(module_id: int) -> None:
    _registry.pop(module_id, None)


def get(module_id: int | None) -> GradClipConfig | None:
    if module_id is None:
        return None
    return _registry.get(module_id)


def apply_clip(t: torch.Tensor, max_norm: float | None, clip_value: float | None) -> torch.Tensor:
    """Apply L2-norm clipping first (if set), then element-wise value clamp (if set).

    Returns t unchanged if both are None.
    """
    if max_norm is None and clip_value is None:
        return t
    before_absmax = t.abs().max().item()
    if max_norm is not None:
        norm = t.norm()
        if norm > max_norm:
            t = t * (max_norm / norm)
    if clip_value is not None:
        t = t.clamp(-clip_value, clip_value)
    after_absmax = t.abs().max().item()
    print(f"[grad_clip] apply_clip: before={before_absmax:.6f} after={after_absmax:.6f} max_norm={max_norm} clip_value={clip_value}", flush=True)
    return t
