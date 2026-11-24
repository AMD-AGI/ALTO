# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Flex attention with the varlen backend's LSE gradient.

``torch.nn.attention.varlen`` drops ``grad_lse`` in backward, so when an
``out_transform`` reads the LSE (GPT-OSS sinks) varlen trains only the sinks
through it. Flex backpropagates the LSE into dQ/dK. Detaching the LSE that
``out_transform`` sees makes Flex train like varlen.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.config.override import derive
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)


class _DetachedLse:
    # TorchTitan reshards inputs by the positional names in this signature.
    def forward(self, q_THK, k_THK, v_THV, *, out_transform=None, **kwargs):
        if out_transform is not None:
            transform = out_transform
            out_transform = lambda out, lse: transform(out, lse.detach())  # noqa: E731
        return super().forward(q_THK, k_THK, v_THV, out_transform=out_transform, **kwargs)


class DetachedLseFlexInnerAttention(_DetachedLse, FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass


class DetachedLseSlidingWindowFlexInnerAttention(_DetachedLse, SlidingWindowFlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(SlidingWindowFlexInnerAttention.Config):
        pass


_DETACHED_LSE = {
    FlexInnerAttention.Config: DetachedLseFlexInnerAttention.Config,
    SlidingWindowFlexInnerAttention.Config: DetachedLseSlidingWindowFlexInnerAttention.Config,
}


def detach_flex_lse(model_config) -> None:
    """Replace every Flex inner attention in ``model_config`` with its detached-LSE twin."""
    for _fqn, config, parent, attr in model_config.traverse(FlexInnerAttention.Config):
        replacement = derive(config, _DETACHED_LSE[type(config)])
        if isinstance(parent, list):
            parent[attr] = replacement
        else:
            setattr(parent, attr, replacement)
