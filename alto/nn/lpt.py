# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Low-precision Linear and GroupedLinear modules.

These modules own a :class:`TrainingOpConfig` directly. Their parameters stay
plain tensors, so TorchTitan's SPMD and FSDP sharding can run without a tensor
subclass. ``training_op_config is None`` means the module was installed before
the LPT recipe was applied and is not ready to build.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields

import spmd_types as spmd
import torch

from torchtitan.config.override import derive
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.linear import GroupedLinear, Linear

from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from alto.kernels.fp4.amdfp4 import (
    _quantize_then_amdfp4_scaled_grouped_mm,
    _to_amdfp4_then_scaled_mm,
)
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.functional import (
    _quantize_then_mxfp_scaled_grouped_mm,
)
from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.functional import (
    _quantize_then_nvfp4_scaled_grouped_mm,
)
from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_scaled_mm
from alto.kernels.mxfp8.mxfp8_grouped_gemm import _quantize_then_mxfp8_scaled_grouped_mm
from alto.kernels.mxfp8.mxfp8_linear import _to_mxfp8_then_scaled_mm

MatchFn = Callable[[str, type, str | list[str]], bool]


def require_training_op_config(config) -> TrainingOpConfig:
    """Return the recipe config, or fail while the module is still uninitialized."""
    training_op_config = getattr(config, "training_op_config", None)
    if training_op_config is None:
        raise RuntimeError(
            f"{type(config).__qualname__} is not fully initialized: "
            "training_op_config is unset until the LPT recipe is applied."
        )
    return training_op_config


class LPTModule:
    """Shared state for a low-precision module generated from an upstream class.

    ``Config.training_op_config`` is the recipe state. ``None`` means the recipe
    has not been applied yet. De-oscillation discovers eligible weights through
    :meth:`iter_lpt_weights`.

    ``Config`` is not a slots dataclass. TorchTitan configs already are, and two
    slots dataclasses cannot be combined as bases. Each generated config is a
    slots dataclass that inherits this one.
    """

    @dataclass(kw_only=True)
    class Config:
        training_op_config: TrainingOpConfig | None = None

    training_op_config: TrainingOpConfig
    lpt_qdq_axis: int = -1

    def __init__(self, config: Config) -> None:
        self.training_op_config = require_training_op_config(config)

    def iter_lpt_weights(
        self,
    ) -> Iterator[tuple[torch.nn.Parameter, TrainingOpConfig, int]]:
        """Yield ``(parameter, training_op_config, qdq_axis)``."""
        yield self.weight, self.training_op_config, self.lpt_qdq_axis


_LPT_CLASSES: dict[type, type] = {}


def lpt_module(base: type) -> type:
    """Return an LPT subclass of ``base`` that keeps the parent's forward.

    ``GroupedLinear`` subclasses override ``_grouped_mm``. ``Linear`` subclasses
    override ``_linear``. ``ScaledDotProductInnerAttention`` overrides
    ``attention_BLHK``. Column/row collectives and ``GptOssGroupedLinear``'s
    per-expert bias stay on the parent class, so those types do not need a
    handwritten LPT module.
    """
    cached = _LPT_CLASSES.get(base)
    if cached is not None:
        return cached
    if issubclass(base, GroupedLinear):
        kind = "grouped"
    elif issubclass(base, Linear):
        kind = "linear"
    elif issubclass(base, ScaledDotProductInnerAttention):
        kind = "sdpa"
    else:
        raise TypeError(
            f"{base.__qualname__} is not a Linear, GroupedLinear, or "
            "ScaledDotProductInnerAttention, so it has no LPT compute method "
            "to override."
        )

    name = f"LPT{base.__name__}"

    @dataclass(kw_only=True, slots=True)
    class Config(base.Config, LPTModule.Config):
        pass

    Config.__qualname__ = f"{name}.Config"

    def __init__(self, config: Config) -> None:
        base.__init__(self, config)
        LPTModule.__init__(self, config)
        if kind == "sdpa" and self.training_op_config.precision != "mxfp4":
            raise ValueError(
                f"Unsupported SDPA precision: {self.training_op_config.precision}"
            )

    def extra_repr(self) -> str:
        parent = base.extra_repr(self)
        config_repr = f"training_op_config={self.training_op_config}"
        return f"{parent}, {config_repr}" if parent else config_repr

    namespace: dict[str, object] = {
        "Config": Config,
        "__init__": __init__,
        "extra_repr": extra_repr,
        "__module__": __name__,
    }
    if kind == "linear":
        def _linear(self, input, weight, bias):
            return lpt_linear(input, weight, bias, self.training_op_config)

        namespace["_linear"] = _linear
    elif kind == "grouped":
        def _grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
            return lpt_grouped_mm(
                input_RI, weight_EOI, offsets_E, self.training_op_config
            )

        namespace["_grouped_mm"] = _grouped_mm
    else:
        def attention_BLHK(
            self, q_BLHK, k_BLHK, v_BLHV, *, scale, enable_gqa, is_causal,
        ):
            return lpt_attention_BLHK(
                q_BLHK, k_BLHK, v_BLHV,
                scale=scale, enable_gqa=enable_gqa, is_causal=is_causal,
            )

        def iter_lpt_weights(self):
            yield from ()

        namespace["attention_BLHK"] = attention_BLHK
        namespace["iter_lpt_weights"] = iter_lpt_weights

    generated = type(name, (base, LPTModule), namespace)
    generated.__qualname__ = name
    _LPT_CLASSES[base] = generated
    return generated


def _annotate_like(output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if spmd.is_type_checking() and spmd.has_local_type(source):
        spmd.assert_local_type_like(output, source)
    return output


def lpt_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    config: TrainingOpConfig,
) -> torch.Tensor:
    """Run one low-precision linear on the stored weight and bias."""
    weight_2d, bias_1d = Linear._flatten_weight_and_bias(weight, bias)
    with spmd.no_typecheck():
        if config.precision == "mxfp4":
            y = _to_mxfp4_then_scaled_mm(
                x,
                weight_2d,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_dge=config.use_dge,
                clip_mode=config.clip_mode,
                use_hadamard=config.use_hadamard,
                use_macro_block_scaling=config.two_level_scaling == "blockwise",
                use_uos=config.use_uos,
            )
        elif config.precision == "nvfp4":
            y = _to_nvfp4_then_scaled_mm(
                x,
                weight_2d,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
            )
        elif config.precision == "amdfp4":
            y = _to_amdfp4_then_scaled_mm(
                x,
                weight_2d,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
            )
        elif config.precision in ("mxfp8_e4m3", "mxfp8_e5m2"):
            y = _to_mxfp8_then_scaled_mm(
                x,
                weight_2d,
                fp8_variant="e4m3" if config.precision == "mxfp8_e4m3" else "e5m2",
                use_sr_grad=config.use_sr_grad,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
            )
        else:
            raise ValueError(f"Unsupported LPT linear precision: {config.precision}")
        if bias_1d is not None:
            y = y + bias_1d
    return _annotate_like(y, x)


def lpt_grouped_mm(
    input_RI: torch.Tensor,
    weight_EOI: torch.Tensor,
    offsets_E: torch.Tensor,
    config: TrainingOpConfig,
) -> torch.Tensor:
    """Run grouped GEMM on the stored ``[E, O, I]`` weight."""
    # Dispatch kernels consume the same transposed layout as torch._grouped_mm.
    weight = weight_EOI.bfloat16().transpose(-2, -1)
    with spmd.no_typecheck():
        if config.precision == "mxfp4":
            y = _quantize_then_mxfp_scaled_grouped_mm(
                input_RI,
                weight,
                offs=offsets_E,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_dge=config.use_dge,
                use_hadamard=config.use_hadamard,
                clip_mode=config.clip_mode,
                use_macro_block_scaling=config.two_level_scaling == "blockwise",
                use_uos=config.use_uos,
            )
        elif config.precision == "nvfp4":
            y = _quantize_then_nvfp4_scaled_grouped_mm(
                input_RI,
                weight,
                offs=offsets_E,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
            )
        elif config.precision == "amdfp4":
            y = _quantize_then_amdfp4_scaled_grouped_mm(
                input_RI,
                weight,
                offs=offsets_E,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
            )
        elif config.precision == "mxfp8_e4m3":
            y = _quantize_then_mxfp8_scaled_grouped_mm(
                input_RI,
                weight,
                offs=offsets_E,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
            )
        else:
            raise ValueError(f"Unsupported LPT grouped precision: {config.precision}")
    return _annotate_like(y, input_RI)


def lpt_attention_BLHK(
    q_BLHK: torch.Tensor,
    k_BLHK: torch.Tensor,
    v_BLHV: torch.Tensor,
    *,
    scale: float | None,
    enable_gqa: bool,
    is_causal: bool,
) -> torch.Tensor:
    """Run MXFP4 attention on ``[B, L, H, D]`` inputs."""
    del enable_gqa
    q, k, v = (
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHV.transpose(1, 2),
    )
    _batch, _num_head_q, seqlen_q, head_dim_qk = q.shape
    _batch_k, num_head_kv, seqlen_kv, head_dim_qk_k = k.shape
    _batch_v, num_head_kv_v, seqlen_kv_v, _head_dim_v = v.shape
    if num_head_kv != num_head_kv_v or head_dim_qk != head_dim_qk_k or seqlen_kv != seqlen_kv_v:
        raise ValueError(
            "SDPA key and value heads do not match: "
            f"k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    sm_scale = head_dim_qk ** -0.5 if scale is None else scale
    output = triton_attention_mxfp4(
        q,
        k,
        v,
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=seqlen_q,
        max_seqlens_k=seqlen_kv,
        causal=is_causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]
    return output.transpose(1, 2)


def _assign(parent, attr, value) -> None:
    if isinstance(parent, list):
        parent[attr] = value
    else:
        setattr(parent, attr, value)


def _replacement_for(config, training_op_config: TrainingOpConfig):
    if isinstance(config, LPTModule.Config):
        if config.training_op_config is None:
            config.training_op_config = training_op_config
            if hasattr(config, "inplace_wgrad_accum"):
                config.inplace_wgrad_accum = False
        elif config.training_op_config != training_op_config:
            raise ValueError(
                f"{type(config).__qualname__} already has a different "
                "training_op_config."
            )
        return None
    replacement_cls = lpt_module(config._owner).Config
    deltas: dict[str, object] = {"training_op_config": training_op_config}
    if "inplace_wgrad_accum" in {field.name for field in fields(replacement_cls)}:
        deltas["inplace_wgrad_accum"] = False
    return derive(config, replacement_cls, **deltas)


def apply_lpt_module_configs(
    model_config,
    schemes: list[tuple[TrainingOpConfig, list[str]]],
    is_match: MatchFn,
) -> None:
    """Replace matched linear configs with initialized LPT configs.

    An LPT config may already exist with ``training_op_config is None``. This
    fills that field from the recipe. Any LPT config that is still empty after
    the recipe is applied is rejected.
    """
    source_classes: tuple[type, ...] = (
        Linear.Config,
        GroupedLinear.Config,
        ScaledDotProductInnerAttention.Config,
    )
    matches: dict[str, tuple[object, object, str | int, TrainingOpConfig]] = {}
    for training_op_config, targets in schemes:
        for source_cls in source_classes:
            for fqn, config, parent, attr in model_config.traverse(source_cls):
                if parent is None or not is_match(fqn, config._owner, targets):
                    continue
                previous = matches.get(fqn)
                if previous is not None and previous[3] != training_op_config:
                    raise ValueError(
                        f"{fqn} matches more than one LPT scheme."
                    )
                matches[fqn] = (config, parent, attr, training_op_config)

    for fqn, (config, parent, attr, training_op_config) in matches.items():
        try:
            replacement = _replacement_for(config, training_op_config)
        except ValueError as error:
            raise ValueError(f"{fqn}: {error}") from error
        if replacement is not None:
            _assign(parent, attr, replacement)

    for fqn, config, _parent, _attr in model_config.traverse(LPTModule.Config):
        if config.training_op_config is None:
            raise RuntimeError(
                f"{fqn or type(config).__qualname__} is not fully "
                "initialized: the LPT recipe did not apply "
                "training_op_config."
            )


__all__ = [
    "LPTModule",
    "apply_lpt_module_configs",
    "lpt_module",
    "require_training_op_config",
]
