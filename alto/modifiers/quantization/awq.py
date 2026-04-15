# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# AWQ: Activation-aware Weight Quantization (https://arxiv.org/abs/2306.00978)
#
# Extends QuantizationModifier with AWQ per-channel scale search.
# Only adds: ActivationScaleObservers + _process_block override.
# Sequential infrastructure lives in base.

from functools import partial
from typing import Any

import torch
import tqdm
from compressed_tensors.utils import getattr_chain, match_named_modules
from pydantic import PrivateAttr
from torch.nn import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from alto.observers.base import Observer
from alto.observers.activation_scale import ActivationScaleObserver
from .base import QuantizationModifier
from .calibration import calibrate_activations, update_weight_zp_scale
from .mapping import LayerMapping, resolve_mappings

__all__ = ["AWQModifier"]

AWQ_OBSERVER_BASE_NAME = "awq_act_scale"


def _awq_hook(module: Module, args: Any, base_name: str):
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name=base_name)


@torch.no_grad()
def _pseudo_quantize_tensor(w, n_bit=4, zero_point=True, q_group_size=-1):
    """AWQ round-to-nearest fake quantize (ported from awq/quantize/quantizer.py)."""
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0
    w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    return w.reshape(org_w_shape)


@torch.no_grad()
def _scale_ln_fcs(ln, fcs, scales):
    """Apply AWQ scales: ln.weight /= s, fc.weight *= s."""
    if not isinstance(fcs, list):
        fcs = [fcs]
    scales = scales.to(ln.weight.device).float()
    ln.weight.data = (ln.weight.data.float() / scales).to(ln.weight.dtype)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data = (ln.bias.data.float() / scales).to(ln.bias.dtype)
    for fc in fcs:
        fc.weight.data = (fc.weight.data.float() * scales.view(1, -1)).to(fc.weight.dtype)


class AWQModifier(QuantizationModifier):
    """Extends QuantizationModifier with AWQ per-channel smoothing.

    :param mappings: smooth/balance layer pattern pairs
    :param n_grid: grid search resolution (default 20)
    """

    mappings: list[dict] = []
    n_grid: int = 20
    sequential: bool = True

    _resolved_mappings: list[LayerMapping] = PrivateAttr(default_factory=list)
    _awq_observers: dict[Module, ActivationScaleObserver] = PrivateAttr(default_factory=dict)
    _awq_names: dict[Module, str] = PrivateAttr(default_factory=dict)
    _block_mappings: dict[int, list[LayerMapping]] = PrivateAttr(default_factory=dict)

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_initialize(model_parts, **kwargs)
        for m in model_parts:
            self._resolved_mappings.extend(resolve_mappings(m, self.mappings, self.ignore))
            for mapping in self._resolved_mappings:
                for bl, bn in zip(mapping.balance_layers, mapping.balance_names):
                    if bl in self._awq_observers:
                        continue
                    self._awq_names[bl] = bn
                    obs = Observer.create_instance(
                        "activation_scale", base_name=AWQ_OBSERVER_BASE_NAME,
                        args=None, module=bl, device=device_type,
                    )
                    object.__setattr__(bl, f"{AWQ_OBSERVER_BASE_NAME}_observer", obs)
                    self.register_hook(bl, partial(_awq_hook, base_name=AWQ_OBSERVER_BASE_NAME), "forward_pre")
                    self._awq_observers[bl] = obs
            for blk_idx, (blk_name, _) in enumerate(self._sequential_blocks):
                self._block_mappings[blk_idx] = [
                    mp for mp in self._resolved_mappings
                    if mp.smooth_name.startswith(blk_name + ".")
                ]
        logger.info(f"AWQModifier: {len(self._resolved_mappings)} mappings, {len(self._awq_observers)} observers")
        return result

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_pre_step(model_parts, **kwargs)
        if self.sequential:
            for obs in self._awq_observers.values():
                obs.disable()
        else:
            for obs in self._awq_observers.values():
                obs.enable()
        return result

    def _process_block(self, blk_idx, blk_name, block, block_inputs, block_mods):
        mappings = self._block_mappings.get(blk_idx, [])
        if not mappings:
            for mod in block_mods:
                update_weight_zp_scale(mod)
            return
        block_observers = []
        for mp in mappings:
            for bl in mp.balance_layers:
                if bl in self._awq_observers:
                    obs = self._awq_observers[bl]
                    obs.clear_stats()
                    obs.enable()
                    block_observers.append(obs)
        with torch.no_grad():
            for inp_args, inp_kwargs in block_inputs:
                block(*inp_args, **inp_kwargs)
        for obs in block_observers:
            obs.disable(clear=False)
        w_bit, q_group_size, zero_point = self._read_quant_config()
        q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
        for mp in mappings:
            self._search_and_apply_scales(mp, w_bit, q_config)

    def _nonsequential_post_step(self, model_parts):
        for obs in self._awq_observers.values():
            obs.disable(clear=False)
        w_bit, q_group_size, zero_point = self._read_quant_config()
        q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
        with torch.no_grad():
            for mapping in tqdm.tqdm(self._resolved_mappings, desc="AWQ scale search"):
                self._search_and_apply_scales(mapping, w_bit, q_config)
        self._cleanup_awq()
        for m in model_parts:
            for _, module in tqdm.tqdm(
                list(match_named_modules(m, self.resolved_targets, self.ignore)),
                desc="AWQ weight calibration",
            ):
                update_weight_zp_scale(module)
        return True

    def _post_sequential_cleanup(self, model_parts):
        self._cleanup_awq()
        for m in model_parts:
            for _, module in tqdm.tqdm(
                list(match_named_modules(m, self.resolved_targets, self.ignore)),
                desc="AWQ weight calibration",
            ):
                update_weight_zp_scale(module)

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        self._cleanup_awq()
        return super().on_finalize(model_parts, **kwargs)

    def _cleanup_awq(self):
        for module in self._awq_observers:
            attr = f"{AWQ_OBSERVER_BASE_NAME}_observer"
            if hasattr(module, attr):
                delattr(module, attr)
        self._awq_observers.clear()
        self._awq_names.clear()
        self._resolved_mappings.clear()
        self._block_mappings.clear()

    def _read_quant_config(self):
        for mapping in self._resolved_mappings:
            for bl in mapping.balance_layers:
                qargs = getattr_chain(bl, "quantization_scheme.weights", None)
                if qargs is not None:
                    return qargs.num_bits, (qargs.group_size or -1), (not qargs.symmetric)
        return 4, -1, False

    def _search_and_apply_scales(self, mapping, w_bit, q_config):
        first_bl = mapping.balance_layers[0]
        if first_bl not in self._awq_observers:
            return
        obs = self._awq_observers[first_bl]
        x_max = obs.stats
        if x_max is None or x_max.sum() == 0:
            logger.warning(f"No activation data for {mapping.smooth_name}, skipping")
            return
        x_max = x_max.clone()
        best_error, best_scales = float("inf"), None
        for grid_idx in range(self.n_grid):
            ratio = grid_idx / self.n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1.0
            scales[torch.isnan(scales)] = 1.0
            scales = scales.clamp(min=0.9, max=1.1)
            error = 0.0
            for bl in mapping.balance_layers:
                W = bl.weight.data.float()
                s = scales.view(1, -1).to(W.device)
                W_q = _pseudo_quantize_tensor(W * s, n_bit=w_bit, **q_config)
                err = ((W_q / s - W) * x_max.view(1, -1).to(W.device)).pow(2)
                error += err.sum().item()
            if error < best_error:
                best_error = error
                best_scales = scales.clone()
        if best_scales is None or torch.isnan(best_scales).any():
            logger.warning(f"AWQ search failed for {mapping.smooth_name}")
            return
        s = best_scales.to(mapping.smooth_layer.weight.device)
        logger.info(f"  AWQ {mapping.smooth_name}: scale min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}")
        _scale_ln_fcs(mapping.smooth_layer, mapping.balance_layers, s)
