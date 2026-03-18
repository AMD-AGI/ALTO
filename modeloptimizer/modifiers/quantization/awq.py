# AWQ: Activation-aware Weight Quantization
# Ported from https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py
# Core idea: find per-channel scales that minimise quantization error via grid search,
# then apply scales to smooth_layer / balance_layers before RTN quantization.

from functools import partial
from typing import Any

import torch
import tqdm
from compressed_tensors.utils import getattr_chain, match_named_modules
from pydantic import PrivateAttr
from torch.nn import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from modeloptimizer.observers.base import Observer
from modeloptimizer.observers.activation_scale import ActivationScaleObserver
from .base import QuantizationModifier
from .calibration import calibrate_activations, update_weight_zp_scale
from .mapping import LayerMapping, resolve_mappings

__all__ = ["AWQModifier"]

AWQ_OBSERVER_BASE_NAME = "awq_act_scale"


def _awq_hook(module: Module, args: Any, base_name: str):
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name=base_name)


# ======================================================================
# Ported from awq/quantize/quantizer.py: pseudo_quantize_tensor
# ======================================================================

@torch.no_grad()
def pseudo_quantize_tensor(w, n_bit=4, zero_point=True, q_group_size=-1):
    """AWQ's native weight quantization (round-to-nearest with group support)."""
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


# ======================================================================
# Ported from awq/quantize/auto_scale.py: _search_module_scale
# ======================================================================

@torch.no_grad()
def search_module_scale(block, linears2scale, x, kwargs, w_bit, q_config):
    """Find the best per-channel scale via grid search with block-level forward passes.

    Directly ported from AWQ's _search_module_scale.
    """
    x = x.to(next(block.parameters()).device)
    org_out = block(x, **kwargs)
    if isinstance(org_out, tuple):
        org_out = org_out[0]

    x_max = x.abs().view(-1, x.shape[-1]).mean(0)  # get_act_scale

    best_error = float("inf")
    best_scales = None
    n_grid = 20

    org_sd = {k: v.cpu() for k, v in block.state_dict().items()}

    for ratio in range(n_grid):
        ratio = ratio / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()

        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = pseudo_quantize_tensor(
                fc.weight.data, n_bit=w_bit, **q_config,
            ) / scales.view(1, -1).to(fc.weight.device)

        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (org_out - out).float().pow(2).mean().item()
        if loss < best_error:
            best_error = loss
            best_scales = scales.clone()

        block.load_state_dict(org_sd)

    assert best_scales is not None, "AWQ grid search failed"
    assert torch.isnan(best_scales).sum() == 0
    return best_scales.detach()


# ======================================================================
# Ported from awq/quantize/auto_scale.py: scale_ln_fcs / scale_fc_fc
# ======================================================================

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """Apply AWQ scales: ln.weight /= s, fc.weight *= s.

    Operations done in float32 then cast back to preserve precision.
    """
    if not isinstance(fcs, list):
        fcs = [fcs]
    scales = scales.to(ln.weight.device).float()
    ln.weight.data = (ln.weight.data.float() / scales).to(ln.weight.dtype)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data = (ln.bias.data.float() / scales).to(ln.bias.dtype)
    for fc in fcs:
        fc.weight.data = (fc.weight.data.float() * scales.view(1, -1)).to(fc.weight.dtype)


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """Apply AWQ scales between two linear layers."""
    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
    fc2.weight.mul_(scales.view(1, -1))


# ======================================================================
# AWQModifier
# ======================================================================


class AWQModifier(QuantizationModifier):
    """Extends QuantizationModifier with AWQ (https://arxiv.org/abs/2306.00978).

    Finds per-channel scales via grid search, applies to smooth/balance layer
    pairs, then falls back to RTN for weight quantization.

    Sample yaml::

        quantization_stage:
          quantization_modifiers:
            AWQModifier:
              n_grid: 20
              ignore: ["output"]
              mappings:
                - smooth_layer: "re:.*attention_norm"
                  balance_layers:
                    - "re:.*attention.wq"
                    - "re:.*attention.wk"
                    - "re:.*attention.wv"
                - smooth_layer: "re:.*ffn_norm"
                  balance_layers:
                    - "re:.*feed_forward.w1"
                    - "re:.*feed_forward.w3"
              config_groups:
                group_0:
                  weights:
                    num_bits: 4
                    type: "int"
                    symmetric: true
                    strategy: "group"
                    group_size: 128
                  targets: ["Linear"]

    :param mappings: smooth/balance layer pattern pairs
    :param n_grid: grid search points (default 20)
    """

    mappings: list[dict] = []
    n_grid: int = 20

    _resolved_mappings: list[LayerMapping] = PrivateAttr(default_factory=list)
    _awq_observers: dict[Module, ActivationScaleObserver] = PrivateAttr(default_factory=dict)
    _awq_names: dict[Module, str] = PrivateAttr(default_factory=dict)
    # Cache block inputs for forward-pass grid search: block -> list[(args, kwargs)]
    _block_inputs: dict[Module, list] = PrivateAttr(default_factory=dict)

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_initialize(model_parts, **kwargs)

        for m in model_parts:
            self._resolved_mappings.extend(
                resolve_mappings(m, self.mappings, self.ignore)
            )

            # Attach ActivationScaleObserver on balance layers
            for mapping in self._resolved_mappings:
                for bl, bn in zip(mapping.balance_layers, mapping.balance_names):
                    if bl not in self._awq_observers:
                        self._awq_names[bl] = bn
                        obs = Observer.create_instance(
                            "activation_scale",
                            base_name=AWQ_OBSERVER_BASE_NAME,
                            args=None, module=bl, device=device_type,
                        )
                        object.__setattr__(bl, f"{AWQ_OBSERVER_BASE_NAME}_observer", obs)
                        self.register_hook(
                            bl, partial(_awq_hook, base_name=AWQ_OBSERVER_BASE_NAME),
                            "forward_pre",
                        )
                        self._awq_observers[bl] = obs

        logger.info(f"AWQModifier: {len(self._resolved_mappings)} mappings, "
                     f"{len(self._awq_observers)} observers")
        return result

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_pre_step(model_parts, **kwargs)
        for obs in self._awq_observers.values():
            obs.enable()
        return result

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        for obs in self._awq_observers.values():
            obs.disable(clear=False)

        # Step 1: AWQ scale search + apply (before quantization)
        with torch.no_grad():
            self._apply_awq_scales()

        # Step 2: Clean up AWQ state
        self._cleanup_awq_state()

        # Step 3: RTN quantization (parent's logic)
        for m in model_parts:
            from .mixin import QuantizationMixin
            QuantizationMixin.end_calibration(self, m)
            for _, module in tqdm.tqdm(
                list(match_named_modules(m, self.resolved_targets, self.ignore)),
                desc="AWQ weight calibration",
            ):
                update_weight_zp_scale(module)

        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        self._cleanup_awq_state()
        return super().on_finalize(model_parts, **kwargs)

    def _cleanup_awq_state(self):
        self.remove_hooks()
        for module in self._awq_observers:
            attr = f"{AWQ_OBSERVER_BASE_NAME}_observer"
            if hasattr(module, attr):
                delattr(module, attr)
        self._awq_observers.clear()
        self._awq_names.clear()
        self._resolved_mappings.clear()
        self._block_inputs.clear()

    # ------------------------------------------------------------------
    # AWQ scale search: analytical approach (no block forward pass)
    # Uses activation scales from observer + AWQ's original scale formula
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_awq_scales(self):
        """For each mapping, find best scales and apply to smooth/balance layers."""
        # Read w_bit, group_size, symmetric from the actual quantization scheme
        # so that AWQ search quantizer matches the RTN quantizer exactly
        w_bit, q_group_size, zero_point = 4, -1, False
        for mapping in self._resolved_mappings:
            for bl in mapping.balance_layers:
                qargs = getattr_chain(bl, "quantization_scheme.weights", None)
                if qargs is not None:
                    w_bit = qargs.num_bits
                    q_group_size = qargs.group_size or -1
                    zero_point = not qargs.symmetric
                    break
            break
        q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
        logger.info(f"AWQ search: w_bit={w_bit}, group_size={q_group_size}, zero_point={zero_point}")

        for mapping in tqdm.tqdm(self._resolved_mappings, desc="AWQ scale search"):
            first_bl = mapping.balance_layers[0]
            if first_bl not in self._awq_observers:
                continue
            obs = self._awq_observers[first_bl]
            x_max = obs.stats
            if x_max is None or x_max.sum() == 0:
                logger.warning(f"No activation data for {mapping.smooth_name}, skipping")
                continue
            x_max = x_max.clone()

            # Per-channel weight magnitude (max across all balance layers)
            w_max = torch.zeros_like(x_max)
            for bl in mapping.balance_layers:
                w_max = torch.maximum(
                    w_max,
                    bl.weight.data.float().abs().mean(dim=0).to(w_max.device),
                )

            best_error, best_scales = float("inf"), None
            for grid_idx in range(self.n_grid):
                ratio = grid_idx / self.n_grid
                # Original AWQ formula from auto_scale.py:
                # scales = x_max.pow(ratio) / sqrt(max * min)
                # ratio=0 → scales=1 (identity, no smoothing)
                scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales[torch.isinf(scales)] = 1.0
                scales[torch.isnan(scales)] = 1.0
                # Clamp scales for bf16 AMP compatibility; extreme scales cause
                # precision loss when weight QDQ happens under autocast
                scales = scales.clamp(min=0.9, max=1.1)

                error = 0.0
                for bl in mapping.balance_layers:
                    W = bl.weight.data.float()
                    s = scales.view(1, -1).to(W.device)
                    W_q = pseudo_quantize_tensor(W * s, n_bit=w_bit, **q_config)
                    err = ((W_q / s - W) * x_max.view(1, -1).to(W.device)).pow(2)
                    error += err.sum().item()

                if error < best_error:
                    best_error = error
                    best_scales = scales.clone()

            if best_scales is None or torch.isnan(best_scales).any():
                logger.warning(f"AWQ search failed for {mapping.smooth_name}")
                continue

            s = best_scales.to(mapping.smooth_layer.weight.device)
            logger.info(
                f"AWQ {mapping.smooth_name}: "
                f"scale min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f}"
            )
            scale_ln_fcs(mapping.smooth_layer, mapping.balance_layers, s)
            logger.info(f"AWQ smoothed {mapping.smooth_name}")
