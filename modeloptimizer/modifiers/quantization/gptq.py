# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af/src/llmcompressor/modifiers/quantization/gptq/
# licensed under the Apache License 2.0

from copy import copy
from functools import partial
from typing import Any

import torch
import tqdm
from compressed_tensors.quantization import (
    QuantizationStatus, QuantizationStrategy, fake_quantize,
)
from compressed_tensors.utils import getattr_chain, match_named_modules
from pydantic import PrivateAttr
from torch.nn import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from modeloptimizer.observers.base import Observer
from modeloptimizer.observers.hessian import PRECISION, HessianGPTQObserver
from .base import QuantizationModifier
from .calibration import calibrate_activations, update_weight_zp_scale

__all__ = ["GPTQModifier"]

HESSIAN_BASE_NAME = "gptq_hessian"


def _hessian_hook(module: Module, args: Any, base_name: str):
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name=base_name)


class GPTQModifier(QuantizationModifier):
    """Extends QuantizationModifier with GPTQ (https://arxiv.org/abs/2210.17323).

    :param block_size: columns per quantization block (default 128)
    :param dampening_frac: Hessian diagonal dampening (default 0.01)
    """

    block_size: int = 128
    dampening_frac: float = 0.01

    _hessian_observers: dict[Module, HessianGPTQObserver] = PrivateAttr(default_factory=dict)
    _hessian_names: dict[Module, str] = PrivateAttr(default_factory=dict)

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_initialize(model_parts, **kwargs)
        for m in model_parts:
            for name, module in match_named_modules(m, self.resolved_targets, self.ignore):
                if not getattr_chain(module, "quantization_scheme.weights", None):
                    continue
                if isinstance(module, torch.nn.Embedding):
                    continue
                self._hessian_names[module] = name
                obs = Observer.create_instance("hessian_gptq", base_name=HESSIAN_BASE_NAME,
                                               args=None, module=module, device=device_type)
                object.__setattr__(module, f"{HESSIAN_BASE_NAME}_observer", obs)
                self.register_hook(module, partial(_hessian_hook, base_name=HESSIAN_BASE_NAME), "forward_pre")
                self._hessian_observers[module] = obs
        logger.info(f"GPTQModifier: {len(self._hessian_observers)} modules")
        return result

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        result = super().on_pre_step(model_parts, **kwargs)
        for obs in self._hessian_observers.values():
            obs.enable()
        return result

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        for obs in self._hessian_observers.values():
            obs.disable(clear=False)
        for m in model_parts:
            from .mixin import QuantizationMixin
            QuantizationMixin.end_calibration(self, m)
            for _, module in tqdm.tqdm(
                list(match_named_modules(m, self.resolved_targets, self.ignore)),
                desc="GPTQ quantization",
            ):
                if module in self._hessian_observers:
                    self._gptq_quantize(module)
                else:
                    update_weight_zp_scale(module)
        self.remove_hooks()
        for module in self._hessian_observers:
            attr = f"{HESSIAN_BASE_NAME}_observer"
            if hasattr(module, attr):
                delattr(module, attr)
        self._hessian_observers.clear()
        self._hessian_names.clear()
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return super().on_finalize(model_parts, **kwargs)

    # ------------------------------------------------------------------

    def _gptq_quantize(self, module: Module):
        """Per-module GPTQ: RTN for scale/zp, then re-quantize with Hessian."""
        obs = self._hessian_observers[module]
        quant_args = getattr_chain(module, "quantization_scheme.weights")
        logger.info(f"Quantizing {self._hessian_names[module]} ({obs.num_samples} samples)")

        W_orig = module.weight.data.clone()
        update_weight_zp_scale(module)  # RTN: compute scale/zp + QDQ

        scale = module.weight_scale.data.clone().to(PRECISION)
        zp = module.weight_zero_point.data.clone() if hasattr(module, "weight_zero_point") else None
        gs = getattr(module, "weight_global_scale", None)

        # GPTQ re-quantize with Hessian error propagation
        loss, W_q = _gptq_block_quantize(
            W_orig.to(PRECISION), obs.stats / obs.num_samples,
            scale, zp, quant_args, gs, self.block_size, self.dampening_frac,
        )
        module.weight.data.copy_(W_q.to(module.weight.dtype))
        module.quantization_status = QuantizationStatus.COMPRESSED
        logger.info(f"  {self._hessian_names[module]}: loss={loss:.6f}")


# ======================================================================
# GPTQ block-wise quantization (OBQ Section 3.4)
# ======================================================================

def _gptq_block_quantize(W, H, scale, zp, quant_args, global_scale, blocksize, dampfrac):
    strategy = quant_args.strategy
    num_rows, num_cols = W.shape

    g_idx = None
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        g_idx = torch.arange(num_cols, device=W.device, dtype=torch.int) // quant_args.group_size

    col_args = copy(quant_args)
    if g_idx is not None:
        col_args.strategy = QuantizationStrategy.CHANNEL

    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = dampfrac * torch.mean(torch.diag(H))
    diag = torch.arange(num_cols, device=H.device)
    H[diag, diag] += damp
    try:
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)
    except torch._C._LinAlgError:
        logger.warning("Hessian inversion failed, falling back to RTN.")
        Hinv = torch.eye(num_cols, dtype=PRECISION, device=H.device)

    Q, losses = torch.zeros_like(W), torch.zeros(num_rows, device=W.device)

    for i1 in range(0, num_cols, blocksize):
        i2 = min(i1 + blocksize, num_cols)
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, Err1, L1 = torch.zeros_like(W1), torch.zeros_like(W1), torch.zeros_like(W1)

        for i in range(i2 - i1):
            w, d = W1[:, i], Hinv1[i, i]
            s, z = _col_scale(scale, zp, strategy, g_idx, i1 + i)
            q = fake_quantize(w, s, z, col_args, global_scale=global_scale)

            Q1[:, i], L1[:, i] = q, (w - q) ** 2 / d ** 2
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        losses += torch.sum(L1, 1) / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    return torch.sum(losses).item(), Q


def _col_scale(scale, zp, strategy, g_idx, col):
    if strategy == QuantizationStrategy.TENSOR:
        return scale, zp
    if strategy == QuantizationStrategy.CHANNEL:
        return (scale[:, 0] if scale.ndim > 1 else scale,
                zp[:, 0] if zp is not None and zp.ndim > 1 else zp)
    gi = g_idx[col]
    return scale[:, gi], (zp[:, gi] if zp is not None else None)
