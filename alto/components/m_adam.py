# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""``m_adam``: a hybrid additive/multiplicative optimizer and its
torchtitan :class:`OptimizersContainer` wiring.

``m_adam`` decomposes every weight via ``torch.frexp`` into a mantissa and
an exponent (``w = m * 2**e``) and applies two coupled updates each step:

* an **AdamW** (additive) update on the mantissa, and
* a **Madam-style** (multiplicative, RMSProp-normalized) update on the
  exponent -- i.e. gradient descent on the log2-magnitude.

Net effect (with ``weight_decay_e == 0``)::

    w_new  ~=  w * 2**(delta_e)   +   delta_w_AdamW
               \\_ multiplicative (Madam)   \\_ additive (AdamW)

This explicit mantissa/exponent split lets the exponent (dynamic range)
and mantissa (precision) be controlled with independent learning rates,
weight decays, and -- for the exponent -- an independent schedule, which
is why it is a natural fit for low-precision / quantization-aware training.

The :class:`MAdamOptimizersContainer` adapts ``m_adam`` to torchtitan's
``config.optimizer.build(...)`` path so it can be selected from a model's
``config_registry`` just like the built-in ``Adam``/``AdamW``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.logging import logger

__all__ = [
    "m_adam",
    "MAdamOptimizersContainer",
]


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


def _sched_value(
    mode: str,
    *,
    base_lr: float,
    t: int,
    total_steps: Optional[int],
    warmup_steps: int,
    min_lr_ratio: float,
    logcosine_alpha: float,
) -> float:
    if warmup_steps > 0 and t < warmup_steps:
        return base_lr * (float(t) / float(max(1, warmup_steps)))
    if total_steps is None or mode == "constant":
        return base_lr

    T = max(1, total_steps - max(0, warmup_steps))
    p = min(1.0, max(0.0, (t - warmup_steps) / T))
    rmin = float(min_lr_ratio)

    if mode == "linear":
        return base_lr * (rmin + (1.0 - rmin) * (1.0 - p))
    if mode == "cosine":
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return base_lr * (rmin + (1.0 - rmin) * c)
    if mode == "logcosine":
        a = float(logcosine_alpha)
        g = math.exp(-a * (1.0 - math.cos(math.pi * p)))
        g1 = math.exp(-2.0 * a)
        s = (g - g1) / (1.0 - g1 + 1e-12)
        return base_lr * (rmin + (1.0 - rmin) * s)

    return base_lr


class m_adam(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr_m: float = 1e-3,
        lr_e: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        p_scale: float = 3.0,
        g_bound: float = 20.0,
        *,
        weight_decay_m: float = 0.0,
        weight_decay_e: float = 0.0,
        abs_clamp: bool = False,
        abs_clamp_floor: float = 0.0,
        clip_e_final: bool = False,
        e_final_min: float = -60.0,
        e_final_max: float = 60.0,
        use_de_step_cap: bool = True,
        de_step_cap: float = 0.5,
        tie_e_to_m: bool = False,
        sched_e: str = "constant",
        total_steps_e: Optional[int] = None,
        warmup_steps_e: int = 0,
        min_lr_ratio_e: float = 0.0,
        logcosine_alpha_e: float = 6.0,
        track_dw_rms: bool = True,
    ):
        if not (0.0 <= lr_m and 0.0 <= lr_e):
            raise ValueError("Learning rates must be non-negative.")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0,1).")

        ratio_e_init = float(lr_e) / max(1e-20, float(lr_m))

        defaults = dict(
            lr=lr_m,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            p_scale=p_scale,
            g_bound=g_bound,
            weight_decay_m=weight_decay_m,
            weight_decay_e=weight_decay_e,
            abs_clamp=abs_clamp,
            abs_clamp_floor=abs_clamp_floor,
            clip_e_final=clip_e_final,
            e_final_min=e_final_min,
            e_final_max=e_final_max,
            use_de_step_cap=use_de_step_cap,
            de_step_cap=de_step_cap,
            tie_e_to_m=tie_e_to_m,
            ratio_e_init=ratio_e_init,
            sched_e=sched_e,
            total_steps_e=total_steps_e,
            warmup_steps_e=warmup_steps_e,
            min_lr_ratio_e=min_lr_ratio_e,
            lr_e_base=float(lr_e),
            logcosine_alpha_e=float(logcosine_alpha_e),
            t=0,
            last_lr_e=float(lr_e),
            last_lr_m=float(lr_m),
        )
        super().__init__(params, defaults)

        # Per-step diagnostic accumulators for the exponent- vs mantissa-branch
        # weight change. Populated during step() and consumed/logged by the
        # MAdamOptimizersContainer. Kept off the param_groups so they never
        # enter the optimizer state_dict / checkpoint schema.
        self.track_dw_rms = bool(track_dw_rms)
        self._dw_exp_sumsq: Optional[torch.Tensor] = None
        self._dw_man_sumsq: Optional[torch.Tensor] = None
        self._dw_numel: int = 0
        self.last_rms_dw_exponent: Optional[float] = None
        self.last_rms_dw_mantissa: Optional[float] = None

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = closure() if closure is not None else None

        if self.track_dw_rms:
            self._dw_exp_sumsq = None
            self._dw_man_sumsq = None
            self._dw_numel = 0

        for grp in self.param_groups:
            t = int(grp.get("t", 0))

            lr_m = float(grp["lr"])
            grp["last_lr_m"] = lr_m

            if grp["tie_e_to_m"]:
                lr_e = lr_m * float(grp["ratio_e_init"])
            else:
                lr_e = _sched_value(
                    mode=grp["sched_e"],
                    base_lr=float(grp["lr_e_base"]),
                    t=t,
                    total_steps=grp["total_steps_e"],
                    warmup_steps=int(grp["warmup_steps_e"]),
                    min_lr_ratio=float(grp["min_lr_ratio_e"]),
                    logcosine_alpha=float(grp["logcosine_alpha_e"]),
                )

            grp["last_lr_e"] = float(lr_e)

            b1 = float(grp["beta1"])
            b2 = float(grp["beta2"])
            eps = float(grp["eps"])
            gmax = float(grp["g_bound"])

            wd_m = float(grp["weight_decay_m"])
            wd_e = float(grp["weight_decay_e"])

            clamp_w = bool(grp["abs_clamp"])
            w_floor0 = float(grp["abs_clamp_floor"])

            clip_e = bool(grp["clip_e_final"])
            emin = float(grp["e_final_min"])
            emax = float(grp["e_final_max"])

            cap_de = bool(grp["use_de_step_cap"])
            de_cap = float(grp["de_step_cap"])

            for p in grp["params"]:
                if p.grad is None:
                    continue

                dt = p.data.dtype
                w = p.data
                g = (
                    p.grad.data.to(dt)
                    if p.grad.data.dtype != dt
                    else p.grad.data
                )

                st = self.state[p]

                if not st:
                    st["step_m"] = 0
                    st["step_e"] = 0
                    st["w_exp_avg"] = torch.zeros_like(w, dtype=dt)
                    st["w_exp_avg_sq"] = torch.zeros_like(w, dtype=dt)
                    st["exp_avg_sq"] = torch.zeros_like(w, dtype=dt)

                    init = _rms(w.float()).item()
                    st["max"] = max(
                        grp["p_scale"] * (init + 1e-12),
                        w_floor0,
                    )

                mw = st["w_exp_avg"]
                vw = st["w_exp_avg_sq"]
                ve = st["exp_avg_sq"]

                wf = w.float()
                gw = g.float()

                m, e = torch.frexp(wf)
                e_use = (
                    torch.clamp(e, min=emin, max=emax)
                    if clip_e
                    else e
                )

                st["step_e"] += 1
                se = st["step_e"]

                e_stats = torch.clamp(
                    e_use,
                    min=-60.0,
                    max=60.0,
                ).to(wf.dtype)

                w_cur = m * torch.exp2(e_stats)
                ge = gw * w_cur * math.log(2.0)

                ve_f = ve.float()
                ge_clip = ge.clamp(-1e19, 1e19)

                ve_f.mul_(b2).addcmul_(
                    ge_clip,
                    ge_clip,
                    value=1.0 - b2,
                )

                den = (
                    ve_f / (1.0 - b2**se)
                ).sqrt_().clamp_(min=eps)

                ge_n = (ge / den).clamp_(-gmax, gmax)
                dw_e = -lr_e * ge_n

                w_floor = max(
                    1e-8,
                    float(_rms(wf)) * 1e-6,
                )

                dabs = w_cur.abs().clamp_min(w_floor)

                r = (dw_e / dabs).clamp(
                    min=-0.75 + 1e-6,
                    max=0.75,
                )

                de = torch.log1p(r) / math.log(2.0)

                if cap_de:
                    de = de.clamp(
                        min=-de_cap,
                        max=de_cap,
                    )

                e_new = (
                    e_use.to(wf.dtype) * (1.0 - lr_e * wd_e)
                    + de
                )

                if clip_e:
                    e_new = torch.clamp(
                        e_new,
                        min=emin,
                        max=emax,
                    )

                ve.copy_(ve_f.to(dt))

                st["step_m"] += 1
                sm = st["step_m"]

                exp_scale = torch.exp2(-e_use.to(wf.dtype))
                gm = gw * torch.exp2(e_use.to(wf.dtype))
                gm_dt = gm.to(dt)
                gm_scaled = gm_dt * exp_scale

                mw.mul_(b1).add_(
                    gm_scaled,
                    alpha=1.0 - b1,
                )

                vw.mul_(b2).addcmul_(
                    gm_scaled,
                    gm_scaled,
                    value=1.0 - b2,
                )

                mh = mw.float() / (1.0 - b1**sm)
                vh = vw.float() / (1.0 - b2**sm)

                dw = -lr_m * mh / (vh.sqrt() + eps)

                if wd_m != 0.0:
                    w_adamw = (
                        wf * (1.0 - lr_m * wd_m)
                        + dw
                    )
                else:
                    w_adamw = wf + dw

                d_w = w_adamw - wf
                d_m = d_w * torch.exp2(-e_new)
                m_new = m + d_m
                w_new = m_new * torch.exp2(e_new)

                if self.track_dw_rms:
                    # Additive split of this step's weight change:
                    #   Δw_exponent = m * 2**e_new - w   (the "w * 2**de - w" term)
                    #   Δw_mantissa = d_w                (the AdamW additive part)
                    # (Δw_exponent + Δw_mantissa == w_new - w, pre-clamp.)
                    dwe = m * torch.exp2(e_new) - wf
                    dwm = d_w
                    dwe_l = dwe.detach()
                    dwm_l = dwm.detach()
                    if hasattr(dwe_l, "to_local"):
                        dwe_l = dwe_l.to_local()
                        dwm_l = dwm_l.to_local()
                    se = dwe_l.double().pow(2).sum()
                    sm = dwm_l.double().pow(2).sum()
                    if self._dw_exp_sumsq is None:
                        self._dw_exp_sumsq = se
                        self._dw_man_sumsq = sm
                    else:
                        self._dw_exp_sumsq = self._dw_exp_sumsq + se
                        self._dw_man_sumsq = self._dw_man_sumsq + sm
                    self._dw_numel += dwe_l.numel()

                if clamp_w:
                    w_new.clamp_(
                        -st["max"],
                        st["max"],
                    )

                w.copy_(w_new.to(dt))

            grp["t"] = t + 1

        return loss


class MAdamOptimizersContainer(OptimizersContainer):
    """:class:`OptimizersContainer` that builds :class:`m_adam`.

    Selectable from a model ``config_registry`` via::

        config.optimizer = MAdamOptimizersContainer.Config(lr=1e-3, lr_e=1e-2)

    The standard ``lr`` field maps to ``m_adam``'s ``lr_m`` (the additive
    AdamW branch), so the usual torchtitan LR scheduler drives ``lr_m``
    for free; the exponent learning rate ``lr_e`` is scheduled
    independently via ``sched_e`` (or tied to ``lr_m`` with
    ``tie_e_to_m``).

    Note: the inherited ``weight_decay`` and ``implementation`` fields are
    unused -- ``m_adam`` has decoupled ``weight_decay_m`` / ``weight_decay_e``
    and does not support the fused/foreach implementations.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        name: str = "m_adam"
        lr: float = 1e-3
        """Learning rate for the additive (AdamW / mantissa) branch (m_adam ``lr_m``)."""

        lr_e: float = 1e-2
        """Learning rate for the multiplicative (Madam / exponent) branch."""

        beta1: float = 0.9
        beta2: float = 0.999
        eps: float = 1e-8

        p_scale: float = 3.0
        """Magnitude-clamp bound as a multiple of the initial weight RMS (used only when ``abs_clamp``)."""

        g_bound: float = 20.0
        """Clamp on the RMS-normalized exponent gradient."""

        weight_decay_m: float = 0.0
        weight_decay_e: float = 0.0

        abs_clamp: bool = False
        """Clamp weights to +/- p_scale * rms(init) (the original Madam safety net; off by default)."""
        abs_clamp_floor: float = 0.0

        clip_e_final: bool = False
        e_final_min: float = -60.0
        e_final_max: float = 60.0

        use_de_step_cap: bool = True
        de_step_cap: float = 0.5

        tie_e_to_m: bool = False
        """If set, lr_e = lr_m * (lr_e / lr_m at init), so lr_e tracks the lr_m schedule."""

        sched_e: str = "constant"
        """Schedule for lr_e: 'constant' | 'linear' | 'cosine' | 'logcosine'."""
        total_steps_e: Optional[int] = None
        warmup_steps_e: int = 0
        min_lr_ratio_e: float = 0.0
        logcosine_alpha_e: float = 6.0

        track_dw_rms: bool = True
        """Log RMS of the per-step exponent-branch vs mantissa-branch weight change every step."""

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name != "m_adam":
            raise NotImplementedError(
                f"MAdamOptimizersContainer only builds 'm_adam', got {name!r}."
            )
        return m_adam

    @staticmethod
    def _build_optimizer_kwargs(config: "MAdamOptimizersContainer.Config") -> dict[str, Any]:
        return {
            "lr_m": config.lr,
            "lr_e": config.lr_e,
            "beta1": config.beta1,
            "beta2": config.beta2,
            "eps": config.eps,
            "p_scale": config.p_scale,
            "g_bound": config.g_bound,
            "weight_decay_m": config.weight_decay_m,
            "weight_decay_e": config.weight_decay_e,
            "abs_clamp": config.abs_clamp,
            "abs_clamp_floor": config.abs_clamp_floor,
            "clip_e_final": config.clip_e_final,
            "e_final_min": config.e_final_min,
            "e_final_max": config.e_final_max,
            "use_de_step_cap": config.use_de_step_cap,
            "de_step_cap": config.de_step_cap,
            "tie_e_to_m": config.tie_e_to_m,
            "sched_e": config.sched_e,
            "total_steps_e": config.total_steps_e,
            "warmup_steps_e": config.warmup_steps_e,
            "min_lr_ratio_e": config.min_lr_ratio_e,
            "logcosine_alpha_e": config.logcosine_alpha_e,
            "track_dw_rms": config.track_dw_rms,
        }

    def step(self, *args, **kwargs) -> None:
        super().step(*args, **kwargs)
        self._log_dw_rms()

    def _log_dw_rms(self) -> None:
        """Aggregate the per-step exponent/mantissa weight-change RMS across all
        model-part optimizers and all ranks, then log one line on rank 0.

        RMS is invariant to parameter *replication* (sum-of-squares and count
        scale together), so a plain SUM all-reduce over the world yields the
        correct global RMS whether params are FSDP-sharded or replicated.
        """
        opts = [o for o in self.optimizers if getattr(o, "track_dw_rms", False)]
        if not opts:
            return

        exp_sumsq: Optional[torch.Tensor] = None
        man_sumsq: Optional[torch.Tensor] = None
        numel = 0
        step = 0
        for o in opts:
            step = max(step, int(o.param_groups[0].get("t", 0)))
            if o._dw_exp_sumsq is None:
                continue
            exp_sumsq = (
                o._dw_exp_sumsq if exp_sumsq is None else exp_sumsq + o._dw_exp_sumsq
            )
            man_sumsq = (
                o._dw_man_sumsq if man_sumsq is None else man_sumsq + o._dw_man_sumsq
            )
            numel += o._dw_numel

        if exp_sumsq is None or numel == 0:
            return

        device = exp_sumsq.device
        packed = torch.stack([
            exp_sumsq.double(),
            man_sumsq.double(),
            torch.tensor(float(numel), dtype=torch.float64, device=device),
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        exp_ss, man_ss, n = packed.tolist()
        if n <= 0:
            return

        rms_exp = math.sqrt(max(0.0, exp_ss) / n)
        rms_man = math.sqrt(max(0.0, man_ss) / n)
        self.last_rms_dw_exponent = rms_exp
        self.last_rms_dw_mantissa = rms_man

        is_dist = dist.is_available() and dist.is_initialized()
        if (not is_dist) or dist.get_rank() == 0:
            ratio = rms_exp / rms_man if rms_man > 0 else float("inf")
            logger.info(
                f"[m_adam] step: {step}  rms_dw_exp: {rms_exp:.6e}  "
                f"rms_dw_man: {rms_man:.6e}  exp/man: {ratio:.4f}"
            )
