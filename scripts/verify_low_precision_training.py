#!/usr/bin/env python3
"""Runtime functional verification that a low-precision recipe is actually
training the wrapped layers — i.e. not silently falling back to BF16 / FP32
high-precision paths.

This complements ``scripts/verify_lpt_wrapper_active.py`` (which only inspects
training logs post-hoc): here we build a *toy* model, attach the same
``LowPrecisionTrainingModifier`` that production recipes use, run a few real
forward / backward / optimizer steps, and probe 8 independent signals.

Eight checks
------------
  C1 wrapper module identity
  C2 forward-path invocation (counter-instrumented quant kernel routing)
  C3 backward gradient flow (grad lands on wrapper, not on ._data)
  C4 parameter update after optimizer step
  C5 loss contribution (perturb wrapped weight → loss must shift)
  C6 high-precision fallback detection (bf16_* / bypass_* / env vars)
  C7 precision metadata sanity (PTS / block scale / QDQ round-trip)
  C8 optimizer state tracking (param in param_groups + state populated)

Each check produces ``PASS / FAIL / WARN / SKIP`` independently. The overall
verdict is ``FAIL`` if any check fails, ``WARN`` if any warns and none fails,
``PASS`` otherwise.

Side-effect safety: the script builds a toy model in process memory only;
no checkpoints or recipe files are mutated.

Example
-------
    python scripts/verify_low_precision_training.py \
        --recipe alto/models/gpt_oss/configs/lpt_nvfp4_20b_r8_recipe.yaml \
        --steps 3
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Disable Triton-global constexpr requirement so the NVFP4 kernels can run
# under the verifier process (matches the production orchestrator setting).
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Recipe loading
# ----------------------------------------------------------------------

DEFAULT_RECIPES = {
    "nvfp4": {
        "scheme": "nvfp4",
        "targets": ["Linear"],
        "ignore": ["output"],
        "use_2dblock_x": False,
        "use_2dblock_w": True,
        "use_hadamard": True,
        "use_sr_grad": True,
        "use_dge": False,
        "clip_mode": "none",
        "two_level_scaling": "tensorwise",
    },
    "mxfp4": {
        "scheme": "mxfp4",
        "targets": ["Linear"],
        "ignore": ["output"],
        "use_2dblock_x": False,
        "use_2dblock_w": True,
        "use_hadamard": True,
        "use_sr_grad": True,
        "use_dge": False,
        "clip_mode": "none",
        "two_level_scaling": "none",
    },
    "bf16": None,  # sentinel: no LPT modifier
}


def _load_recipe(recipe_path: Optional[str], precision: Optional[str]) -> dict:
    if recipe_path is None and precision is None:
        raise SystemExit("either --recipe or --precision must be specified")
    if recipe_path is not None:
        import yaml

        with open(recipe_path) as f:
            doc = yaml.safe_load(f)
        ts = doc.get("training_stage") or doc
        mods = ts.get("lpt_modifiers") or {}
        params = mods.get("LowPrecisionTrainingModifier")
        if params is None:
            raise SystemExit(
                f"recipe {recipe_path!r} has no "
                "training_stage.lpt_modifiers.LowPrecisionTrainingModifier"
            )
        return dict(params)
    if precision == "bf16":
        return {}
    if precision not in DEFAULT_RECIPES:
        raise SystemExit(
            f"unknown --precision={precision!r}; "
            f"expected one of {sorted(DEFAULT_RECIPES)}"
        )
    return dict(DEFAULT_RECIPES[precision])


def _expected_class_name(precision: str) -> Optional[str]:
    return {
        "mxfp4": "MXFP4TrainingWeightWrapperTensor",
        "nvfp4": "NVFP4TrainingWeightWrapperTensor",
        "mxfp8_e4m3": "MXFP8TrainingWeightWrapperTensor",
        "mxfp8_e5m2": "MXFP8TrainingWeightWrapperTensor",
    }.get(precision)


# ----------------------------------------------------------------------
# Toy model
# ----------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.wq = nn.Linear(dim, dim, dtype=dtype)
        self.wk = nn.Linear(dim, dim, dtype=dtype)
        self.wv = nn.Linear(dim, dim, dtype=dtype)
        self.wo = nn.Linear(dim, dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Minimal attention-shaped MAC: no softmax, no positional mixing —
        # we only need a forward path that drives all four wq/wk/wv/wo
        # Linear modules so the wrapped quant kernels are exercised.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        return self.wo(q + k + v)


class _MLP(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, dtype=dtype)
        self.down = nn.Linear(4 * dim, dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.attention = _Attn(dim, dtype)
        self.mlp = _MLP(dim, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x + self.attention(x))


class ToyLPTModel(nn.Module):
    """Toy mixed-precision-friendly module that mimics the targets the
    GPT-OSS LPT recipe wraps: ``attention.{wq,wk,wv,wo}`` + ``mlp.{up,down}``,
    with an ``output`` Linear that should be matched by ``ignore=["output"]``.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        vocab_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, dtype=dtype)
        self.layers = nn.ModuleList(
            [_Block(dim, dtype) for _ in range(num_layers)]
        )
        self.output = nn.Linear(dim, vocab_size, dtype=dtype)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.layers:
            h = blk(h)
        return self.output(h)


# ----------------------------------------------------------------------
# Check results
# ----------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VerifierContext:
    args: argparse.Namespace
    recipe: dict
    precision: Optional[str]
    expected_class: Optional[str]
    model: nn.Module
    device: torch.device
    dtype: torch.dtype
    wrapped_fqns: list[str] = field(default_factory=list)
    unwrapped_target_fqns: list[str] = field(default_factory=list)
    forward_call_log: dict = field(default_factory=dict)
    optim: Optional[torch.optim.Optimizer] = None


# ----------------------------------------------------------------------
# Wrapper module identity (C1)
# ----------------------------------------------------------------------

def _candidate_target_fqns(model: nn.Module, ignore_patterns: list[str]) -> tuple[list[str], list[str]]:
    """Return (fqns that match a Linear target, fqns that match the ignore
    pattern explicitly)."""
    ignored, matched = [], []
    compiled = [re.compile(_glob_to_re(p)) for p in ignore_patterns]
    for fqn, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(rx.match(fqn) for rx in compiled):
                ignored.append(fqn)
            else:
                matched.append(fqn)
    return matched, ignored


def _glob_to_re(pat: str) -> str:
    """Translate the recipe ignore pattern syntax to a regex.

    The torchtitan recipe uses literal strings (e.g. ``"output"``) and
    ``re:<regex>`` for explicit regex. Mirror that here so we can do a
    purely-local cross-check of which modules should have been swapped.
    """
    if pat.startswith("re:"):
        return f"^(?:.*\\.)?{pat[3:]}$"
    # Treat the bare string as ".*pat$" so "output" matches both "output"
    # and "transformer.output" without forcing the caller to spell the FQN.
    return f"^(?:.*\\.)?{re.escape(pat)}$"


def check_C1_wrapper_module_identity(ctx: VerifierContext) -> CheckResult:
    """Walk the model and verify every Linear that should be wrapped has had
    its weight replaced by the expected ``*TrainingWeightWrapperTensor``."""
    if ctx.expected_class is None:
        return CheckResult(
            "C1 wrapper module identity",
            SKIP,
            "BF16 baseline run; no wrapping is expected.",
        )

    from alto.kernels.dispatch.tensor import TrainingWeightWrapperBaseTensor

    expected_targets, ignored = _candidate_target_fqns(ctx.model, ctx.recipe.get("ignore", []))
    wrapped, unwrapped = [], []
    bad_cls = []
    for fqn in expected_targets:
        module: nn.Module = dict(ctx.model.named_modules())[fqn]
        w = module.weight
        is_wrapped = isinstance(w.data, TrainingWeightWrapperBaseTensor)
        if not is_wrapped:
            unwrapped.append(fqn)
            continue
        wrapped.append(fqn)
        if type(w.data).__name__ != ctx.expected_class:
            bad_cls.append((fqn, type(w.data).__name__))

    ctx.wrapped_fqns = list(wrapped)
    ctx.unwrapped_target_fqns = list(unwrapped)

    ok = (
        len(expected_targets) > 0
        and len(unwrapped) == 0
        and len(bad_cls) == 0
    )
    return CheckResult(
        "C1 wrapper module identity",
        PASS if ok else FAIL,
        detail=(
            f"linears_to_swap={len(expected_targets)} "
            f"wrapped={len(wrapped)} unwrapped_targets={len(unwrapped)} "
            f"ignored={len(ignored)} unexpected_class={bad_cls}"
        ),
        metrics={
            "expected_targets": expected_targets,
            "wrapped_fqns": wrapped,
            "unwrapped_target_fqns": unwrapped,
            "ignored_fqns": ignored,
            "unexpected_class_fqns": [t[0] for t in bad_cls],
        },
    )


# ----------------------------------------------------------------------
# Forward-path invocation (C2)
# ----------------------------------------------------------------------

def _instrument_dispatch_symbols(ctx: VerifierContext):
    """Wrap the dispatch tensor's quant kernel entry points with counters so
    we can prove the low-precision forward path was actually invoked.

    Returns a context manager that restores the originals on exit.
    """
    import alto.kernels.dispatch.tensor as dispatch_tensor

    SYMBOLS = [
        "_to_nvfp4_then_scaled_mm",
        "_quantize_then_nvfp4_scaled_grouped_mm",
        "_to_mxfp4_then_scaled_mm",
        "_quantize_then_mxfp_scaled_grouped_mm",
        "_to_mxfp8_then_scaled_mm",
    ]
    counters = {sym: 0 for sym in SYMBOLS}
    originals = {}

    def _wrap(sym: str, fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            counters[sym] += 1
            return fn(*args, **kwargs)

        wrapped.__name__ = fn.__name__
        return wrapped

    @contextlib.contextmanager
    def _ctx_mgr():
        for sym in SYMBOLS:
            if hasattr(dispatch_tensor, sym):
                originals[sym] = getattr(dispatch_tensor, sym)
                setattr(dispatch_tensor, sym, _wrap(sym, originals[sym]))
        try:
            yield counters
        finally:
            for sym, original in originals.items():
                setattr(dispatch_tensor, sym, original)

    return _ctx_mgr()


def _toy_step(ctx: VerifierContext, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward+backward of the toy model and return (loss, logits)."""
    tokens = torch.randint(0, vocab_size, (ctx.args.batch, ctx.args.seq), device=ctx.device)
    targets = torch.randint(0, vocab_size, (ctx.args.batch, ctx.args.seq), device=ctx.device)
    logits = ctx.model(tokens)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1)
    )
    loss.backward()
    return loss, logits


def check_C2_forward_path_invocation(ctx: VerifierContext) -> CheckResult:
    if ctx.expected_class is None:
        return CheckResult("C2 forward-path invocation", SKIP, "BF16 baseline.")

    counters: dict = {}
    try:
        with _instrument_dispatch_symbols(ctx) as cnt:
            counters = cnt
            tokens = torch.randint(
                0, ctx.args.vocab, (ctx.args.batch, ctx.args.seq), device=ctx.device
            )
            with torch.no_grad():
                _ = ctx.model(tokens)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C2 forward-path invocation",
            FAIL,
            detail=f"forward raised {type(e).__name__}: {e}",
            error=traceback.format_exc(),
        )

    ctx.forward_call_log = dict(counters)
    if ctx.precision and ctx.precision.startswith("nvfp4"):
        relevant = {k: v for k, v in counters.items() if "nvfp4" in k}
    elif ctx.precision and ctx.precision.startswith("mxfp4"):
        relevant = {k: v for k, v in counters.items() if "mxfp" in k}
    elif ctx.precision and ctx.precision.startswith("mxfp8"):
        relevant = {k: v for k, v in counters.items() if "mxfp8" in k}
    else:
        relevant = dict(counters)
    total_relevant = sum(relevant.values())
    return CheckResult(
        "C2 forward-path invocation",
        PASS if total_relevant > 0 else FAIL,
        detail=(
            f"quant kernel calls during forward: {counters}; "
            f"relevant_for_precision={ctx.precision!r}: total={total_relevant}"
        ),
        metrics={"counters": counters, "relevant_counter_sum": total_relevant},
    )


# ----------------------------------------------------------------------
# Backward gradient flow (C3)
# ----------------------------------------------------------------------

def check_C3_backward_gradient_flow(ctx: VerifierContext) -> CheckResult:
    """One forward + backward; assert ``param.grad`` lives on the *wrapper*,
    not on ``param._data.grad`` (the production "wrapper frozen" footgun
    described in ``test_nvfp_dispatch_guards.py``).
    """
    from alto.kernels.dispatch.tensor import TrainingWeightWrapperBaseTensor

    is_lp_run = ctx.expected_class is not None
    try:
        ctx.model.zero_grad(set_to_none=True)
        loss, _ = _toy_step(ctx, vocab_size=ctx.args.vocab)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C3 backward gradient flow",
            FAIL,
            detail=f"backward raised {type(e).__name__}: {e}",
            error=traceback.format_exc(),
        )

    loss_val = float(loss.detach())
    if not torch.isfinite(loss):
        return CheckResult(
            "C3 backward gradient flow",
            FAIL,
            detail=f"loss is not finite: {loss_val}",
        )

    leaks_to_data = []
    none_grads_on_wrapped = []
    grad_norms = []
    for fqn in ctx.wrapped_fqns:
        module = dict(ctx.model.named_modules())[fqn]
        p = module.weight
        if not is_lp_run:
            continue
        if p.grad is None:
            none_grads_on_wrapped.append(fqn)
            continue
        if isinstance(p.data, TrainingWeightWrapperBaseTensor) and p._data.grad is not None:
            leaks_to_data.append(fqn)
        grad_norms.append(float(p.grad.detach().float().norm()))

    if is_lp_run and not ctx.wrapped_fqns:
        return CheckResult(
            "C3 backward gradient flow",
            FAIL,
            detail="no wrapped Linear modules to check; C1 should have failed first",
        )

    ok = (
        torch.isfinite(loss).item()
        and (not is_lp_run or not none_grads_on_wrapped)
        and not leaks_to_data
    )
    return CheckResult(
        "C3 backward gradient flow",
        PASS if ok else FAIL,
        detail=(
            f"loss={loss_val:.6f} wrapped_with_grad="
            f"{len(grad_norms)} wrapped_without_grad="
            f"{len(none_grads_on_wrapped)} leaks_to_data={len(leaks_to_data)} "
            f"mean_grad_norm={(sum(grad_norms)/max(1,len(grad_norms))):.4e}"
        ),
        metrics={
            "loss": loss_val,
            "wrapped_with_grad_count": len(grad_norms),
            "wrapped_without_grad_count": len(none_grads_on_wrapped),
            "leaks_to_data": leaks_to_data,
            "none_grads_on_wrapped": none_grads_on_wrapped,
            "grad_norms_sample": grad_norms[:5],
        },
    )


# ----------------------------------------------------------------------
# Parameter update (C4) + Optimizer state (C8)
# ----------------------------------------------------------------------

def _underlying_data(p: nn.Parameter) -> torch.Tensor:
    from alto.kernels.dispatch.tensor import TrainingWeightWrapperBaseTensor

    if isinstance(p.data, TrainingWeightWrapperBaseTensor):
        return p.data._data
    return p.data


def check_C4_parameter_update(ctx: VerifierContext) -> CheckResult:
    """Run a few optimizer steps and assert wrapped weights move beyond
    floating-point noise floor.  ``param.grad`` is cleared each step."""
    steps = max(1, ctx.args.steps)
    snapshots_before = {}
    for fqn in ctx.wrapped_fqns:
        module = dict(ctx.model.named_modules())[fqn]
        snapshots_before[fqn] = _underlying_data(module.weight).detach().clone()

    optim = torch.optim.AdamW(ctx.model.parameters(), lr=1e-2)
    ctx.optim = optim

    losses = []
    for _ in range(steps):
        ctx.model.zero_grad(set_to_none=True)
        loss, _ = _toy_step(ctx, vocab_size=ctx.args.vocab)
        loss_val = float(loss.detach())
        if not torch.isfinite(loss):
            return CheckResult(
                "C4 parameter update",
                FAIL,
                detail=f"step loss not finite: {loss_val}",
            )
        losses.append(loss_val)
        optim.step()

    stale, moved = [], []
    rel_changes = []
    for fqn, before in snapshots_before.items():
        module = dict(ctx.model.named_modules())[fqn]
        after = _underlying_data(module.weight)
        delta = (after.float() - before.float()).norm().item()
        ref = max(before.float().norm().item(), 1e-12)
        rel = delta / ref
        rel_changes.append((fqn, rel))
        if rel > 1e-6:
            moved.append(fqn)
        else:
            stale.append(fqn)

    ok = ctx.expected_class is None or (
        len(ctx.wrapped_fqns) > 0 and not stale
    )
    return CheckResult(
        "C4 parameter update",
        PASS if ok else FAIL,
        detail=(
            f"steps={steps} losses={losses} "
            f"wrapped_moved={len(moved)} wrapped_stale={len(stale)} "
            f"max_rel_change={max((c for _, c in rel_changes), default=0):.4e}"
        ),
        metrics={
            "step_losses": losses,
            "wrapped_moved": moved,
            "wrapped_stale": stale,
            "rel_changes_sample": rel_changes[:5],
        },
    )


def check_C8_optimizer_state(ctx: VerifierContext) -> CheckResult:
    if ctx.optim is None:
        return CheckResult(
            "C8 optimizer state",
            SKIP,
            "optimizer was not constructed (C4 prerequisite missing)",
        )

    state = ctx.optim.state
    tracked_params = {id(p) for g in ctx.optim.param_groups for p in g["params"]}
    wrapped_param_ids = []
    for fqn in ctx.wrapped_fqns:
        module = dict(ctx.model.named_modules())[fqn]
        wrapped_param_ids.append(id(module.weight))

    not_tracked = [fqn for fqn, pid in zip(ctx.wrapped_fqns, wrapped_param_ids) if pid not in tracked_params]

    populated = 0
    missing_state = []
    for fqn, pid in zip(ctx.wrapped_fqns, wrapped_param_ids):
        # AdamW stores state keyed by the python id of the parameter object.
        param = dict(ctx.model.named_modules())[fqn].weight
        s = state.get(param, {})
        if s.get("step", 0) >= 1 and "exp_avg" in s and "exp_avg_sq" in s:
            populated += 1
        else:
            missing_state.append(fqn)

    ok = not not_tracked and (ctx.expected_class is None or not missing_state)
    return CheckResult(
        "C8 optimizer state",
        PASS if ok else FAIL,
        detail=(
            f"wrapped_in_param_groups={len(wrapped_param_ids) - len(not_tracked)}/"
            f"{len(wrapped_param_ids)} wrapped_with_optim_state={populated}/"
            f"{len(wrapped_param_ids)} missing_state={len(missing_state)}"
        ),
        metrics={
            "not_in_param_groups": not_tracked,
            "missing_state": missing_state,
        },
    )


# ----------------------------------------------------------------------
# Loss-contribution perturbation (C5)
# ----------------------------------------------------------------------

def check_C5_loss_contribution(ctx: VerifierContext) -> CheckResult:
    """Perturb each wrapped weight by a small epsilon and assert the loss
    actually shifts.  If a wrapped weight is *off* the forward critical path
    (e.g. the wrapper is unreachable), this will report ``loss invariant``.
    """
    if not ctx.wrapped_fqns and ctx.expected_class is not None:
        return CheckResult(
            "C5 loss contribution",
            FAIL,
            "no wrapped layers to perturb",
        )
    if ctx.expected_class is None:
        return CheckResult("C5 loss contribution", SKIP, "BF16 baseline.")

    eps = float(ctx.args.perturb_eps)
    sample_n = max(1, len(ctx.wrapped_fqns) // 2)
    sampled = ctx.wrapped_fqns[:sample_n]
    tokens = torch.randint(
        0, ctx.args.vocab, (ctx.args.batch, ctx.args.seq), device=ctx.device
    )
    targets = torch.randint(
        0, ctx.args.vocab, (ctx.args.batch, ctx.args.seq), device=ctx.device
    )

    def _loss() -> float:
        with torch.no_grad():
            logits = ctx.model(tokens)
            return float(
                nn.functional.cross_entropy(
                    logits.reshape(-1, ctx.args.vocab), targets.reshape(-1)
                )
            )

    # base loss is deterministic only for the RNE forward path; the production
    # NVFP4 default recipe sets ``use_sr_grad=True`` which makes forward-on-bwd
    # SR-QDQ stochastic, but the *forward* QDQ used during eval is RNE under
    # ``torch.no_grad``.  Run base twice as a sanity check and use the second.
    _ = _loss()
    base = _loss()

    # Multiplicative perturbation: ``w := w * (1 + eps)``.  Survives QDQ
    # because PTS / per-block scale rescale proportionally, so the quantised
    # weight changes by ≈ eps (not by a sub-grid sliver).  An invariant loss
    # under this perturbation is genuine evidence that the layer is off the
    # forward critical path.
    invariant, sensitive = [], []
    eps_eff = max(eps, 1.0e-2)
    threshold = max(1e-3, 1e-3 * abs(base))
    for fqn in sampled:
        module = dict(ctx.model.named_modules())[fqn]
        data = _underlying_data(module.weight)
        backup = data.detach().clone()
        try:
            data.mul_(1.0 + eps_eff)
            shifted = _loss()
        finally:
            data.copy_(backup)
        delta = abs(shifted - base)
        if delta > threshold:
            sensitive.append((fqn, delta))
        else:
            invariant.append((fqn, delta))

    n_sensitive = len(sensitive)
    n_sampled = len(sampled)
    if n_sensitive == 0:
        status = FAIL
    elif n_sensitive >= max(1, n_sampled // 2):
        status = PASS
    else:
        status = WARN
    return CheckResult(
        "C5 loss contribution",
        status,
        detail=(
            f"perturb_eps={eps_eff} base_loss={base:.4f} threshold={threshold:.4e} "
            f"sampled={n_sampled} sensitive={n_sensitive} invariant={len(invariant)}"
        ),
        metrics={
            "base_loss": base,
            "perturb_eps": eps_eff,
            "threshold": threshold,
            "sampled_fqns": sampled,
            "sensitive_sample": sensitive[:5],
            "invariant_sample": invariant[:5],
        },
    )


# ----------------------------------------------------------------------
# High-precision fallback detection (C6)
# ----------------------------------------------------------------------

KNOWN_FALLBACK_ENV_VARS = (
    # Any env-var that the Phase B investigation introduced as a "knob".
    # The corresponding per-module bypass fields (``bf16_forward``,
    # ``bf16_backward``, ``bypass_dY_dgrad`` / ``bypass_w_dgrad`` /
    # ``bypass_dY_wgrad`` / ``bypass_x_wgrad`` and the ``shared_dY_sr_seed``
    # debug knob) have been removed from production code; this check now
    # only watches for stray env-var fallbacks left over from old shell
    # wrappers, which is still a useful guard.
    "NVFP4_DY_BLOCK_SIZE",
    "NVFP4_DY_SCALE_ROUND_MODE",
    "NVFP4_DY_SCALE_NO_CLAMP",
    "NVFP4_DY_SCALE_MARGIN",
    "NVFP4_DY_DGRAD_HADAMARD",
    "NVFP4_DY_USE_MXFP4_QDQ",
    "NVFP4_DY_QDQ_MODE",
)


def check_C6_fallback_detection(ctx: VerifierContext) -> CheckResult:
    if ctx.expected_class is None:
        return CheckResult(
            "C6 high-precision fallback detection",
            SKIP,
            "BF16 baseline; no LPT bypass flags relevant.",
        )

    active_env = {
        k: os.environ[k]
        for k in KNOWN_FALLBACK_ENV_VARS
        if os.environ.get(k) and os.environ[k].lower() not in ("0", "false", "")
    }

    ok = not active_env
    return CheckResult(
        "C6 high-precision fallback detection",
        PASS if ok else FAIL,
        detail=f"env_fallback={active_env}",
        metrics={"env_fallback": active_env},
    )


# ----------------------------------------------------------------------
# Precision metadata sanity (C7)
# ----------------------------------------------------------------------

def check_C7_precision_metadata(ctx: VerifierContext) -> CheckResult:
    if ctx.expected_class is None:
        return CheckResult("C7 precision metadata", SKIP, "BF16 baseline.")
    if not ctx.wrapped_fqns:
        return CheckResult("C7 precision metadata", FAIL, "no wrapped layers")

    fqn = ctx.wrapped_fqns[0]
    module = dict(ctx.model.named_modules())[fqn]
    w = module.weight._data.detach().to(ctx.device)

    try:
        if ctx.precision == "nvfp4":
            from alto.kernels.fp4.nvfp4.nvfp_quantization import (
                E4M3_EPS,
                F4_E2M1_MAX,
                F8E4M3_MAX,
                compute_dynamic_outer_scale,
                convert_from_nvfp4,
                convert_to_nvfp4,
            )

            pts = compute_dynamic_outer_scale(w)
            pts_val = float(pts.item())
            # The spec outer-scale (a.k.a. per-tensor scale / s_global) is FP32
            # and *not* clamped to E4M3_EPS; if a well-conditioned weight's
            # outer-scale lands exactly on E4M3_EPS, the spec-fix has likely
            # been reverted.
            spec_floor_active = pts_val == E4M3_EPS and (
                float(w.float().abs().max())
                / (F8E4M3_MAX * F4_E2M1_MAX)
            ) < E4M3_EPS
            data_lp, scales = convert_to_nvfp4(
                w,
                block_size=16,
                axis=-1,
                is_2d_block=False,
                outer_scale=pts,
                update_outer_scale=False,
            )
            wq = convert_from_nvfp4(
                data_lp,
                scales,
                output_dtype=w.dtype,
                block_size=16,
                axis=-1,
                is_2d_block=False,
                outer_scale=pts,
            )
            err = float((wq.float() - w.float()).norm() / max(1e-12, w.float().norm()))
            ok = (
                torch.isfinite(pts).all().item()
                and torch.isfinite(scales).all().item()
                and 0 < err < 0.5
                and not spec_floor_active
            )
            return CheckResult(
                "C7 precision metadata",
                PASS if ok else WARN,
                detail=(
                    f"sample_fqn={fqn} pts={pts_val:.4e} "
                    f"qdq_rel_err={err:.4e} "
                    f"scales_finite={bool(torch.isfinite(scales).all().item())} "
                    f"pts_at_e4m3_floor={spec_floor_active}"
                ),
                metrics={
                    "sample_fqn": fqn,
                    "pts": pts_val,
                    "qdq_rel_err": err,
                    "spec_floor_active": spec_floor_active,
                },
            )
        elif ctx.precision == "mxfp4":
            from alto.kernels.fp4.mxfp4.mxfp_quantization import (
                convert_from_mxfp4,
                convert_to_mxfp4,
            )

            data_lp, scales = convert_to_mxfp4(
                w,
                block_size=32,
                axis=-1,
                is_2d_block=False,
                use_sr=False,
                use_asm=False,
                clip_mode="none",
            )
            wq = convert_from_mxfp4(
                data_lp,
                scales,
                output_dtype=w.dtype,
                block_size=32,
                axis=-1,
                is_2d_block=False,
                use_asm=False,
                clip_mode="none",
            )
            err = float((wq.float() - w.float()).norm() / max(1e-12, w.float().norm()))
            ok = (
                torch.isfinite(scales).all().item()
                and 0 < err < 0.5
            )
            return CheckResult(
                "C7 precision metadata",
                PASS if ok else WARN,
                detail=(
                    f"sample_fqn={fqn} qdq_rel_err={err:.4e} "
                    f"scales_finite={bool(torch.isfinite(scales).all().item())}"
                ),
                metrics={"sample_fqn": fqn, "qdq_rel_err": err},
            )
        else:
            return CheckResult(
                "C7 precision metadata",
                SKIP,
                f"no metadata probe implemented for precision={ctx.precision!r}",
            )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "C7 precision metadata",
            FAIL,
            detail=f"metadata probe failed: {type(e).__name__}: {e}",
            error=traceback.format_exc(),
        )


# ----------------------------------------------------------------------
# Build / orchestration
# ----------------------------------------------------------------------

CHECK_REGISTRY: dict[str, Callable[[VerifierContext], CheckResult]] = {
    "C1": check_C1_wrapper_module_identity,
    "C2": check_C2_forward_path_invocation,
    "C3": check_C3_backward_gradient_flow,
    "C4": check_C4_parameter_update,
    "C5": check_C5_loss_contribution,
    "C6": check_C6_fallback_detection,
    "C7": check_C7_precision_metadata,
    "C8": check_C8_optimizer_state,
}


def _aggregate(results: list[CheckResult]) -> str:
    if any(r.status == FAIL for r in results):
        return FAIL
    if any(r.status == WARN for r in results):
        return WARN
    return PASS


def _build_context(args: argparse.Namespace) -> VerifierContext:
    recipe = _load_recipe(args.recipe, args.precision)
    precision = recipe.get("scheme") if recipe else None
    expected_class = _expected_class_name(precision) if precision else None

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = ToyLPTModel(
        dim=args.hidden,
        num_layers=args.num_layers,
        vocab_size=args.vocab,
        dtype=dtype,
    ).to(device)

    if recipe:
        from alto.modifiers.lpt import LowPrecisionTrainingModifier  # noqa: F401
        # ``LowPrecisionTrainingModifier`` lives at ``alto.modifiers.lpt.base``;
        # the ``alto.modifiers.lpt`` package re-exports it via __init__.
        from alto.modifiers.lpt.base import LowPrecisionTrainingModifier as _LPT

        mod = _LPT(**recipe)
        mod.on_convert(model)

    return VerifierContext(
        args=args,
        recipe=recipe or {},
        precision=precision,
        expected_class=expected_class,
        model=model,
        device=device,
        dtype=dtype,
    )


def _format_md(results: list[CheckResult], overall: str, ctx: VerifierContext) -> str:
    lines = [
        "# Low-Precision Training Verification Report",
        "",
        f"- precision: `{ctx.precision}`",
        f"- expected_class: `{ctx.expected_class}`",
        f"- device: `{ctx.device}` dtype: `{ctx.dtype}`",
        f"- recipe: {json.dumps(ctx.recipe, ensure_ascii=False)}",
        "",
        f"## OVERALL: {overall}",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for r in results:
        d = r.detail.replace("|", "\\|")
        lines.append(f"| {r.name} | {r.status} | {d} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=str, default=None)
    ap.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["nvfp4", "mxfp4", "bf16"],
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--perturb-eps", type=float, default=5e-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--checks",
        type=str,
        default="C1,C2,C3,C4,C5,C6,C7,C8",
    )
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--md-out", type=str, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any WARN appears (not just FAIL)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.device == "cpu":
        print("[warn] running on CPU; NVFP4/MXFP4 Triton kernels need CUDA. "
              "Most checks will SKIP or FAIL.", file=sys.stderr)
    ctx = _build_context(args)

    selected = [c.strip() for c in args.checks.split(",") if c.strip()]
    bad = [c for c in selected if c not in CHECK_REGISTRY]
    if bad:
        raise SystemExit(f"unknown checks {bad}; available: {sorted(CHECK_REGISTRY)}")

    t0 = time.time()
    results: list[CheckResult] = []
    for cid in selected:
        try:
            r = CHECK_REGISTRY[cid](ctx)
        except Exception as e:  # noqa: BLE001
            r = CheckResult(
                f"{cid} <crashed>",
                FAIL,
                detail=f"check raised {type(e).__name__}: {e}",
                error=traceback.format_exc(),
            )
        results.append(r)

    overall = _aggregate(results)
    elapsed = time.time() - t0

    if not args.quiet:
        print(f"[verify_low_precision_training] OVERALL: {overall}  "
              f"recipe={args.recipe or args.precision}  device={ctx.device}  "
              f"elapsed={elapsed:.1f}s")
        for r in results:
            tag = {PASS: "OK ", FAIL: "BAD", WARN: "WRN", SKIP: "---"}[r.status]
            print(f"  [{tag}] {r.name}: {r.detail}")
            if r.error and not args.quiet:
                print("        traceback (truncated):")
                for line in r.error.splitlines()[-4:]:
                    print(f"          {line}")

    json_out = Path(
        args.json_out
        or f"/tmp/verify_low_precision_training_{int(time.time())}.json"
    )
    json_payload = {
        "overall": overall,
        "elapsed_sec": elapsed,
        "recipe_arg": args.recipe,
        "precision_arg": args.precision,
        "device": str(ctx.device),
        "expected_class": ctx.expected_class,
        "recipe_resolved": ctx.recipe,
        "checks": [dataclasses.asdict(r) for r in results],
    }
    json_out.write_text(json.dumps(json_payload, indent=2, default=str))
    if not args.quiet:
        print(f"json_report={json_out}")

    if args.md_out:
        md_out = Path(args.md_out)
        md_out.write_text(_format_md(results, overall, ctx))
        if not args.quiet:
            print(f"md_report={md_out}")

    rc = 0
    if overall == FAIL:
        rc = 1
    elif overall == WARN and args.strict:
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
