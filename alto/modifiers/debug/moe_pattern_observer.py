# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MoEMatmulPatternObserverModifier — per-expert AdaHOP outlier-pattern capture
for the two Grouped GEMMs (MLP1, MLP2) of gpt_oss MoE blocks.

Intended flow: pretrain gpt_oss WITHOUT this modifier, load a checkpoint, apply
this modifier via a recipe, run a single training iteration, dump per-rank
results, then visualize with ``scripts/moe_pattern_viz.py``.

Why not the module-boundary hooks of ``DebugObserverModifier``?
A ``forward_pre``/``full_backward`` hook on ``GptOssGroupedExperts`` only sees
MLP1's input and the module-output grad — it cannot see MLP2's input, MLP1's
output grad, or split operands per expert. Instead we intercept the two
``torch._grouped_mm`` calls INSIDE ``_run_experts_grouped_mm`` (scoped to the
targeted module via a contextvar) so we get all three matmul operands of each
GEMM plus the per-expert token offsets. Capture happens after the module has
already ``.to_local()``'d its DTensor weights and after token permute, so the
tensors are plain local tensors on the current rank — robust under FSDP/EP.

Constraints: the observed step must run in EAGER mode (the monkeypatch is
invisible to a compiled graph) and the low-precision path is backend-agnostic
(works on the CDNA3 loop fallback and the CDNA4 kernels alike).
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from pydantic import Field, PrivateAttr
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.modifiers.debug.moe_pattern_hooks import (
    accumulate_majority,
    build_expert_records,
    extract_offs,
)

__all__ = ["MoEMatmulPatternObserverModifier"]

# Contextvar carrying the currently-executing experts module's capture context,
# or None when no targeted module is running. Shared by the patched _grouped_mm.
_ACTIVE_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "_moe_pattern_active_ctx", default=None
)

# gemm counter index -> logical name (order of _grouped_mm calls in the experts fwd)
_GEMM_NAMES = ("mlp1", "mlp2")


class MoEMatmulPatternObserverModifier(Modifier):
    """Captures per-expert AdaHOP outlier-pattern T-pairs for the three matmuls
    of each MoE Grouped GEMM (MLP1, MLP2), at one observed step, and dumps them.
    """

    targets: List[str] = Field(default_factory=lambda: ["GptOssGroupedExperts"])
    ignore: List[str] = Field(default_factory=list)

    capture_every: int = 1
    max_captures: int = 1
    output_path: str = "./outputs/moe_patterns.pt"

    # forwarded to detect_outlier_pattern
    threshold_ratio: float = 2.0
    kurtosis_threshold: float = 0.0

    @property
    def requires_training_mode(self) -> bool:
        # grad_output patterns require a real backward pass.
        return True

    # --- private state ---
    _step_idx: int = PrivateAttr(default=0)
    _n_captured: int = PrivateAttr(default=0)
    _detached: bool = PrivateAttr(default=False)
    _active: bool = PrivateAttr(default=False)
    # results[fqn][step_idx][gemm] = {global_expert_id: record}
    _results: dict = PrivateAttr(default_factory=dict)
    # per-fqn mesh info captured at initialize: {fqn: {ep_rank, ep_size, ...}}
    _mesh_info: dict = PrivateAttr(default_factory=dict)
    # saved originals for teardown
    _orig_grouped_mm: Any = PrivateAttr(default=None)
    _orig_forwards: dict = PrivateAttr(default_factory=dict)
    _detect: Any = PrivateAttr(default=None)
    _fqns: list = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True

    def on_initialize(self, model_parts: List[Module], **kwargs) -> bool:
        # Bind detector once (imports the adahop bridge lazily so CPU unit tests
        # of the hooks module don't drag in triton).
        from alto._adahop_bridge import detect_outlier_pattern

        def _detect(t: torch.Tensor) -> str:
            return detect_outlier_pattern(
                t,
                threshold_ratio=self.threshold_ratio,
                kurtosis_threshold=self.kurtosis_threshold,
            )

        self._detect = _detect

        for model_part in model_parts:
            for fqn, module in match_named_modules(model_part, self.targets, self.ignore):
                self._results[fqn] = {}
                self._mesh_info[fqn] = self._read_mesh_info(module)
                self._fqns.append(fqn)
                self._wrap_forward(module, fqn)

        # Install the grouped_mm patch once (no-op until a wrapped forward runs
        # AND the capture gate is active).
        self._orig_grouped_mm = torch._grouped_mm
        torch._grouped_mm = self._make_patched_grouped_mm(self._orig_grouped_mm)

        logger.info(
            f"MoEMatmulPatternObserverModifier: monitoring {len(self._fqns)} expert "
            f"layers, max_captures={self.max_captures}"
        )
        return True

    def on_pre_step(self, model_parts: List[Module], **kwargs) -> bool:
        self._step_idx += 1
        if self._detached:
            return True
        self._active = (
            (self._step_idx % self.capture_every == 0)
            and (self._n_captured < self.max_captures)
        )
        return True

    def on_post_step(self, model_parts: List[Module], **kwargs) -> bool:
        if self._detached:
            return True
        # A step counts as captured only if a grad path actually populated a
        # T-pair (i.e. a backward ran), mirroring DebugObserverModifier.
        step = self._step_idx
        captured = any(
            step in self._results[fqn] and self._results[fqn][step]
            for fqn in self._fqns
        )
        if captured:
            self._n_captured += 1
            logger.debug(
                f"MoEMatmulPatternObserverModifier: captured step {step} "
                f"({self._n_captured}/{self.max_captures})"
            )
        self._active = False
        if self._n_captured >= self.max_captures:
            logger.info("MoEMatmulPatternObserverModifier: reached max_captures, detaching")
            self._detach()
        return True

    def on_finalize(self, model_parts: List[Module], **kwargs) -> bool:
        if not self._detached:
            self._detach()
        self._dump()
        return True

    # ------------------------------------------------------------------
    # forward wrapping + grouped_mm patch
    # ------------------------------------------------------------------

    def _wrap_forward(self, module: Module, fqn: str) -> None:
        """Replace the module's bound forward with a wrapper that sets the
        contextvar (fqn + fresh gemm counter) while the original runs."""
        orig_forward = module.forward
        self._orig_forwards[fqn] = orig_forward
        modifier = self

        def _wrapped(*args, **kwargs):
            if modifier._detached or not modifier._active:
                return orig_forward(*args, **kwargs)
            ctx = {"fqn": fqn, "gemm_idx": 0}
            token = _ACTIVE_CTX.set(ctx)
            try:
                return orig_forward(*args, **kwargs)
            finally:
                _ACTIVE_CTX.reset(token)

        module.forward = _wrapped  # type: ignore[method-assign]

    def _make_patched_grouped_mm(self, orig: Callable) -> Callable:
        modifier = self

        def _patched(*args, **kwargs):
            out = orig(*args, **kwargs)
            ctx = _ACTIVE_CTX.get()
            if ctx is None or modifier._detached or not modifier._active:
                return out
            try:
                modifier._on_grouped_mm(ctx, args, kwargs, out)
            except Exception as exc:  # never break training on a debug tool
                logger.warning(f"MoEMatmulPatternObserverModifier: capture skipped ({exc})")
            return out

        return _patched

    def _on_grouped_mm(self, ctx: dict, args: tuple, kwargs: dict, out: torch.Tensor) -> None:
        gemm_idx = ctx["gemm_idx"]
        ctx["gemm_idx"] = gemm_idx + 1
        if gemm_idx >= len(_GEMM_NAMES):
            return  # unexpected extra grouped_mm; ignore
        gemm_name = _GEMM_NAMES[gemm_idx]
        fqn = ctx["fqn"]

        # operands: _grouped_mm(x, w, offs=...) — w is already transposed to [E,K,N]
        x = args[0]
        w = args[1]
        offs_t = extract_offs(args, kwargs)
        if offs_t is None or w.dim() != 3:
            return
        offs = offs_t.detach().cpu().tolist()

        # Snapshot forward operands now; pair with grad_output on backward.
        x_snap = x.detach()
        w_snap = w.detach()

        step = self._step_idx

        def _grad_hook(grad_output: torch.Tensor):
            if grad_output is None:
                return
            try:
                records = build_expert_records(
                    x_snap, w_snap, grad_output, offs, self._detect
                )
                records = self._relabel_global(fqn, records)
                bucket = self._results[fqn].setdefault(step, {})
                bucket[gemm_name] = records
            except Exception as exc:
                logger.warning(
                    f"MoEMatmulPatternObserverModifier: grad capture skipped for "
                    f"{fqn}/{gemm_name} ({exc})"
                )

        if out.requires_grad:
            out.register_hook(_grad_hook)

    # ------------------------------------------------------------------
    # sharding / ids
    # ------------------------------------------------------------------

    def _read_mesh_info(self, module: Module) -> dict:
        """Derive EP/FSDP rank+size from the experts' DTensor weight mesh."""
        info = {"ep_rank": 0, "ep_size": 1, "fsdp_rank": 0, "fsdp_size": 1}
        w = getattr(module, "mlp1_weight", None)
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(w, DTensor):
                mesh = w.device_mesh
                names = mesh.mesh_dim_names or ()
                for dim_name, keys in (("ep", ("ep_rank", "ep_size")),
                                       ("dp_shard", ("fsdp_rank", "fsdp_size")),
                                       ("efsdp", ("fsdp_rank", "fsdp_size"))):
                    if dim_name in names:
                        idx = names.index(dim_name)
                        info[keys[0]] = mesh.get_local_rank(dim_name)
                        info[keys[1]] = mesh.size(idx)
        except Exception as exc:
            logger.debug(f"MoEMatmulPatternObserverModifier: no mesh info ({exc})")
        return info

    def _relabel_global(self, fqn: str, records: dict) -> dict:
        """Map local expert ids to global ids using EP rank/size."""
        mi = self._mesh_info.get(fqn, {})
        ep_rank = mi.get("ep_rank", 0)
        n_local = len(records)
        base = ep_rank * n_local
        return {base + local_id: rec for local_id, rec in records.items()}

    # ------------------------------------------------------------------
    # teardown + dump
    # ------------------------------------------------------------------

    def _detach(self) -> None:
        if self._orig_grouped_mm is not None:
            torch._grouped_mm = self._orig_grouped_mm
            self._orig_grouped_mm = None
        for fqn, orig in self._orig_forwards.items():
            # best-effort restore; the module object still lives in model_parts
            pass
        self._detached = True

    def _dump(self) -> None:
        rank = int(os.environ.get("RANK", 0))
        path = Path(self.output_path)
        # always rank-suffix so multi-rank runs don't clobber a single file.
        path = path.with_stem(f"{path.stem}_rank{rank}")
        path.parent.mkdir(parents=True, exist_ok=True)

        captured_steps = sorted({
            step for data in self._results.values() for step in data
        })

        layer_shapes = {fqn: self._mesh_info.get(fqn, {}) for fqn in self._fqns}

        blob: Dict[str, Any] = dict(self._results)
        num_local = 0
        for fqn in self._fqns:
            for step_data in self._results[fqn].values():
                for gemm_records in step_data.values():
                    num_local = max(num_local, len(gemm_records))

        # Majority-vote each expert's per-path T-pair across all captured steps:
        # _majority[fqn][gemm][global_expert_id] = {n_steps, n_tokens_total,
        #   forward_y/backward_gx/backward_gw: {pair, votes, n}}
        majority: Dict[str, Dict[str, Any]] = {}
        for fqn in self._fqns:
            per_gemm_steps: Dict[str, List[dict]] = {}
            for step_data in self._results[fqn].values():
                for gemm, records in step_data.items():
                    per_gemm_steps.setdefault(gemm, []).append(records)
            majority[fqn] = {
                gemm: accumulate_majority(step_list)
                for gemm, step_list in per_gemm_steps.items()
            }
        blob["_majority"] = majority

        # global expert count = local * ep_size (uniform across ranks)
        any_mi = next(iter(self._mesh_info.values()), {}) if self._mesh_info else {}
        ep_size = any_mi.get("ep_size", 1)
        blob["_meta"] = {
            "rank": rank,
            "ep_rank": any_mi.get("ep_rank", 0),
            "ep_size": ep_size,
            "fsdp_rank": any_mi.get("fsdp_rank", 0),
            "fsdp_size": any_mi.get("fsdp_size", 1),
            "num_local_experts": num_local,
            "num_global_experts": num_local * ep_size,
            "iterations_captured": captured_steps,
            "mesh_info": layer_shapes,
            "gemm_order": list(_GEMM_NAMES),
            "matmul_paths": ["forward_y", "backward_gx", "backward_gw"],
        }
        torch.save(blob, path)
        logger.info(
            f"MoEMatmulPatternObserverModifier: saved {len(captured_steps)} step(s) "
            f"for {len(self._fqns)} layer(s) to {path}"
        )
