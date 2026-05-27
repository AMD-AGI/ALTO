#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Capture real-model activation bundles for op-level NVFP4 / MXFP4 SNR tests.

Why this script exists
======================
The op-level SNR tests under ``tests/unittest/{nvfp4,mxfp4}/`` currently
feed synthetic Gaussian + sparse-outlier tensors into the FP4 paths.
That hides three real concerns that only show up on production inputs:

* Transformer activations have a much heavier tail than Gaussian, which
  stresses the per-block ``amax`` scale encoding (E8M0 for MX vs E4M3
  for NV) very differently from synthetic data.
* MoE expert dispatch is severely skewed -- some experts see almost no
  tokens and others see clumps -- so per-expert grouped-GEMM blocks
  behave nothing like uniform random.
* On large-K + stochastic-rounding configurations we have an open
  question whether NVFP4 inner-only dX really collapses on real
  distributions or only on synthetic noise.  The synthetic answer is
  "yes"; the real-data answer is unknown.

This script answers all three by running GPT-OSS-20B forward (BF16,
**no LpT conversion**) over a few c4 batches and dumping the inputs /
weights / outputs of six representative layers to disk, where the test
side can replay them through NVFP4 vs MXFP4 against the same BF16
reference.  Output format and on-disk layout are pinned in
``alto.kernels.fp4.real_data_utils`` (single source of truth shared
with the consumer tests).

Forward-only design
-------------------
This first version is *forward only* on purpose:

* The node currently hosting GPT-OSS-20B has an amdgpu PM-firmware bug
  that throttles SCLK to ~3 MHz under load (~65x slower than spec).  A
  full forward+backward of the 20B model at that throttle is not
  feasible; forward-only halves the wall-clock and avoids the autograd
  graph entirely, making the capture survivable on the throttled node.
* Real ``dy`` is the *least* important real-data signal for op-level
  SNR: the NV / MX kernel difference dominates ``dy`` distribution
  effects in practice, and forward-only still nails the heavy-tail
  ``x`` distribution that we actually want to test.
* We still want to be able to compare dX / dW against a BF16 reference
  *somehow*, so the script stores three synthetic ``dy`` strategies per
  bundle.  The consumer test picks one (or iterates) and computes its
  own BF16 dx_ref / dw_ref inside the test, keeping the comparison
  apples-to-apples regardless of which dy is used.

When the node is healthy, pass ``--with-backward`` to additionally
register a backward hook and store the *real* ``dy`` under the
``dy_real`` key; the consumer fixture transparently prefers ``dy_real``
when present.

Entry point
-----------
Run via ``torchrun`` (single GPU is fine; matches the ``gpt_oss_20b``
config's ``TP=1, EP=1``).  Dump-specific options are passed as
environment variables to avoid fighting ``ConfigManager``:

::

    ALTO_DUMP_OUT=/wekafs/zhitwang/test_data/real_activations/step5000_ckpt_20260526 \\
    ALTO_DUMP_TAGS=dense_early,dense_mid,dense_late,grouped_early,grouped_mid,grouped_late \\
    ALTO_DUMP_NUM_STEPS=3 \\
    ALTO_DUMP_WITH_BACKWARD=0 \\
    HSA_NO_SCRATCH_RECLAIM=1 \\
    torchrun --nproc_per_node=1 -m scripts.dump_real_test_activations \\
        --module gpt_oss --config gpt_oss_20b \\
        --training.local_batch_size 2 --training.seq_len 2048 \\
        --training.steps 0 --validator.enable false

Use the wrapper ``scripts/dump_real_test_activations.sh`` to iterate
over multiple batch / seq shapes back-to-back without re-loading the
model.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchtitan.experiments.forge.example_train import main as forge_main
from torchtitan.tools.logging import logger

from alto.train import Trainer as AltoTrainer
from alto.kernels.fp4 import real_data_utils as rdu


# ---------------------------------------------------------------------------
# Synthetic ``dy`` strategies
# ---------------------------------------------------------------------------
#
# Stored alongside ``y`` so the consumer test can pick the strategy that
# best matches the comparison it wants to make:
#
#   * ``dy_normal_std_y`` -- ``N(0, std(y))``.  Closest match to a real
#     upstream gradient in *scale*, while staying Gaussian in shape.
#     This is the recommended default for "compare NV vs MX dX/dW SNR
#     under the same noisy upstream".
#
#   * ``dy_ones`` -- ``ones_like(y)``.  Trivial uniform gradient; isolates
#     the kernel's accumulation behaviour from any dy distribution effect.
#     Useful as a sanity check that the backward kernel paths don't
#     diverge under degenerate inputs.
#
# When ``--with-backward`` is set, a third key ``dy_real`` is added with
# the genuine BF16 ``grad_output`` captured from a full backward pass;
# tests prefer it when present.

def _make_dy_normal_std_y(y: torch.Tensor, seed: int) -> torch.Tensor:
    """``N(0, std(y))`` with a deterministic seed so reruns are stable."""
    g = torch.Generator(device=y.device).manual_seed(seed)
    std = y.float().std().clamp(min=1e-8).to(y.dtype)
    return torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype) * std


def _make_dy_ones(y: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(y)


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------
#
# Per layer-tag we keep an in-memory entry containing the captured ``x``,
# ``y`` (and ``dy_real`` if backward is enabled), plus the weight tensors
# we need to snapshot once.  We snapshot weights *every step* so the
# bundle is self-contained -- the test side does not need a separate
# weight file or a manifest cross-reference.

class _LayerCapture:
    """Owns the hook handles and the per-step captured tensors for one tag."""

    def __init__(self, tag: str, module: nn.Module, with_backward: bool):
        self.tag = tag
        self.module = module
        self.kind = rdu.kind_for_tag(tag)
        self.with_backward = with_backward

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._x: torch.Tensor | None = None
        self._num_tokens_per_expert: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._dy_real: torch.Tensor | None = None

    # -- registration -------------------------------------------------------

    def register(self) -> None:
        self._handles.append(self.module.register_forward_pre_hook(self._pre))
        self._handles.append(self.module.register_forward_hook(self._fwd))
        if self.with_backward:
            self._handles.append(
                self.module.register_full_backward_hook(self._bwd)
            )

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # -- hook callbacks -----------------------------------------------------

    def _pre(self, module, args):
        # Dense (nn.Linear):       args = (x,)
        # Grouped (GptOssGroupedExperts):  args = (x, num_tokens_per_expert)
        # We .detach().clone() because the upstream caller may free or
        # in-place-modify the tensor before save time.
        if self.kind.is_dense:
            assert len(args) >= 1
            x = args[0]
            self._x = x.detach().clone()
        else:
            assert len(args) >= 2, (
                f"grouped capture expected (x, num_tokens_per_expert); got {len(args)} args"
            )
            self._x = args[0].detach().clone()
            self._num_tokens_per_expert = args[1].detach().clone()

    def _fwd(self, module, args, output):
        self._y = output.detach().clone()

    def _bwd(self, module, grad_input, grad_output):
        # grad_output is a tuple matching the forward output.  For both
        # capture kinds the output is a single Tensor.
        if grad_output and grad_output[0] is not None:
            self._dy_real = grad_output[0].detach().clone()

    # -- bundle assembly ----------------------------------------------------

    def snapshot_weights(self) -> dict[str, torch.Tensor]:
        """Return a fresh on-CPU copy of the weight tensors we need."""
        if self.kind.is_dense:
            # nn.Linear.weight is the canonical dense weight; bias is not
            # quantised by our NVFP4 / MXFP4 linear paths so we drop it.
            return {rdu.KEY_W: self.module.weight.detach().to("cpu").clone()}
        # Grouped: snapshot both grouped-mm weights so the test can
        # exercise either mlp1 or mlp2 GEMM independently.  bias is
        # similarly not quantised, dropped.
        return {
            rdu.KEY_MLP1_W: self.module.mlp1_weight.detach().to("cpu").clone(),
            rdu.KEY_MLP2_W: self.module.mlp2_weight.detach().to("cpu").clone(),
        }

    def consume_bundle(
        self,
        *,
        step: int,
        bs: int,
        seq: int,
        dy_seed: int,
        run_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a self-contained bundle and clear the per-step captures.

        Synthesised ``dy_*`` keys are built on-GPU (cheap on the y shape)
        and then moved to CPU together with x / y, so the resulting
        bundle is fully CPU-resident and safe to ``torch.save`` to weka.
        """
        if self._x is None or self._y is None:
            raise RuntimeError(
                f"layer tag {self.tag!r} produced no capture this step; "
                f"the model forward may have skipped this module"
            )

        # Generate dy variants on whatever device y currently lives on,
        # then move everything to CPU in one go.
        dy_normal = _make_dy_normal_std_y(self._y, seed=dy_seed)
        dy_ones = _make_dy_ones(self._y)

        bundle: dict[str, Any] = {
            rdu.KEY_X: self._x.to("cpu"),
            rdu.KEY_Y: self._y.to("cpu"),
            rdu.KEY_DY_NORMAL_STD_Y: dy_normal.to("cpu"),
            rdu.KEY_DY_ONES: dy_ones.to("cpu"),
        }
        if self.kind.is_grouped:
            assert self._num_tokens_per_expert is not None
            bundle[rdu.KEY_NUM_TOKENS_PER_EXPERT] = (
                self._num_tokens_per_expert.to("cpu")
            )
        bundle.update(self.snapshot_weights())
        if self.with_backward and self._dy_real is not None:
            bundle[rdu.KEY_DY_REAL] = self._dy_real.to("cpu")

        bundle[rdu.KEY_METADATA] = {
            **run_metadata,
            "layer_tag": self.tag,
            "module_path": rdu.LAYER_TAG_TO_PATH[self.tag],
            "kind": self.kind.name,
            "shape_x": list(self._x.shape),
            "shape_y": list(self._y.shape),
            "bs": bs,
            "seq": seq,
            "step": step,
            "has_dy_real": rdu.KEY_DY_REAL in bundle,
        }

        # Clear so the next step starts from a clean slate (and so we
        # do not silently reuse stale data if a hook misfires).
        self._x = None
        self._y = None
        self._dy_real = None
        self._num_tokens_per_expert = None
        return bundle


# ---------------------------------------------------------------------------
# DumpTrainer
# ---------------------------------------------------------------------------

class DumpTrainer(AltoTrainer):
    """ForgeTrainer subclass that replaces ``train()`` with a dump loop.

    Re-uses every Trainer service we want (model build, HF ckpt load,
    dataloader, attention-mask creation, parallelism setup) and only
    overrides ``train()`` so we control the per-step lifecycle.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dump_out = Path(os.environ["ALTO_DUMP_OUT"])
        tags_env = os.environ.get("ALTO_DUMP_TAGS", ",".join(rdu.ALL_TAGS))
        self.dump_tags: list[str] = [t.strip() for t in tags_env.split(",") if t.strip()]
        for t in self.dump_tags:
            if t not in rdu.LAYER_TAG_TO_PATH:
                raise ValueError(
                    f"ALTO_DUMP_TAGS contains unknown tag {t!r}; "
                    f"known tags: {list(rdu.LAYER_TAG_TO_PATH)}"
                )

        self.dump_num_steps = int(os.environ.get("ALTO_DUMP_NUM_STEPS", "3"))
        self.dump_with_backward = (
            os.environ.get("ALTO_DUMP_WITH_BACKWARD", "0") == "1"
        )
        self.dump_seed = int(os.environ.get("ALTO_DUMP_SEED", "1234"))

        # Captures and (bs, seq) snapshot for filename construction.
        self._captures: dict[str, _LayerCapture] = {}
        self._bs = config.training.local_batch_size
        self._seq = config.training.seq_len

    # ---- module resolution -------------------------------------------------

    def _resolve_modules(self) -> dict[str, nn.Module]:
        """Map each requested layer tag to a module in the live model.

        Raises a clear error if any tag does not match a module; we
        prefer hard failure here over silently skipping a layer.
        """
        assert len(self.model_parts) == 1, (
            "DumpTrainer requires PP=1 (single model part) so module paths "
            "line up with the unsharded model namespace"
        )
        model = self.model_parts[0]
        named = dict(model.named_modules())
        resolved: dict[str, nn.Module] = {}
        for tag in self.dump_tags:
            path = rdu.LAYER_TAG_TO_PATH[tag]
            # ModuleDict child indexing renders as ``layers.5.moe.experts``
            # which is what named_modules() emits, so an exact match is safe.
            if path not in named:
                raise KeyError(
                    f"layer tag {tag!r} -> path {path!r} not found in model; "
                    f"available top-level layer paths sample: "
                    f"{[p for p in list(named)[:10] if '.layers.' in p or p.startswith('layers.')]}"
                )
            resolved[tag] = named[path]
        return resolved

    # ---- bundle persistence -----------------------------------------------

    def _run_metadata(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "model_name": cfg.model_spec.name,
            "model_flavor": cfg.model_spec.flavor,
            "hf_assets_path": getattr(cfg, "hf_assets_path", None),
            "training_seq_len": cfg.training.seq_len,
            "training_local_batch_size": cfg.training.local_batch_size,
            "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
            "with_backward": self.dump_with_backward,
            "torch_version": torch.__version__,
        }

    def _write_manifest(self, run_metadata: dict[str, Any]) -> None:
        """Write or merge ``manifest.json`` for the dump root.

        We *merge* rather than overwrite so successive script invocations
        with different ``--training.local_batch_size`` / ``--training.seq_len``
        can accumulate into the same run_tag directory without trampling
        each other's records.
        """
        mpath = rdu.manifest_path(self.dump_out)
        run_tag = self.dump_out.name
        new_entry = {
            "schema_version": rdu.SCHEMA_VERSION,
            "run_tag": run_tag,
            "model_sha": run_metadata.get("hf_assets_path") or "unknown",
            "captured_at": run_metadata["captured_at"],
            "layers": [
                {
                    "tag": t,
                    "module_path": rdu.LAYER_TAG_TO_PATH[t],
                    "kind": rdu.kind_for_tag(t).name,
                }
                for t in self.dump_tags
            ],
            "captures": [],
        }
        if mpath.exists():
            try:
                existing = json.loads(mpath.read_text())
            except json.JSONDecodeError:
                existing = None
            if existing and existing.get("schema_version") == rdu.SCHEMA_VERSION:
                new_entry["captures"] = existing.get("captures", [])
                # Preserve the original captured_at as the run's first time.
                new_entry["captured_at"] = existing.get(
                    "captured_at", new_entry["captured_at"]
                )

        new_entry["captures"].append({
            "bs": self._bs,
            "seq": self._seq,
            "num_steps": self.dump_num_steps,
            "tags": list(self.dump_tags),
            "with_backward": self.dump_with_backward,
            "captured_at": run_metadata["captured_at"],
        })
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(new_entry, indent=2))
        logger.info(f"[dump] wrote manifest {mpath}")

    def _save_step_bundles(self, step: int, run_metadata: dict[str, Any]) -> None:
        for tag, cap in self._captures.items():
            bundle = cap.consume_bundle(
                step=step,
                bs=self._bs,
                seq=self._seq,
                dy_seed=self.dump_seed + step * 17 + hash(tag) % 1000,
                run_metadata=run_metadata,
            )
            out_dir = self.dump_out / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / rdu.bundle_filename(self._bs, self._seq, step)
            torch.save(bundle, out_path)
            logger.info(
                f"[dump] saved {tag} step={step} -> {out_path} "
                f"(x={list(bundle[rdu.KEY_X].shape)}, "
                f"y={list(bundle[rdu.KEY_Y].shape)})"
            )

    # ---- dump loop --------------------------------------------------------

    def train(self):  # type: ignore[override]
        """Forward-only capture loop; replaces the standard training loop."""
        # Only rank 0 writes bundles + manifest; other ranks just run forward.
        is_writer = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        # Hooks must be registered *after* checkpoint load so the model
        # weights are real, not the initial random ones.
        self.checkpointer.load(step=self.config.checkpoint.load_step)
        logger.info(
            f"[dump] checkpoint loaded; starting forward-only capture "
            f"for {self.dump_num_steps} step(s), tags={self.dump_tags}, "
            f"with_backward={self.dump_with_backward}"
        )

        modules = self._resolve_modules()
        for tag, mod in modules.items():
            cap = _LayerCapture(tag, mod, with_backward=self.dump_with_backward)
            cap.register()
            self._captures[tag] = cap

        run_metadata = self._run_metadata()
        if is_writer:
            self._write_manifest(run_metadata)

        try:
            data_iterator = self.batch_generator(self.dataloader)
            ctx = torch.no_grad() if not self.dump_with_backward else _nullctx()
            with ctx:
                for step in range(self.dump_num_steps):
                    input_dict, labels = next(data_iterator)
                    for k, v in input_dict.items():
                        if isinstance(v, torch.Tensor):
                            input_dict[k] = v.to(self.device)
                    labels = labels.to(self.device)

                    # global_valid_tokens placeholder: only used by training
                    # loss scaling, irrelevant here.
                    global_valid_tokens = torch.tensor(1.0, device=self.device)
                    result = self.forward_step(
                        input_dict, labels, global_valid_tokens
                    )

                    if self.dump_with_backward and result is not None:
                        # Drive a tiny scalar so backward hooks fire; the
                        # absolute scale of dy is recorded as captured.
                        result.float().sum().backward()

                    if is_writer:
                        self._save_step_bundles(step, run_metadata)
                    else:
                        # Non-writer ranks still need to clear captures so
                        # the next step starts fresh.
                        for cap in self._captures.values():
                            cap.consume_bundle(
                                step=step,
                                bs=self._bs,
                                seq=self._seq,
                                dy_seed=0,
                                run_metadata=run_metadata,
                            )
        finally:
            for cap in self._captures.values():
                cap.remove()
            self._captures.clear()

        logger.info("[dump] capture complete")


# ---------------------------------------------------------------------------
# Minimal null context (Python 3.10 stdlib has nullcontext but we keep this
# inline for clarity: when with_backward is True we deliberately do NOT
# wrap forward in no_grad so the backward hook can capture dy).

class _nullctx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    # Re-uses ForgeTrainer's argparse / ConfigManager pipeline; users
    # only need to supply ``--module gpt_oss --config gpt_oss_20b`` plus
    # any training-side overrides they want.
    forge_main(DumpTrainer)
