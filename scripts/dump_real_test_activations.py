#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""One-shot dump of REAL GPT-OSS-20B activations for op-level SNR tests.

Captures, for a handful of representative layers, the real tensors that flow
through them during a few forward+backward steps on real c4 data:

  * dense attention linears (wq/wk/wv/wo):  x, w, y, dy
  * MoE grouped experts (mlp1 GEMM):        x (routed tokens), w (mlp1), ntpe

These are saved as ``.pt`` bundles under
``/wekafs/zhitwang/test_data/real_activations/<run_tag>/`` and consumed by
``tests/unittest/nvfp4/test_nvfp_real_data.py`` (marker ``real_data``) to
compare NVFP4 vs MXFP4 SNR vs a BF16 reference on real distributions.

DESIGN
------
* Single GPU, BF16, NO low-precision converter, NO TP/EP/FSDP.  We want the
  clean high-precision (x, w, y, dy) reference distributions; the FP4 paths
  are applied later inside the test, not here.  GPT-OSS-20B in BF16 (~40 GB)
  fits on one MI300X (192 GB).
* Loads the trained weights from a DCP checkpoint (resharded TP=4 -> single
  device by DCP).  ==> VERIFY AT RUNTIME: DCP reshard onto a single,
  unparallelized model is the riskiest step; if it fails, fall back to
  running this under the same torchrun parallelism as training.

SAFETY
------
Refuses to run while a training job is active (it would steal GPU compute and
slow the run).  Only run after training has finished / freed the GPUs.

USAGE
-----
  HSA_NO_SCRATCH_RECLAIM=1 python scripts/dump_real_test_activations.py \
      --dump-folder /wekafs/zhitwang/alto_runs/nvfp4_test2_16k_20260524_130342 \
      --ckpt-step latest \
      --module alto.models.gpt_oss --config gpt_oss_20b_lpt \
      --layers 1,12,20 \
      --num-batches 2 \
      --out-root /wekafs/zhitwang/test_data/real_activations
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Safety guard: never run while training is using the GPUs.
# ---------------------------------------------------------------------------

def _assert_no_training_running() -> None:
    try:
        out = subprocess.run(
            ["pgrep", "-af", r"alto\.train|torchrun.*alto"],
            capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:
        out = ""
    live = [ln for ln in out.splitlines()
            if "alto.train" in ln or "torchrun" in ln]
    live = [ln for ln in live if "dump_real_test_activations" not in ln]
    if live:
        print("ERROR: a training/torchrun process appears to be running:",
              file=sys.stderr)
        for ln in live[:5]:
            print("   ", ln, file=sys.stderr)
        print("Refusing to start the dump — it would contend for GPU compute "
              "and slow training.  Run this only after the GPUs are free.",
              file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Model build + checkpoint load.
# ==> The exact API here MUST be verified at runtime (Phase 1).  Isolated in
#     one function so the capture/save logic below stays decoupled from it.
# ---------------------------------------------------------------------------

def build_bf16_model_and_load_ckpt(module: str, config: str,
                                   dump_folder: str, ckpt_step: str):
    """Build a single-device BF16 GPT-OSS-20B and load trained weights.

    Returns the nn.Module (on cuda, eval-ish but with grad enabled) plus a
    small object exposing a real c4 batch iterator.

    VERIFY AT RUNTIME:
      * ConfigManager entry + model_spec.build() signature.
      * Disabling parallelism (TP=EP=FSDP=1) and the model_converters (we
        want a plain BF16 model, not NVFP4-wrapped).
      * DCP reshard of a TP=4-saved checkpoint onto this single-device model.
    """
    from torchtitan.config.manager import ConfigManager  # type: ignore

    cm = ConfigManager()
    cfg = cm.load_config_from_registry(  # VERIFY: exact method name
        module=module, config=config,
    )

    # Force single-device, no low-precision conversion: we want clean BF16.
    cfg.parallelism.tensor_parallel_degree = 1
    cfg.parallelism.expert_parallel_degree = 1
    cfg.parallelism.expert_tensor_parallel_degree = 1
    cfg.model_converters = None          # VERIFY: how to disable converters
    cfg.training.steps = 0

    raise NotImplementedError(
        "build_bf16_model_and_load_ckpt is a runtime-verified stub. "
        "Complete the ConfigManager/model build + DCP load wiring during "
        "Phase 1 on a free GPU (see docstring VERIFY markers). The capture "
        "and save logic below is ready and decoupled from this function."
    )


# ---------------------------------------------------------------------------
# Hook-based capture (robust, standard PyTorch — ready to use).
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Registers fwd/bwd hooks on chosen modules and stores one sample each."""

    def __init__(self, model, dense_names, grouped_names):
        self.dense_names = set(dense_names)
        self.grouped_names = set(grouped_names)
        self.store: dict[str, dict] = {}
        self._handles = []
        name_of = {m: n for n, m in model.named_modules()}

        for name, module in model.named_modules():
            if name in self.dense_names:
                self._handles.append(
                    module.register_forward_hook(self._dense_fwd(name)))
                self._handles.append(
                    module.register_full_backward_hook(self._dense_bwd(name)))
            elif name in self.grouped_names:
                self._handles.append(
                    module.register_forward_hook(self._grouped_fwd(name)))

    def _dense_fwd(self, name):
        def hook(module, inputs, output):
            if name in self.store and "x" in self.store[name]:
                return  # keep only the first sample
            self.store.setdefault(name, {})
            self.store[name].update(
                kind="dense_linear",
                x=inputs[0].detach().to(torch.bfloat16).cpu(),
                w=module.weight.detach().to(torch.bfloat16).cpu(),
                y=output.detach().to(torch.bfloat16).cpu(),
            )
        return hook

    def _dense_bwd(self, name):
        def hook(module, grad_input, grad_output):
            if name in self.store and "dy" in self.store[name]:
                return
            self.store.setdefault(name, {})
            self.store[name]["dy"] = grad_output[0].detach().to(
                torch.bfloat16).cpu()
        return hook

    def _grouped_fwd(self, name):
        def hook(module, inputs, output):
            if name in self.store and "x" in self.store[name]:
                return
            # GptOssGroupedExperts.forward(x, num_tokens_per_expert)
            x = inputs[0]
            ntpe = inputs[1] if len(inputs) > 1 else None
            self.store.setdefault(name, {})
            self.store[name].update(
                kind="grouped_gemm",
                x=x.detach().to(torch.bfloat16).cpu(),
                w=module.mlp1_weight.detach().to(torch.bfloat16).cpu(),
                num_tokens_per_expert=(
                    ntpe.detach().to(torch.int32).cpu()
                    if ntpe is not None else None),
            )
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


def _layer_module_names(layers):
    """Map requested layer indices to (dense_names, grouped_names, tag_map)."""
    dense, grouped, tag_map = [], [], {}
    depth_tag = {min(layers): "early", sorted(layers)[len(layers) // 2]: "mid",
                 max(layers): "late"}
    for li in layers:
        dn = f"layers.{li}.attention.wq"
        gn = f"layers.{li}.moe.experts"
        dense.append(dn)
        grouped.append(gn)
        depth = depth_tag.get(li, f"l{li}")
        tag_map[dn] = f"dense_{depth}"
        tag_map[gn] = f"grouped_{depth}"
    return dense, grouped, tag_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-folder", required=True,
                    help="training run dir containing checkpoint/step-N")
    ap.add_argument("--ckpt-step", default="latest")
    ap.add_argument("--module", default="alto.models.gpt_oss")
    ap.add_argument("--config", default="gpt_oss_20b_lpt")
    ap.add_argument("--layers", default="1,12,20",
                    help="comma-separated layer indices to capture")
    ap.add_argument("--num-batches", type=int, default=2)
    ap.add_argument("--out-root",
                    default="/wekafs/zhitwang/test_data/real_activations")
    args = ap.parse_args()

    _assert_no_training_running()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(2)

    layers = [int(x) for x in args.layers.split(",")]
    dense_names, grouped_names, tag_map = _layer_module_names(layers)

    run_tag = f"step_{args.ckpt_step}_{_dt.datetime.utcnow():%Y%m%d_%H%M%S}"
    out_dir = Path(args.out_root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model, data = build_bf16_model_and_load_ckpt(
        args.module, args.config, args.dump_folder, args.ckpt_step)

    cap = ActivationCapture(model, dense_names, grouped_names)

    model.train()  # enable grad path so backward hooks fire
    for _ in range(args.num_batches):
        batch = data.next_batch()          # VERIFY: real-data iterator API
        loss = model(**batch)              # VERIFY: forward/loss signature
        loss.backward()
        model.zero_grad(set_to_none=True)

    cap.remove()

    # Persist one bundle per captured module.
    manifest = {"run_tag": run_tag, "ckpt_step": args.ckpt_step,
                "module": args.module, "config": args.config,
                "layers": layers, "captured": []}
    for mod_name, bundle in cap.store.items():
        tag = tag_map.get(mod_name, mod_name.replace(".", "_"))
        bundle["metadata"] = {"module_name": mod_name, "run_tag": run_tag,
                              "ckpt_step": args.ckpt_step}
        sub = out_dir / tag
        sub.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, sub / "bundle.pt")
        manifest["captured"].append({"tag": tag, "module": mod_name,
                                     "keys": sorted(bundle.keys())})
        print(f"saved {tag}: {mod_name}  keys={sorted(bundle.keys())}")

    import json
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Maintain a 'latest' symlink for the test fixture default.
    latest = Path(args.out_root) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir, target_is_directory=True)
    except OSError as e:
        print(f"warn: could not update 'latest' symlink: {e}", file=sys.stderr)

    print(f"\nDone. Bundles under {out_dir}")
    print(f"Run tests with: ALTO_REAL_DATA_DIR={out_dir} "
          f"python -m pytest -m real_data -v -s tests/unittest/nvfp4/")


if __name__ == "__main__":
    main()
