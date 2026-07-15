# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Diagnostic: does the AdaHOP calibration blob restore identically on every rank?

Hypothesis under test
---------------------
On resume, adahop jobs deadlock on the first training step inside the MoE
``ALLTOALL_BASE`` collective. The suspected cause is that the checkpointed
calibration state -- a plain pickled Python blob restored by DCP
(alto/modifiers/lpt/adahop_internals/calibration_state.py) -- is NOT delivered
identically to all ranks. If rank 0 sees ``completed=True`` (and takes the
"apply modes, skip calibration" branch) while some other rank sees
``completed=False`` (and takes ``_arm_calibration()``), the two branches issue
different collectives and the expert-parallel all-to-all hangs forever. Raising
the NCCL timeout cannot fix a control-flow divergence.

This script reproduces ONLY the calibration-state load path -- no model, no
training, no MoE -- so it finishes in seconds instead of hanging for 30 min.
Every rank loads the blob, then all ranks all-gather (completed, step_idx,
n_modes, modes_hash) and rank 0 reports whether they agree.

Run with torchrun so every rank participates, and DO NOT filter rank output:

    cd /home/ybouquet/projects/ALTO
    CKPT=./gpt_oss_20b-pretrain-subset-mxfp4-adahop-srfix-mi300x-outputs/checkpoint/step-500 \
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        scripts/check_adahop_resume_divergence.py

Each rank also writes /tmp/adahop_divergence_rank<N>.txt so results survive any
output filtering.
"""

import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def _modes_hash(modes_by_fqn) -> str:
    """Deterministic hash of the per-layer modes dict (order-independent)."""
    blob = json.dumps(modes_by_fqn, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ckpt = os.environ.get("CKPT")
    if not ckpt:
        print("ERROR: set CKPT=<path to checkpoint/step-N dir>", file=sys.stderr)
        return 2
    if not os.path.isdir(ckpt):
        print(f"ERROR: CKPT is not a directory: {ckpt}", file=sys.stderr)
        return 2

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # RCCL/NCCL share the "nccl" backend name on ROCm.
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Import AFTER process group init so any module-level state is fresh.
    from alto.modifiers.lpt.adahop_internals.calibration_state import (
        get_adahop_calibration_state,
        is_calibration_completed,
        get_calibration_modes,
        reset_calibration_state,
    )

    # Start every rank from the fresh-run default so what we observe after the
    # load is purely what DCP delivered -- not leftover state.
    reset_calibration_state()

    before_completed = is_calibration_completed()

    # Reproduce EXACTLY the training resume path: register the Stateful manager
    # under the same key the trainer uses ("adahop_calibration") and let DCP
    # populate the module-level _STATE via load_state_dict.
    manager = get_adahop_calibration_state()
    state = {"adahop_calibration": manager}

    load_error = ""
    try:
        dcp.load(state, checkpoint_id=ckpt)
    except Exception as e:  # noqa: BLE001 -- we want to see per-rank failures
        load_error = f"{type(e).__name__}: {e}"

    from alto.modifiers.lpt.adahop_internals.calibration_state import get_calibration_state_dict
    completed = is_calibration_completed()
    modes = get_calibration_modes()
    n_modes = len(modes)
    step_idx = get_calibration_state_dict().get("step_idx", 0)
    h = _modes_hash(modes) if modes else "<empty>"

    # Which training branch WOULD this rank take on step 501?
    branch = "APPLY_MODES(skip calib)" if completed else "ARM_CALIBRATION"

    line = (f"[rank {rank}/{world}] before_completed={before_completed} "
            f"after_completed={completed} step_idx={step_idx} "
            f"n_modes={n_modes} modes_hash={h} branch={branch} "
            f"load_error={load_error or '<none>'}")

    # Persist per-rank so nothing is lost to output filtering. Default to /tmp;
    # set DIVERGENCE_OUT_DIR to a bind-mounted dir to keep files after the
    # container is torn down.
    out_dir = os.environ.get("DIVERGENCE_OUT_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"adahop_divergence_rank{rank}.txt"), "w") as f:
        f.write(line + "\n")
    print(line, flush=True)

    # Gather a compact tuple from every rank to rank 0 for a verdict.
    record = {
        "rank": rank,
        "completed": bool(completed),
        "step_idx": int(step_idx),
        "n_modes": int(n_modes),
        "modes_hash": h,
        "branch": branch,
        "load_error": load_error,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, record)

    rc = 0
    if rank == 0:
        print("\n==================== VERDICT ====================", flush=True)
        for r in sorted(gathered, key=lambda x: x["rank"]):
            print(f"  rank {r['rank']}: completed={r['completed']} "
                  f"step_idx={r['step_idx']} n_modes={r['n_modes']} hash={r['modes_hash']} "
                  f"branch={r['branch']} err={r['load_error'] or '<none>'}",
                  flush=True)

        completed_set = {r["completed"] for r in gathered}
        hash_set = {r["modes_hash"] for r in gathered}
        branch_set = {r["branch"] for r in gathered}
        err_ranks = [r["rank"] for r in gathered if r["load_error"]]

        print("\n  ---- analysis ----", flush=True)
        if err_ranks:
            print(f"  ✗ LOAD FAILED on ranks {err_ranks} -- blob not restorable there.", flush=True)
            rc = 1
        if len(branch_set) > 1:
            print(f"  ✗ DIVERGENCE CONFIRMED: ranks disagree on the step-501 branch "
                  f"{sorted(branch_set)}. This is the deadlock: different ranks issue "
                  f"different collectives.", flush=True)
            rc = 1
        elif len(completed_set) > 1:
            print(f"  ✗ DIVERGENCE: 'completed' flag differs across ranks {completed_set}.", flush=True)
            rc = 1
        elif len(hash_set) > 1:
            print(f"  ⚠ modes agree on branch but DIFFER in content across ranks "
                  f"{sorted(hash_set)} -- would not deadlock on control flow but numerics "
                  f"diverge per rank.", flush=True)
            rc = 1
        else:
            print(f"  ✓ ALL RANKS AGREE: completed={completed_set.pop()}, "
                  f"identical modes_hash, same branch. Calibration-state restore is NOT "
                  f"the divergence source -- look elsewhere (e.g. Hadamard seed, "
                  f"or a genuinely different collective ordering).", flush=True)
        print("=================================================", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
