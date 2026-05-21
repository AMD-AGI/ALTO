#!/usr/bin/env python3
"""Generic NVFP4 end-to-end run launcher for Phase B mitigation sweeps.

Reads:
    PROBE_RECIPE     : absolute path to a LowPrecisionTrainingModifier yaml.
                       The string ``"none"`` (case-insensitive) is a sentinel
                       that selects the pure BF16 baseline -- no model
                       converters are attached and the model runs in its
                       declared dtype.  Used by E12a apples-to-apples init
                       sweeps where BF16 / MXFP4 / NVFP4 must share the same
                       launcher (and therefore the same default seed and same
                       initialization codepath).
    PROBE_RUN_NAME   : short tag used to suffix dump_folder.

Invokes ``alto.train`` on the GPT-OSS-20B pretrain config with the recipe
attached via ``model_converters``.  Other CLI flags (``--training.steps``,
``--checkpoint.no-enable`` etc.) are forwarded by torchrun's argv.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _maybe_enable_probe_debug() -> None:
    """Opt-in crash diagnostics for long-horizon runs.

    Activated by ``PROBE_DEBUG=1``.  Designed so that NEITHER the kernel /
    autograd / collective code paths nor any numeric output changes; only
    extra diagnostic artifacts are emitted.  Off by default → byte-identical
    behaviour with prior runs.

    What it does (each rank, lazily, before any torch import):
      1. Per-rank ``faulthandler`` writing to a dedicated log file.  Captures
         the Python stack of all threads on SIGSEGV / SIGABRT / SIGFPE /
         SIGBUS / SIGILL — exactly the signals that previously vanished
         without a traceback.  Also dumps every 120s so a hung rank shows
         where it is stuck.  Also registers SIGTERM so collateral kills (the
         7 ranks torchrun shuts down after the first abort) leave a record
         too.
      2. NCCL flight recorder + desync debug + per-rank ``NCCL_DEBUG=WARN``
         file.  Zero overhead in the happy path; on async errors NCCL dumps
         the last few thousand collective records to disk.
      3. ``TORCH_SHOW_CPP_STACKTRACES`` so any C++ throw carries a real
         backtrace.

    Explicitly NOT enabled (would slow training):
      - ``CUDA_LAUNCH_BLOCKING`` / ``HIP_LAUNCH_BLOCKING``
      - ``NCCL_DEBUG=INFO|TRACE``
      - ``TORCH_DISTRIBUTED_DEBUG=DETAIL``
    """
    if os.environ.get("PROBE_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    default_dir = (
        f"/wekafs/hanwang2/workspace/alto_runs/orchestrator_logs/"
        f"probe_debug_{os.environ.get('PROBE_RUN_NAME', 'nvfp4_e2e')}"
    )
    debug_dir = os.environ.get("PROBE_DEBUG_DIR", default_dir)
    nccl_dir = os.path.join(debug_dir, "nccl")
    fh_dir = os.path.join(debug_dir, "faulthandler")
    os.makedirs(nccl_dir, exist_ok=True)
    os.makedirs(fh_dir, exist_ok=True)

    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault(
        "NCCL_DEBUG_FILE", os.path.join(nccl_dir, f"rank{rank}_pid%p.log")
    )
    os.environ.setdefault("AMD_LOG_LEVEL", "2")

    import faulthandler
    import signal

    fh_path = os.path.join(fh_dir, f"rank{rank}.log")
    fh_file = open(fh_path, "w", buffering=1)
    fh_file.write(
        f"[probe_debug] rank={rank} pid={os.getpid()} cwd={os.getcwd()}\n"
        f"[probe_debug] NCCL_DEBUG={os.environ.get('NCCL_DEBUG')} "
        f"TORCH_NCCL_TRACE_BUFFER_SIZE="
        f"{os.environ.get('TORCH_NCCL_TRACE_BUFFER_SIZE')}\n"
    )
    faulthandler.enable(file=fh_file, all_threads=True)
    faulthandler.dump_traceback_later(120, repeat=True, file=fh_file, exit=False)
    for sig_name in ("SIGTERM", "SIGUSR1", "SIGUSR2"):
        if hasattr(signal, sig_name):
            faulthandler.register(
                getattr(signal, sig_name),
                file=fh_file,
                all_threads=True,
                chain=True,
            )
    print(
        f"[probe_debug] rank={rank} faulthandler→{fh_path} "
        f"nccl_log_dir={nccl_dir}",
        flush=True,
    )


_maybe_enable_probe_debug()


from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.gpt_oss import config_registry as titan_gpt_oss_configs

from alto.components.converter import ModelOptConverter
from alto.models.gpt_oss import config_registry as gpt_oss_configs


RECIPE = os.environ.get("PROBE_RECIPE")
RUN_NAME = os.environ.get("PROBE_RUN_NAME", "nvfp4_e2e")
_BF16_SENTINELS = {"", "none", "bf16", "baseline"}
USE_BF16_BASELINE = RECIPE is None or RECIPE.strip().lower() in _BF16_SENTINELS

if not USE_BF16_BASELINE:
    assert Path(RECIPE).is_file(), (
        f"PROBE_RECIPE must point at an existing yaml or be one of "
        f"{sorted(_BF16_SENTINELS)} for the BF16 baseline; got {RECIPE!r}"
    )


def gpt_oss_20b_nvfp4_recipe_e2e_lpt():
    config = gpt_oss_configs.gpt_oss_20b_pretrain()
    config.dump_folder = (
        f"{gpt_oss_configs.GPT_OSS_RUN_ROOT}/gpt_oss_20b-{RUN_NAME}"
    )
    if not USE_BF16_BASELINE:
        config.model_converters = ModelConvertersContainer.Config(converters=[
            ModelOptConverter.Config(recipe=RECIPE),
        ],)
    return config


gpt_oss_configs.gpt_oss_20b_nvfp4_recipe_e2e_lpt = (
    gpt_oss_20b_nvfp4_recipe_e2e_lpt
)
if (
    hasattr(gpt_oss_configs, "__all__")
    and "gpt_oss_20b_nvfp4_recipe_e2e_lpt" not in gpt_oss_configs.__all__
):
    gpt_oss_configs.__all__.append("gpt_oss_20b_nvfp4_recipe_e2e_lpt")

titan_gpt_oss_configs.gpt_oss_20b_nvfp4_recipe_e2e_lpt = (
    gpt_oss_20b_nvfp4_recipe_e2e_lpt
)
if (
    hasattr(titan_gpt_oss_configs, "__all__")
    and "gpt_oss_20b_nvfp4_recipe_e2e_lpt" not in titan_gpt_oss_configs.__all__
):
    titan_gpt_oss_configs.__all__.append("gpt_oss_20b_nvfp4_recipe_e2e_lpt")


if __name__ == "__main__":
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    runpy.run_module("alto.train", run_name="__main__", alter_sys=True)
    sys.exit(0)
