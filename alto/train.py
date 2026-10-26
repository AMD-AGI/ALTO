# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import time
from collections.abc import Iterator
from typing import Any

import torch
from torch.distributed.pipelining.schedules import PipelineScheduleMulti, PipelineScheduleSingle
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer as TitanTrainer

from alto.components.converter import ModelOptConverter

def log_calibration(
    metrics_processor: MetricsProcessor,
    micro_step: int,
    extra_metrics: dict[str, Any] | None = None,
):
    device_mem_stats = metrics_processor.device_memory_monitor.get_peak_stats()

    metrics = {
        "calibration_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
        "calibration_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
        "calibration_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
        "calibration_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_processor.logger.log(metrics, micro_step)

    sl.log_trace_scalar(
        {
            "alto.calibration.micro_step": micro_step,
            "alto.calibration.memory.max_reserved_gib": device_mem_stats.max_reserved_gib,
            "alto.calibration.memory.max_reserved_pct": device_mem_stats.max_reserved_pct,
        }
    )
    metrics_processor.time_last_log = time.perf_counter()
    metrics_processor.device_memory_monitor.reset_peak_stats()


def log_stage2_optimization(
    metrics_processor: MetricsProcessor,
    micro_step: int,
    lr: float,
    student_loss: float,
    aggregate_loss: float,
    extra_metrics: dict[str, Any] | None = None,
):
    device_mem_stats = metrics_processor.device_memory_monitor.get_peak_stats()

    metrics = {
        "stage2_optimization_metrics/student_loss": student_loss,
        "stage2_optimization_metrics/aggregate_loss": aggregate_loss,
        "stage2_optimization_metrics/lr": lr,
        "stage2_optimization_metrics/memory/max_active(GiB)": device_mem_stats.max_active_gib,
        "stage2_optimization_metrics/memory/max_active(%)": device_mem_stats.max_active_pct,
        "stage2_optimization_metrics/memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
        "stage2_optimization_metrics/memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_processor.logger.log(metrics, micro_step)

    sl.log_trace_scalar(
        {
            "alto.stage2.micro_step": micro_step,
            "alto.stage2.student_loss": student_loss,
            "alto.stage2.aggregate_loss": aggregate_loss,
            "alto.stage2.lr": lr,
            "alto.stage2.memory.max_reserved_gib": device_mem_stats.max_reserved_gib,
            "alto.stage2.memory.max_reserved_pct": device_mem_stats.max_reserved_pct,
        }
    )

    metrics_processor.time_last_log = time.perf_counter()
    metrics_processor.device_memory_monitor.reset_peak_stats()


class Trainer(TitanTrainer):
    """Trainer that also runs forward-only calibration for model optimizer recipes.

    Recipes whose modifiers do not require training replace the optimizer step
    with calibration forwards followed by the converters' post-optimizer hook.
    """

    def __init__(self, config: TitanTrainer.Config):
        super().__init__(config)

        self.training_mode = True
        self.enable_data_cache = False

        self._input_cache: list[list[TrainingMicrobatch]] = []
        self._output_cache: list[torch.Tensor] = []

        converter = next(
            (c for c in self.model_converters.converters if isinstance(c, ModelOptConverter)),
            None,
        )
        if converter is not None:
            self.training_mode = converter.requires_training_mode
            sl.log_trace_instant(
                "alto.training_mode.enabled"
                if self.training_mode
                else "alto.training_mode.disabled"
            )
            self.enable_data_cache = converter.requires_replay_buffer
            sl.log_trace_instant(
                "alto.data_replay_buffer.enabled"
                if self.enable_data_cache
                else "alto.data_replay_buffer.disabled"
            )

    def cache_input(self, microbatch_groups: list[list[TrainingMicrobatch]]):
        if self.enable_data_cache:
            self._input_cache = microbatch_groups

    def cache_output(self, output: torch.Tensor):
        if self.enable_data_cache:
            self._output_cache.append(output)

    def get_cached_input(self):
        yield from self._input_cache

    def get_cached_output(self):
        yield from self._output_cache

    def clear_cached_input(self):
        self._input_cache.clear()

    def clear_cached_output(self):
        self._output_cache.clear()

    def forward_step(self, microbatch_group: list[TrainingMicrobatch]) -> torch.Tensor | None:
        """Run a forward-only pass over one microbatch group and return the model output.

        Returns ``None`` on pipeline ranks without the last stage.
        """
        engine = self.engine
        parallelism_context = engine.parallelism_context
        routing_token_counts = torch.stack([mb.routing_token_counts for mb in microbatch_group]).sum(0)
        prepared = engine._preprocess_microbatch_group(
            microbatch_group,
            global_routing_token_counts=routing_token_counts.to(engine.device),
        )

        with parallelism_context.activate_spmd():
            if not parallelism_context.pp_enabled:
                inputs, _labels, model_kwargs, _loss_kwargs = prepared
                model = engine.model_parts[0]
                # Calibration needs logits even when ChunkedLossWrapper owns lm_head.
                skip_lm_head = model._skip_lm_head
                model._skip_lm_head = False
                try:
                    return model(inputs, **model_kwargs)
                finally:
                    model._skip_lm_head = skip_lm_head

            arg_mbs, kwarg_mbs, _target_mbs, _loss_kwarg_mbs = prepared
            pp_schedule = engine.pp_schedule
            outputs = pp_schedule.eval(
                arg_mbs=arg_mbs,
                kwarg_mbs=kwarg_mbs,
                target_mbs=None,
                losses=None,
                return_outputs=engine.pp_has_last_stage,
            )
            # Forward-only runs have no backward pass to release stage outputs.
            if isinstance(pp_schedule, PipelineScheduleSingle):
                pp_schedule._stage.clear_runtime_states()
            elif isinstance(pp_schedule, PipelineScheduleMulti):
                for stage in pp_schedule._stages:
                    stage.clear_runtime_states()
            return outputs if engine.pp_has_last_stage else None

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]):
        if self.training_mode:
            return super().train_step(data_iterator)

        engine = self.engine
        assert not engine.parallelism_context.dp_cp_enabled, "DP_CP is not supported in post-training"

        # All groups form one calibration step; each group feeds one forward call.
        microbatch_groups: list[list[TrainingMicrobatch]] = []
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    microbatch_group.append(next(data_iterator))
            microbatch_groups.append(microbatch_group)

        self.cache_input(microbatch_groups)

        with torch.no_grad():
            for micro_step, microbatch_group in enumerate(microbatch_groups):
                result = self.forward_step(microbatch_group)
                if result is not None:
                    self.cache_output(result.detach().cpu())

                if self.metrics_processor.should_log(micro_step):
                    log_calibration(self.metrics_processor, micro_step)

        engine.num_completed_steps += 1
        post_step_kwargs = {
            "forward_step": self.forward_step,
            "input_iterator": self.get_cached_input(),
            "output_iterator": self.get_cached_output(),
            "metrics_processor": self.metrics_processor,
            "log_function": log_stage2_optimization,
            "is_last_step": not self.should_continue_training(),
        }
        self.model_converters.post_optimizer_hook(engine.model_parts, **post_step_kwargs)
        self.clear_cached_input()
        self.clear_cached_output()


# ConfigLoader options; any other ``--a.b.c VALUE`` argument sets a config field.
_LOADER_OPTIONS = {"--module", "--config", "--override", "--comm-backend", "--output-dir", "--resume-step"}
_LOADER_FLAGS = {"--print-config", "-h", "--help"}


def split_field_overrides(argv: list[str]) -> tuple[list[str], list[tuple[str, str | None]]]:
    """Separate ConfigLoader arguments from ``--dotted.field[=]VALUE`` overrides.

    A field override without a value (followed by another option or nothing)
    sets the field to ``True``.
    """
    loader_args, overrides = [], []
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, sep, value = arg.partition("=")
        if name in _LOADER_OPTIONS or name in _LOADER_FLAGS or not arg.startswith("--"):
            loader_args.append(arg)
            if name in _LOADER_OPTIONS and not sep and i + 1 < len(argv):
                loader_args.append(argv[i + 1])
                i += 1
        elif sep:
            overrides.append((name[2:], value))
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            overrides.append((name[2:], argv[i + 1]))
            i += 1
        else:
            overrides.append((name[2:], None))
        i += 1
    return loader_args, overrides


def apply_field_overrides(config: Any, overrides: list[tuple[str, str | None]]) -> None:
    import yaml

    for path, raw in overrides:
        *parents, field_name = path.replace("-", "_").split(".")
        target = config
        for parent in parents:
            target = getattr(target, parent)
            if target is None:
                raise ValueError(f"Cannot override {path!r}: {parent!r} is None in the selected config")
        if not hasattr(target, field_name):
            raise ValueError(f"Cannot override {path!r}: {type(target).__qualname__} has no field {field_name!r}")
        current = getattr(target, field_name)
        if raw is None:
            value = True
        elif isinstance(current, str):
            value = raw
        else:
            value = yaml.safe_load(raw)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
        setattr(target, field_name, value)
        sl.log_trace_instant(f"alto.config_override.{path}")


def main() -> None:
    """Run the config selected on the command line with ALTO's trainer.

    ``TRAIN_FILE`` selects this entrypoint. The loaded object remains the
    upstream trainer config; it is not copied into another config class.
    """
    import os
    import sys

    import torchtitan
    from torchtitan.config import ConfigLoader
    from torchtitan.observability.logging import init_logger

    init_logger()
    sl.log_trace_message("alto.torchtitan_version", f"torchtitan version: {torchtitan.__version__}")

    loader_args, overrides = split_field_overrides(sys.argv[1:])
    config = ConfigLoader().load(loader_args)
    apply_field_overrides(config, overrides)
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer: Trainer | None = None
    try:
        trainer = Trainer(config)
        if config.create_seed_checkpoint:
            if int(os.environ["WORLD_SIZE"]) != 1:
                raise RuntimeError("Must create seed checkpoint using a single device, to disable sharding.")
            if config.checkpointer is None:
                raise RuntimeError("Must configure checkpointer when creating a seed checkpoint.")
            trainer.engine.save_checkpoint(last_step=True)
            sl.log_trace_instant("alto.seed_checkpoint_created")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        sl.log_trace_instant("alto.process_group_destroyed")


if __name__ == "__main__":
    main()
