# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Iterable, Any
from contextlib import contextmanager
import time
import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer as TitanTrainer
from torchtitan.experiments.forge.example_train import Trainer as ForgeTrainer, main as forge_main
from torchtitan.components.metrics import MetricsProcessor
from alto.components.converter import ModelOptConverter
from torchtitan.tools.logging import logger


def log_calibration(
    metrics_processor: MetricsProcessor,
    micro_step: int,
    extra_metrics: dict[str, Any] | None = None,
):
    time_delta = time.perf_counter() - metrics_processor.time_last_log

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

    color = metrics_processor.color
    logger.info(f"{color.orange}calibration micro_step: {micro_step:2}  "
                f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%){color.reset}")
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
    time_delta = time.perf_counter() - metrics_processor.time_last_log

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

    color = metrics_processor.color
    logger.info(f"{color.red}stage2 optimization micro_step: {micro_step:2}  "
                f"{color.green}student_loss: {student_loss:7.4f}  "
                f"{color.green}aggregate_loss: {aggregate_loss:7.4f}  "
                f"{color.blue}lr: {lr:7.4f}  "
                f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                f"({device_mem_stats.max_reserved_pct:.2f}%){color.reset}")

    metrics_processor.time_last_log = time.perf_counter()
    metrics_processor.device_memory_monitor.reset_peak_stats()


class Trainer(ForgeTrainer):

    def __init__(self, config: TitanTrainer.Config):
        super().__init__(config)

        self.training_mode = True
        self.enable_data_cache = False

        self._input_cache = []
        self._output_cache = []

        if not self.model_converters.is_empty() and any(
                isinstance(converter, ModelOptConverter) for converter in self.model_converters.converters):
            converter = next(
                converter for converter in self.model_converters.converters if isinstance(converter, ModelOptConverter))

            if converter.requires_training_mode:
                logger.info("training mode enabled")
                self.training_mode = True
            else:
                logger.info("training mode disabled")
                self.training_mode = False

            if converter.requires_replay_buffer:
                logger.info("data replay buffer enabled")
                self.enable_data_cache = True
            else:
                logger.info("data replay buffer disabled")
                self.enable_data_cache = False

    def cache_input(self, microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]]):
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

    @contextmanager
    def pp_no_loss_function(self, pp_schedule):
        loss_fn = pp_schedule._loss_fn
        has_backward = pp_schedule._has_backward
        pp_schedule._loss_fn = None
        pp_schedule._has_backward = False
        yield
        pp_schedule._loss_fn = loss_fn
        pp_schedule._has_backward = has_backward

    def pp_forward_step(
        self,
        *,
        input_dict_mbs: list[dict[str, torch.Tensor]],
        label_mbs: list[torch.Tensor],
        global_valid_tokens: float,
    ) -> torch.Tensor:
        arg_mbs: list[tuple[torch.Tensor, ...]] = []
        kwarg_mbs: list[dict[str, Any]] = []
        target_mbs = None
        losses = None
        for input_dict, labels in zip(input_dict_mbs, label_mbs, strict=True):
            inputs, labels, extra_kwargs = self.post_dataloading_process(
                input_dict, labels
            )
            if self.pp_has_first_stage:
                arg_mbs.append((inputs,))
            kwarg_mbs.append(extra_kwargs)

        loss_kwargs = {"global_valid_tokens": global_valid_tokens}
        with self.train_context():
            with self.pp_no_loss_function(self.pp_schedule):
                result = self.pp_schedule.step(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs,
                    losses=losses,
                    loss_kwargs=loss_kwargs,
                    return_outputs=self.pp_has_last_stage,
                )

        return result if self.pp_has_last_stage else None

    def forward_step(
        self,
        input_dict: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
        labels: torch.Tensor | list[torch.Tensor],
        global_valid_tokens: float,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            assert isinstance(input_dict, list)
            assert isinstance(labels, list)
            return self.pp_forward_backward_step(
                input_dict_mbs=input_dict,
                label_mbs=labels,
                global_valid_tokens=global_valid_tokens,
            )

        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)
        inputs, labels, extra_kwargs = self.post_dataloading_process(input_dict, labels)

        # Non-PP forward / backward
        with self.train_context():
            assert len(model_parts) == 1
            result = model_parts[0](inputs, **extra_kwargs)

        return result

    def train_step(
        self,
        data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ):
        if self.training_mode:
            return super().train_step(data_iterator)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        assert not parallel_dims.dp_cp_enabled, "DP_CP is not supported in post-training"

        # Collect all microbatches on CPU and count total valid tokens
        # All groups form one optimizer step; each group feeds one fwd-bwd call.
        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pipeline_parallel_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    input_dict, labels = next(data_iterator)
                local_valid_tokens += (labels != IGNORE_INDEX).sum()
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)

        self.cache_input(microbatch_groups)

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        with torch.no_grad():
            for _microbatch, microbatches in enumerate(microbatch_groups):
                input_dict_mbs = []
                label_mbs = []
                for input_dict, labels in microbatches:
                    for key, value in input_dict.items():
                        if isinstance(value, torch.Tensor):
                            input_dict[key] = value.to(self.device)
                    input_dict_mbs.append(input_dict)
                    label_mbs.append(labels.to(self.device))

                if parallel_dims.pp_enabled:
                    fwd_bwd_input_dict = input_dict_mbs
                    fwd_bwd_labels = label_mbs
                else:
                    assert len(input_dict_mbs) == len(label_mbs) == 1
                    fwd_bwd_input_dict = input_dict_mbs[0]
                    fwd_bwd_labels = label_mbs[0]
                
                result = self.forward_step(
                    input_dict=fwd_bwd_input_dict,
                    labels=fwd_bwd_labels,
                    global_valid_tokens=global_valid_tokens,
                )
                self.cache_output(result.detach().cpu())

                # log metrics
                if not self.metrics_processor.should_log(_microbatch):
                    continue

                log_calibration(self.metrics_processor, _microbatch)

        post_step_kwargs = {
            "forward_step": self.forward_step,
            "input_iterator": self.get_cached_input(),
            "output_iterator": self.get_cached_output(),
            "metrics_processor": self.metrics_processor,
            "log_function": log_stage2_optimization,
            "is_last_step": not self.should_continue_training(),
        }
        self.model_converters.post_optimizer_hook(
            self.model_parts,
            **post_step_kwargs,
        )
        self.clear_cached_input()
        self.clear_cached_output()


if __name__ == "__main__":
    forge_main(Trainer)
