from typing import Iterable, Any
from contextlib import contextmanager
import time
import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer as TitanTrainer
from torchtitan.experiments.forge.example_train import Trainer as ForgeTrainer, main as forge_main
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.tools.logging import logger


def log_calibration(
    metrics_processor: MetricsProcessor,
    micro_step: int,
    extra_metrics: dict[str, Any] | None = None,
):
    time_delta = time.perf_counter() - metrics_processor.time_last_log

    device_mem_stats = metrics_processor.device_memory_monitor.get_peak_stats()

    # tokens per second per device, abbreviated as tps
    tps = metrics_processor.ntokens_since_last_log / (
        time_delta * metrics_processor.parallel_dims.non_data_parallel_size)

    metrics = {
        "calibration_metrics/throughput(tps)":
            tps,
        "calibration_metrics/memory/max_active(GiB)":
            device_mem_stats.max_active_gib,
        "calibration_metrics/memory/max_active(%)":
            device_mem_stats.max_active_pct,
        "calibration_metrics/memory/max_reserved(GiB)":
            device_mem_stats.max_reserved_gib,
        "calibration_metrics/memory/max_reserved(%)":
            device_mem_stats.max_reserved_pct,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_processor.logger.log(metrics, micro_step)

    color = metrics_processor.color
    logger.info(
        f"{color.orange}calibration micro_step: {micro_step:2}  "
        f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)  "
        f"{color.blue}tps: {round(tps):,}{color.reset}")

    metrics_processor.ntokens_since_last_log = 0
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

    # tokens per second per device, abbreviated as tps
    tps = metrics_processor.ntokens_since_last_log / (
        time_delta * metrics_processor.parallel_dims.non_data_parallel_size)

    metrics = {
        "stage2_optimization_metrics/student_loss":
            student_loss,
        "stage2_optimization_metrics/aggregate_loss":
            aggregate_loss,
        "stage2_optimization_metrics/lr":
            lr,
        "stage2_optimization_metrics/throughput(tps)":
            tps,
        "stage2_optimization_metrics/memory/max_active(GiB)":
            device_mem_stats.max_active_gib,
        "stage2_optimization_metrics/memory/max_active(%)":
            device_mem_stats.max_active_pct,
        "stage2_optimization_metrics/memory/max_reserved(GiB)":
            device_mem_stats.max_reserved_gib,
        "stage2_optimization_metrics/memory/max_reserved(%)":
            device_mem_stats.max_reserved_pct,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_processor.logger.log(metrics, micro_step)

    color = metrics_processor.color
    logger.info(
        f"{color.red}stage2 optimization micro_step: {micro_step:2}  "
        f"{color.green}student_loss: {student_loss:7.4f}  "
        f"{color.green}aggregate_loss: {aggregate_loss:7.4f}  "
        f"{color.blue}lr: {lr:7.4f}  "
        f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%){color.reset}")

    metrics_processor.ntokens_since_last_log = 0
    metrics_processor.time_last_log = time.perf_counter()
    metrics_processor.device_memory_monitor.reset_peak_stats()


class Trainer(ForgeTrainer):

    def __init__(self, config: TitanTrainer.Config):
        super().__init__(config)

        # Build the collection of model converters. No-op if converters empty
        model_compile_enabled = (config.compile.enable and
                                 "model" in config.compile.components)
        self.model_converters = config.model_converters.build(
            parallel_dims=self.parallel_dims,
            model_compile_enabled=model_compile_enabled,
        )

        self.post_training = True
        self.enable_data_cache = True
        self._input_cache = []
        self._output_cache = []

    def cache_input(self, microbatches: list[tuple[dict[str, torch.Tensor],
                                                   torch.Tensor]]):
        if self.enable_data_cache:
            self._input_cache = microbatches

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

    def forward_step(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, _, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels)

        if parallel_dims.pp_enabled:
            targets, losses = None, None
            result = None
            with self.train_context():
                with self.pp_no_loss_function(self.pp_schedule):
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    elif self.pp_has_last_stage:
                        result = self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                            return_outputs=True,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
        else:
            # Non-PP forward / backward
            with self.train_context():
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    result = model_parts[0](inputs, **extra_inputs,
                                            **extra_kwargs)

        return result

    def train_step(
        self,
        data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ):
        if not self.post_training:
            return super().train_step(data_iterator)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        assert not parallel_dims.dp_cp_enabled, "DP_CP is not supported in post-training"

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        self.cache_input(microbatches)

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens,
                                                      batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        with torch.no_grad():
            for _microbatch, (input_dict, labels) in enumerate(microbatches):
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(self.device)
                labels = labels.to(self.device)

                result = self.forward_step(input_dict, labels,
                                           global_valid_tokens)
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

    def post_training_tasks(self):
        last_step = not self.should_continue_training()
        if last_step:
            self.model_converters.finalize(self.model_parts)

        self.checkpointer.save(
            self.step,
            last_step=last_step,
        )
        # run validation
        if (self.config.validator.enable and
                self.validator.should_validate(self.step)):
            with self.loss_fn.no_rescale():
                self.validator.validate(self.model_parts, self.step)


if __name__ == "__main__":
    forge_main(Trainer)
