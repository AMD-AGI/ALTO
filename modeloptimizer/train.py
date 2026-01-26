from typing import Iterable, Any
from contextlib import contextmanager
import time
import torch
from torchtitan.distributed import utils as dist_utils
from torchtitan.train import Trainer as TorchTitanTrainer, main as torchtitan_main
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import JobConfig
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
        f"{color.orange}calibration step: {micro_step:2}  "
        f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)  "
        f"{color.blue}tps: {round(tps):,}{color.reset}")

    metrics_processor.ntokens_since_last_log = 0
    metrics_processor.time_last_log = time.perf_counter()
    metrics_processor.device_memory_monitor.reset_peak_stats()

class Trainer(TorchTitanTrainer):

    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        self.post_training = True

    @contextmanager
    def pp_no_loss_function(self, pp_schedule):
        loss_fn = pp_schedule._loss_fn
        has_backward = pp_schedule._has_backward
        pp_schedule._loss_fn = None
        pp_schedule._has_backward = False
        yield
        pp_schedule._loss_fn = loss_fn
        pp_schedule._has_backward = has_backward

    @torch.no_grad()
    def forward_step(self, input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, _, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, None)
        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        cp_buffers = [inputs]
        cp_seq_dims = [1]
        if hasattr(model_parts[0], "freqs_cis"):
            cp_buffers += [m.freqs_cis for m in model_parts]
            cp_seq_dims += [0 for _ in model_parts]

        optional_context_parallel_ctx = (dist_utils.create_context_parallel_ctx(
            cp_mesh=parallel_dims.world_mesh["cp"],
            cp_buffers=cp_buffers,
            cp_seq_dims=cp_seq_dims,
            cp_no_restore_buffers={inputs},
            cp_rotate_method=self.job_config.parallelism.
            context_parallel_rotate_method,
        ) if parallel_dims.cp_enabled else None)

        if parallel_dims.pp_enabled:
            targets, losses = None, None
            with self.train_context(optional_context_parallel_ctx):
                with self.pp_no_loss_function(self.pp_schedule):
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        return

    def train_step(self, data_iterator: Iterable[tuple[dict[str, torch.Tensor],
                                                       torch.Tensor]]):
        if not self.post_training:
            return super().train_step(data_iterator)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            self.forward_step(input_dict)

            # log metrics
            if not self.metrics_processor.should_log(_microbatch):
                return

            assert not parallel_dims.dp_cp_enabled, "DP CP is not supported in post-training"

            global_ntokens_seen = self.ntokens_seen

            extra_metrics = {
                "n_tokens_seen": global_ntokens_seen,
            }
            log_calibration(self.metrics_processor, _microbatch, extra_metrics=extra_metrics)

        self.model_converters.post_optimizer_hook(self.model_parts)



    def post_training_tasks(self):

        # TODO: save optimized model
        # self.checkpointer.save(
        #     self.step,
        #     last_step=(self.step >= self.job_config.training.steps),
        # )
        # run validation
        if (self.job_config.validation.enable and
                self.validator.should_validate(self.step)):
            with self.loss_fn.no_rescale():
                self.validator.validate(self.model_parts, self.step)


if __name__ == "__main__":
    torchtitan_main(Trainer)
