from contextlib import contextmanager
import torch
from torchtitan.distributed import utils as dist_utils
from torchtitan.train import Trainer as TorchTitanTrainer, main as torchtitan_main
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

from modeloptimizer.config.registry import SPARSIFICATION_METHODS


class Trainer(TorchTitanTrainer):

    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        # TODO: cache inputs when training
        self._input_cache = []
        self._enable_input_cache = False

        self.sparsification_config = self.job_config.sparsification
        logger.info(f"Sparsification config: {self.sparsification_config}")
        if self.sparsification_config.method != "none":
            self._enable_input_cache = True


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
    def cache_input(self, input_dict: dict[str, torch.Tensor]):
        if self._enable_input_cache:
            self._input_cache.append(input_dict)

    def clear_input_cache(self):
        self._input_cache = []

    def get_input_cache(self):
        if not self._input_cache:
            data_iterator = self.batch_generator(self.dataloader)
            for _ in range(self.gradient_accumulation_steps):
                input_dict = next(data_iterator)[0]
                self.cache_input(input_dict)
        return self._input_cache

    def block_optimization_step(self):
        # TODO: initialize before training loop
        if self.sparsification_config.method != "none":
            sparsification_optimizer = SPARSIFICATION_METHODS[
                self.sparsification_config.method](
                    self.job_config,
                    self.model_parts,
                    self.forward_step,
                    self.get_input_cache(),
                )
            sparsification_optimizer.optimize()

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

    def post_training_tasks(self):

        self.block_optimization_step()
        self.clear_input_cache()
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
