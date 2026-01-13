from typing import Iterable
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

    def cache_input(self, input_dict: dict[str, torch.Tensor]):
        self._input_cache.append(input_dict)

    def clear_input_cache(self):
        self._input_cache = []

    def get_input_cache_iter(self):
        if not self._input_cache:
            data_iterator = self.batch_generator(self.dataloader)
            for _ in range(self.gradient_accumulation_steps):
                input_dict = next(data_iterator)[0]
                self.cache_input(input_dict)
        return iter(self._input_cache)

    def block_optimization_step(self):
        # TODO: initialize before training loop
        sparsification_config = self.job_config.sparsification
        logger.info(f"Sparsification config: {sparsification_config}")
        if sparsification_config.method != "none":
            sparsification_optimizer = SPARSIFICATION_METHODS[
                self.job_config.sparsification.method](
                    self.job_config,
                    self.model_parts,
                    self.forward_step,
                    self.get_input_cache_iter(),
                )
            sparsification_optimizer.optimize()
        raise NotImplementedError("Calibration is not implemented")

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
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = None, None
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    if self.pp_has_last_stage:
                        pred = self.pp_schedule.step(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                            return_outputs=True,
                        )
                    else:
                        self.pp_schedule.step(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                            return_outputs=False,
                        )
                        pred = None
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs,
                                          **extra_kwargs)

        return pred

    def post_training_tasks(self):
        # # run validation
        # if (self.job_config.validation.enable and
        #         self.validator.should_validate(self.step)):
        #     with self.loss_fn.no_rescale():
        #         self.validator.validate(self.model_parts, self.step)
        self.block_optimization_step()
        self.clear_input_cache()


if __name__ == "__main__":
    torchtitan_main(Trainer)
