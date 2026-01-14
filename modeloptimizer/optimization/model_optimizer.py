from typing import TYPE_CHECKING, Optional, Iterable, Callable
import functools
from abc import ABC, abstractmethod

import torch
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from modeloptimizer.config import JobConfig


class BlockwiseOptimizer(ABC):

    def __init__(
        self,
        job_config: "JobConfig",
        model_parts: list[torch.nn.Module],
        forward_step: Callable,
        input_batch: Optional[Iterable[dict[str, torch.Tensor]]] = None,
    ):
        self.model_parts = model_parts
        self.job_config = job_config
        self.forward_step = forward_step
        self.data_free = input_batch is None
        self.input_batch = input_batch

        self.num_blocks = self.model_parts[0].n_layers
        self.optimized = False

    def get_block(self, block_idx: str) -> torch.nn.Module | None:
        assert isinstance(block_idx, str), "block_idx must be a string"
        for model_part in self.model_parts:
            if block_idx in model_part.layers:
                return model_part.layers[block_idx]
        return None

    def optimize(self):
        self.befor_optimize_backbone()
        for block_idx in range(self.num_blocks):
            block_idx = str(block_idx)
            block_module = self.get_block(block_idx)
            logger.info(f'\nblock index: {block_idx} / {self.num_blocks}'
                        f'\nblock: {block_module}')
            self.befor_optimize_block(block_module)
            self.optimize_block(block_module, block_idx)
            self.after_optimize_block(block_module)
        self.after_optimize_backbone()
        self.optimized = True

    def cache_input_hook(self, module, inputs, outputs, name, input_feat_dict,
                         output_feat_dict):
        module_inputs = [i.detach() for i in inputs]
        module_outputs = [o.detach() for o in outputs]
        if len(module_inputs) == 1:
            inp = module_inputs[0]
            out = module_outputs[0]
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
                out = out.unsqueeze(0)
            input_feat_dict[name].append(inp)
            output_feat_dict[name].append(out)
        else:
            input_feat_dict[name].append(tuple(module_inputs))
            output_feat_dict[name].append(tuple(module_outputs))

    def register_hooks(self, modules, input_feat, output_feat):
        handles = []
        if modules is None:
            return handles
        if not self.data_free:
            for name in modules:
                handles.append(modules[name].register_forward_hook(
                    functools.partial(self.cache_input_hook,
                                      name=name,
                                      input_feat_dict=input_feat,
                                      output_feat_dict=output_feat)))
        return handles

    @abstractmethod
    def optimize_block(self, block, block_idx):
        pass

    def befor_optimize_backbone(self):
        pass

    def after_optimize_backbone(self):
        pass

    def befor_optimize_block(self, block):
        pass

    def after_optimize_block(self, block):
        pass

    def save_optimization_metadata(self):
        pass

    def save_optimized_model(self):
        pass

    def save_transformed_model(self):
        pass
