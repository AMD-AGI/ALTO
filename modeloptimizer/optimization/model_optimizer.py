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
        data_iterator: Optional[Iterable[tuple[dict[str, torch.Tensor],
                                               torch.Tensor]]] = None,
    ):
        # TODO: add PP support
        assert len(model_parts
                  ) == 1, "BlockwiseOptimizer does not support PP currently"
        self.model = model_parts[0]
        self.job_config = job_config
        self.forward_step = forward_step
        self.data_free = data_iterator is None
        self.data_iterator = data_iterator

        self.blocks = self.model.layers
        self.num_blocks = len(self.blocks)
        self.optimized = False

    def optimize(self):
        self.befor_optimize_backbone()
        for i in range(len(self.blocks)):
            block_idx = str(i)
            logger.info(f'\nblock index: {block_idx}/{len(self.blocks)} '
                        f'\nblock: {self.blocks[block_idx]}')
            self.befor_optimize_block(self.blocks[block_idx])
            self.optimize_block(self.blocks[block_idx], block_idx)
            self.after_optimize_block(self.blocks[block_idx])
        self.after_optimize_backbone()
        self.optimized = True

    def cache_input_hook(self, module, inputs, outputs, name, input_feat_dict,
                         output_feat_dict):
        module_inputs = [i.detach().cpu() for i in inputs]
        module_outputs = [o.detach().cpu() for o in outputs]
        if len(module_inputs) == 1:
            inp = module_inputs[0]
            out = module_outputs[0]
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
                out = out.unsqueeze(0)
            input_feat_dict[name].append(inp)
            output_feat_dict[name].append(out)
        else:
            input_feat_dict[name].append(tuple(module_inputs))
            output_feat_dict[name].append(tuple(module_outputs))

    def register_hooks(self, modules, input_feat, output_feat):
        handles = []
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
