import functools
from abc import ABCMeta, abstractmethod
from loguru import logger


class BlockwiseOptimizer(metaclass=ABCMeta):
    def __init__(self, model, optimization_config, global_config, input):
        self.model = model
        self.blocks = model.get_blocks()
        self.optimization_config = optimization_config
        self.global_config = global_config
        self.input = input
        self.data_free = False if self.input else True
        self.block_idx = None
        self.num_blocks = len(self.blocks)
        self.optimized = False
        if self.input:
            for i in range(len(input['kwargs'])):
                if 'use_cache' in input['kwargs'][i]:
                    input['kwargs'][i].pop('use_cache')
            for i in range(len(input['kwargs'])):
                if 'past_key_value' in input['kwargs'][i]:
                    input['kwargs'][i]['past_key_value'] = None
            self.n_samples = 0
            for i in range(len(input['data'])):
                self.n_samples += input['data'][i].shape[0]

    def optimize(self):
        self.befor_optimize_backbone()
        for i in range(len(self.blocks)):
            self.block_idx = i
            logger.info(
                f'\nblock index: {self.block_idx}/{len(self.blocks)} '
                f'\nblock: {self.blocks[self.block_idx]}'
            )
            self.befor_optimize_block(self.blocks[self.block_idx])
            self.optimize_block(self.blocks[self.block_idx])
            self.after_optimize_block(self.blocks[self.block_idx])
        self.after_optimize_backbone()
        self.optimized = True

    def cache_input_hook(self, module, inputs, outputs, name, input_feat_dict, output_feat_dict):
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
                handles.append(
                    modules[name].register_forward_hook(
                        functools.partial(
                            self.cache_input_hook, 
                            name=name, 
                            input_feat_dict=input_feat,
                            output_feat_dict=output_feat
                        )
                    )
                )
        return handles

    @abstractmethod
    def optimize_block(self, block):
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