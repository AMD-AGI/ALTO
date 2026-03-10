from typing import Literal

import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from pydantic import PrivateAttr
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers import Modifier
from modeloptimizer.kernels.mxfp4.mxfp_linear import MXFP4Linear

__all__ = ["LowPrecisionTrainingModifier"]


class LowPrecisionTrainingModifier(Modifier):

    precision: Literal["mxfp4_1d2d"] = "mxfp4_1d2d"
    targets: list[str] = ["Linear"]
    ignore: list[str] = ["output"]

    _extra_kwargs: PrivateAttr = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.precision.startswith("mxfp4_"), "Only MXFP4 precision is supported for now."

        if self.precision.endswith("_1d1d"):
            self._extra_kwargs = {
                "use_2dblock_x": False,
                "use_2dblock_w": False,
            }
        elif self.precision.endswith("_1d2d"):
            self._extra_kwargs = {
                "use_2dblock_x": False,
                "use_2dblock_w": True,
            }
        elif self.precision.endswith("_2d2d"):
            self._extra_kwargs = {
                "use_2dblock_x": True,
                "use_2dblock_w": True,
            }
        else:
            raise ValueError(f"Unknown MXFP4 recipe: {self.precision}")

    @property
    def requires_training_mode(self) -> bool:
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        for name, module in match_named_modules(model, self.targets, self.ignore):
            if isinstance(module, torch.nn.Linear):
                lp_module = MXFP4Linear.from_float(module, **self._extra_kwargs)
                model.set_submodule(name, lp_module, strict=True)
            else:
                raise ValueError(f"Unsupported module type: {type(module)}")

        logger.info(f"LowPrecisionTrainingModifier converted model: {model}")
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
