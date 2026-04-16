# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/modifier.py

from abc import abstractmethod
from pydantic import ConfigDict
import torch
from torch.nn import Module
from torchtitan.tools.logging import logger
from alto.modifiers.utils import HooksMixin

__all__ = ["Modifier"]


class Modifier(HooksMixin):

    model_config = ConfigDict(extra="forbid")

    index: int | None = None
    group: str | None = None
    start: float | None = None
    end: float | None = None
    update: float | None = None

    initialized_: bool = False
    finalized_: bool = False
    started_: bool = False
    ended_: bool = False

    @property
    def initialized(self) -> bool:
        return self.initialized_

    @property
    def finalized(self) -> bool:
        return self.finalized_

    @property
    def requires_replay_buffer(self) -> bool:
        return False

    @property
    def requires_training_mode(self) -> bool:
        return False

    def convert(self, model: Module, **kwargs) -> bool:
        return self.on_convert(model, **kwargs)

    def initialize(self, model_parts: list[Module], **kwargs):
        if self.initialized_:
            raise RuntimeError("Cannot initialize a modifier that has already been initialized")
        if self.finalized_:
            raise RuntimeError("Cannot initialize a modifier that has already been finalized")
        assert isinstance(model_parts, list), "model_parts must be a list of nn.Module"

        self.initialized_ = self.on_initialize(model_parts, **kwargs)

    def finalize(self, model_parts: list[Module], **kwargs):
        if self.finalized_:
            raise RuntimeError("cannot finalize a modifier twice")
        if not self.initialized_:
            raise RuntimeError("cannot finalize an uninitialized modifier")
        assert isinstance(model_parts, list), "model_parts must be a list of nn.Module"

        # TODO: all finalization should succeed
        self.finalized_ = self.on_finalize(model_parts, **kwargs)

    def pre_step(self, model_parts: list[Module], **kwargs):
        if not self.initialized_ or self.finalized_:
            raise RuntimeError("cannot call pre-step method on an uninitialized or finalized modifier")
        self.started_ = self.on_pre_step(model_parts, **kwargs)

    def post_step(self, model_parts: list[Module], **kwargs):
        if not self.initialized_ or self.finalized_:
            raise RuntimeError("cannot call post-step method on an uninitialized or finalized modifier")
        self.ended_ = self.on_post_step(model_parts, **kwargs)

    @abstractmethod
    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_convert(self, model: Module, **kwargs) -> bool:
        raise NotImplementedError

    def _infer_sequential_targets(self, model: torch.nn.Module) -> str | list[str]:
        match self.sequential_targets:
            case None:
                block_class_name = next(iter(model.layers.values())).__class__.__name__
                logger.info(f"Inferred sequential targets: {block_class_name}")
                return [block_class_name]
            case str():
                return [self.sequential_targets]
            case _:
                return self.sequential_targets
