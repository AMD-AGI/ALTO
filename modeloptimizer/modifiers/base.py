from abc import abstractmethod
from pydantic import ConfigDict
from torch.nn import Module

from modeloptimizer.modifiers.utils import HooksMixin

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

    def initialize(self, model: Module, **kwargs):
        if self.initialized_:
            raise RuntimeError(
                "Cannot initialize a modifier that has already been initialized"
            )

        if self.finalized_:
            raise RuntimeError(
                "Cannot initialize a modifier that has already been finalized")

        self.initialized_ = self.on_initialize(model, **kwargs)

    def finalize(self, model: Module, **kwargs):
        if self.finalized_:
            raise RuntimeError("cannot finalize a modifier twice")

        if not self.initialized_:
            raise RuntimeError("cannot finalize an uninitialized modifier")

        # TODO: all finalization should succeed
        self.finalized_ = self.on_finalize(model, **kwargs)

    @abstractmethod
    def on_initialize(self, model: Module, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def on_finalize(self, model: Module, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pre_step(self, model: Module, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def post_step(self, model: Module, **kwargs):
        raise NotImplementedError
