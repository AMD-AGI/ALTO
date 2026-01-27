from dataclasses import dataclass, field


@dataclass
class ModelOptimizerConfig:
    recipe: str = ""
    """
    Path to the model optimizer recipe file.
    """


@dataclass
class JobConfig:
    modeloptimizer: ModelOptimizerConfig = field(
        default_factory=ModelOptimizerConfig)
    """
    Model optimizer configuration.
    """
