"""
Shared utilities for mapping resolution between smooth layers and balance layers.
Used by SmoothQuant and AWQ modifiers to identify which normalization layers
feed into which linear projection layers within each transformer block.
"""

import re
from dataclasses import dataclass, field

from torch.nn import Module
from torchtitan.tools.logging import logger

__all__ = ["LayerMapping", "resolve_mappings"]


@dataclass
class LayerMapping:
    """Resolved mapping between a smooth layer and its downstream balance layers.

    :param smooth_name: fully-qualified name of the smooth layer (e.g. LayerNorm)
    :param smooth_layer: the PyTorch module for the smooth layer
    :param balance_layers: list of downstream weight modules to offset the smoothing
    :param balance_names: corresponding fully-qualified names of balance layers
    """

    smooth_name: str
    smooth_layer: Module
    balance_layers: list[Module] = field(default_factory=list)
    balance_names: list[str] = field(default_factory=list)


def resolve_mappings(
    model: Module,
    mappings: list[dict],
    ignore: list[str] | None = None,
) -> list[LayerMapping]:
    """Resolve mapping specifications into concrete module references.

    Each mapping dict should contain:
        - ``smooth_layer``:   regex or exact name pattern for the smooth layer
        - ``balance_layers``: list of regex/name patterns for balance layers

    Layers are matched within the same transformer block, identified by the
    numeric layer index prefix (e.g. ``layers.0``).

    :param model: model whose named modules are searched
    :param mappings: list of mapping dicts from the recipe
    :param ignore: optional list of name patterns to skip
    :return: list of resolved LayerMapping objects
    """
    ignore = ignore or []
    all_named = list(model.named_modules())
    resolved: list[LayerMapping] = []

    for mapping in mappings:
        smooth_pattern = mapping["smooth_layer"]
        balance_patterns = mapping["balance_layers"]

        for smooth_name, smooth_module in all_named:
            if not _matches(smooth_name, smooth_pattern):
                continue
            if _is_ignored(smooth_name, ignore):
                continue

            prefix = _get_block_prefix(smooth_name)

            balance_modules: list[Module] = []
            balance_names: list[str] = []
            for bname, bmodule in all_named:
                if prefix and not bname.startswith(prefix):
                    continue
                if _is_ignored(bname, ignore):
                    continue
                if not hasattr(bmodule, "weight"):
                    continue
                for bp in balance_patterns:
                    if _matches(bname, bp):
                        balance_modules.append(bmodule)
                        balance_names.append(bname)
                        break

            if not balance_modules:
                logger.warning(
                    f"SmoothQuant/AWQ: no balance layers found for "
                    f"{smooth_name}, skipping"
                )
                continue

            resolved.append(
                LayerMapping(
                    smooth_name=smooth_name,
                    smooth_layer=smooth_module,
                    balance_layers=balance_modules,
                    balance_names=balance_names,
                )
            )

    return resolved


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _matches(name: str, pattern: str) -> bool:
    """Check if *name* matches *pattern* (supports ``re:`` prefix for regex)."""
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return name == pattern


def _is_ignored(name: str, ignore: list[str]) -> bool:
    return any(_matches(name, p) for p in ignore)


def _get_block_prefix(name: str) -> str:
    """Extract the transformer block prefix from a module name.

    Example: ``layers.0.attention_norm`` -> ``layers.0``
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return ""
