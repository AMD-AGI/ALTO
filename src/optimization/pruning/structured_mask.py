"""Unified structured pruning mask for LLMs.

Hierarchy (coarse → fine):
    Level 0 (global):    hidden_mask          — width pruning across entire model
    Level 1 (layer):     layer_mask           — remove entire transformer layers
    Level 2 (sublayer):  layers.{i}.self_attn — remove attention sublayer
                         layers.{i}.mlp       — remove MLP sublayer
    Level 3 (component): layers.{i}.self_attn.head_mask  — per-head pruning
                         layers.{i}.mlp.neuron_mask      — per-neuron pruning

Override rule: if a coarser level marks a unit as pruned, finer masks are ignored.
Convention:    True = pruned (removed),  False = kept.
Sparsity:      only keys that are actually needed appear in the dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ── Key constants ──────────────────────────────────────────────────────────

HIDDEN_MASK = "hidden_mask"
LAYER_MASK = "layer_mask"
ATTN_SUBLAYER_FMT = "layers.{}.self_attn"
MLP_SUBLAYER_FMT = "layers.{}.mlp"
HEAD_MASK_FMT = "layers.{}.self_attn.head_mask"
NEURON_MASK_FMT = "layers.{}.mlp.neuron_mask"

# Legacy key formats (for backward compatibility)
_LEGACY_HEAD_KEY_FMT = "layers.{}.self_attn.o_proj"
_LEGACY_NEURON_KEY_FMT = "layers.{}.mlp.down_proj"


@dataclass
class ModelDims:
    """Architectural dimensions needed to interpret the mask."""

    num_layers: int
    num_kv_heads: int
    num_q_heads: int
    intermediate_size: int
    hidden_size: int
    head_dim: int = 0

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_q_heads

    @property
    def kv_groups(self) -> int:
        return self.num_q_heads // self.num_kv_heads


class StructuredPruningMask:
    """Unified mask representation for all LLM structured pruning patterns.

    Supports arbitrary combinations of:
        - Layer dropping
        - Sublayer dropping  (attention / MLP)
        - Attention head pruning  (per-layer, non-uniform)
        - MLP neuron pruning      (per-layer, non-uniform)
        - Hidden-dimension (width) pruning
    """

    def __init__(self, dims: ModelDims):
        self.dims = dims
        self._mask: Dict[str, torch.Tensor] = {}

    # ── property ───────────────────────────────────────────────────────────

    @property
    def mask(self) -> Dict[str, torch.Tensor]:
        return self._mask

    # ══════════════════════════════════════════════════════════════════════
    #  Setters
    # ══════════════════════════════════════════════════════════════════════

    # Level 0 – width
    def set_hidden_mask(self, mask: torch.Tensor):
        assert mask.shape == (self.dims.hidden_size,)
        self._mask[HIDDEN_MASK] = mask.bool()

    # Level 1 – layer
    def set_layer_mask(self, mask: torch.Tensor):
        assert mask.shape == (self.dims.num_layers,)
        self._mask[LAYER_MASK] = mask.bool()

    def set_layer_pruned(self, layer_idx: int, pruned: bool = True):
        if LAYER_MASK not in self._mask:
            self._mask[LAYER_MASK] = torch.zeros(self.dims.num_layers, dtype=torch.bool)
        self._mask[LAYER_MASK][layer_idx] = pruned

    # Level 2 – sublayer
    def set_attn_sublayer_pruned(self, layer_idx: int, pruned: bool = True):
        self._mask[ATTN_SUBLAYER_FMT.format(layer_idx)] = torch.tensor(pruned, dtype=torch.bool)

    def set_mlp_sublayer_pruned(self, layer_idx: int, pruned: bool = True):
        self._mask[MLP_SUBLAYER_FMT.format(layer_idx)] = torch.tensor(pruned, dtype=torch.bool)

    # Level 3 – head / neuron
    def set_head_mask(self, layer_idx: int, mask: torch.Tensor):
        assert mask.shape == (self.dims.num_kv_heads,)
        self._mask[HEAD_MASK_FMT.format(layer_idx)] = mask.bool()

    def set_neuron_mask(self, layer_idx: int, mask: torch.Tensor):
        assert mask.shape == (self.dims.intermediate_size,)
        self._mask[NEURON_MASK_FMT.format(layer_idx)] = mask.bool()

    # ══════════════════════════════════════════════════════════════════════
    #  Queries  (with hierarchy resolution)
    # ══════════════════════════════════════════════════════════════════════

    def is_layer_pruned(self, layer_idx: int) -> bool:
        m = self._mask.get(LAYER_MASK)
        return m is not None and m[layer_idx].item()

    def is_attn_pruned(self, layer_idx: int) -> bool:
        """True if entire attn is gone (by layer or sublayer mask)."""
        if self.is_layer_pruned(layer_idx):
            return True
        m = self._mask.get(ATTN_SUBLAYER_FMT.format(layer_idx))
        return m is not None and m.item()

    def is_mlp_pruned(self, layer_idx: int) -> bool:
        if self.is_layer_pruned(layer_idx):
            return True
        m = self._mask.get(MLP_SUBLAYER_FMT.format(layer_idx))
        return m is not None and m.item()

    def get_head_mask(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Returns [num_kv_heads] bool mask, or None if no head-level pruning."""
        if self.is_attn_pruned(layer_idx):
            return torch.ones(self.dims.num_kv_heads, dtype=torch.bool)
        return self._mask.get(HEAD_MASK_FMT.format(layer_idx))

    def get_neuron_mask(self, layer_idx: int) -> Optional[torch.Tensor]:
        if self.is_mlp_pruned(layer_idx):
            return torch.ones(self.dims.intermediate_size, dtype=torch.bool)
        return self._mask.get(NEURON_MASK_FMT.format(layer_idx))

    def get_hidden_mask(self) -> Optional[torch.Tensor]:
        return self._mask.get(HIDDEN_MASK)

    # ── Derived index helpers ──────────────────────────────────────────────

    def get_kept_head_indices(self, layer_idx: int) -> Optional[torch.Tensor]:
        """KV-head indices to keep. Returns None if no head pruning."""
        mask = self.get_head_mask(layer_idx)
        if mask is None:
            return None
        return (~mask).nonzero(as_tuple=False).flatten().long()

    def get_kept_q_head_indices(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Q-head indices to keep (expanded from KV heads for GQA)."""
        kv_keep = self.get_kept_head_indices(layer_idx)
        if kv_keep is None:
            return None
        g = self.dims.kv_groups
        return torch.cat([torch.arange(h * g, (h + 1) * g) for h in kv_keep.tolist()]).long()

    def get_kept_neuron_indices(self, layer_idx: int) -> Optional[torch.Tensor]:
        mask = self.get_neuron_mask(layer_idx)
        if mask is None:
            return None
        return (~mask).nonzero(as_tuple=False).flatten().long()

    def get_kept_hidden_indices(self) -> Optional[torch.Tensor]:
        mask = self.get_hidden_mask()
        if mask is None:
            return None
        return (~mask).nonzero(as_tuple=False).flatten().long()

    # ══════════════════════════════════════════════════════════════════════
    #  Statistics
    # ══════════════════════════════════════════════════════════════════════

    def summary(self) -> Dict:
        d = self.dims
        stats = {
            "num_layers_total": d.num_layers,
            "num_layers_pruned": 0,
            "sublayer_pruned": {"attn": [], "mlp": []},
            "head_pruning": {},
            "neuron_pruning": {},
            "hidden_pruned": 0,
        }

        for i in range(d.num_layers):
            if self.is_layer_pruned(i):
                stats["num_layers_pruned"] += 1
                continue
            if self.is_attn_pruned(i):
                stats["sublayer_pruned"]["attn"].append(i)
            elif HEAD_MASK_FMT.format(i) in self._mask:
                n = self._mask[HEAD_MASK_FMT.format(i)].sum().item()
                stats["head_pruning"][i] = f"{int(n)}/{d.num_kv_heads}"
            if self.is_mlp_pruned(i):
                stats["sublayer_pruned"]["mlp"].append(i)
            elif NEURON_MASK_FMT.format(i) in self._mask:
                n = self._mask[NEURON_MASK_FMT.format(i)].sum().item()
                stats["neuron_pruning"][i] = f"{int(n)}/{d.intermediate_size}"

        if HIDDEN_MASK in self._mask:
            stats["hidden_pruned"] = int(self._mask[HIDDEN_MASK].sum().item())

        return stats

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self, path: str):
        data = {
            "dims": {
                "num_layers": self.dims.num_layers,
                "num_kv_heads": self.dims.num_kv_heads,
                "num_q_heads": self.dims.num_q_heads,
                "intermediate_size": self.dims.intermediate_size,
                "hidden_size": self.dims.hidden_size,
            },
            "mask": {k: v.detach().cpu() for k, v in self._mask.items()},
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(data, path)

    @classmethod
    def load(cls, path: str) -> "StructuredPruningMask":
        data = torch.load(path, map_location="cpu")
        dims = ModelDims(**data["dims"])
        obj = cls(dims)
        obj._mask = data["mask"]
        return obj

    # ══════════════════════════════════════════════════════════════════════
    #  Legacy format conversion
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_legacy(
        cls,
        legacy_mask: Dict[str, torch.Tensor],
        dims: ModelDims,
        prune_layer: bool = False,
        prune_attn: bool = False,
        prune_mlp: bool = False,
    ) -> "StructuredPruningMask":
        """Convert old-style W_mask (keyed by weight name) to the new format."""
        obj = cls(dims)
        for i in range(dims.num_layers):
            if prune_attn:
                legacy_key = _LEGACY_HEAD_KEY_FMT.format(i)
                if legacy_key in legacy_mask:
                    obj.set_head_mask(i, legacy_mask[legacy_key])
            if prune_mlp:
                legacy_key = _LEGACY_NEURON_KEY_FMT.format(i)
                if legacy_key in legacy_mask:
                    obj.set_neuron_mask(i, legacy_mask[legacy_key])
        return obj

    def to_legacy(self) -> Dict[str, torch.Tensor]:
        """Export to old-style W_mask for backward compatibility."""
        out: Dict[str, torch.Tensor] = {}
        for i in range(self.dims.num_layers):
            hm = self.get_head_mask(i)
            if hm is not None:
                out[_LEGACY_HEAD_KEY_FMT.format(i)] = hm
            nm = self.get_neuron_mask(i)
            if nm is not None:
                out[_LEGACY_NEURON_KEY_FMT.format(i)] = nm
        return out

    # ══════════════════════════════════════════════════════════════════════
    #  Construct from sparsity_dict  (search / MOO output)
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def from_sparsity_dict(
        cls,
        sparsity_dict: Dict[str, float],
        dims: ModelDims,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "StructuredPruningMask":
        """Build mask from per-layer sparsity ratios and optional importance scores.

        Args:
            sparsity_dict: e.g. {"layers.0.self_attn.o_proj": 0.375,
                                  "layers.0.mlp.down_proj": 0.25, ...}
            dims: model dimensions
            importance_scores: per-component importance for ranking which units
                               to prune.  Same key convention as sparsity_dict,
                               values are 1-D tensors (higher = more important).
                               If None, prunes the last N units (placeholder).
        """
        obj = cls(dims)
        for key, sparsity in sparsity_dict.items():
            parts = key.split(".")
            layer_idx = int(parts[1])

            if sparsity >= 1.0:
                if "self_attn" in key:
                    obj.set_attn_sublayer_pruned(layer_idx)
                elif "mlp" in key:
                    obj.set_mlp_sublayer_pruned(layer_idx)
                continue

            if sparsity <= 0.0:
                continue

            if "o_proj" in key or "self_attn" in key:
                n_prune = int(dims.num_kv_heads * sparsity)
                mask = torch.zeros(dims.num_kv_heads, dtype=torch.bool)
                if importance_scores and key in importance_scores:
                    _, idx = importance_scores[key].topk(n_prune, largest=False)
                    mask[idx] = True
                else:
                    mask[-n_prune:] = True
                obj.set_head_mask(layer_idx, mask)

            elif "down_proj" in key or "mlp" in key:
                n_prune = int(dims.intermediate_size * sparsity)
                mask = torch.zeros(dims.intermediate_size, dtype=torch.bool)
                if importance_scores and key in importance_scores:
                    _, idx = importance_scores[key].topk(n_prune, largest=False)
                    mask[idx] = True
                else:
                    mask[-n_prune:] = True
                obj.set_neuron_mask(layer_idx, mask)

        return obj
