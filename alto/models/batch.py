# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Batch sizes and row packing for TorchTitan token microbatches."""

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from torchtitan.components.data import ConcatThenSplitPackingConfig
from torchtitan.components.data.dataset import DatasetConfig, GrainDataset, TextSequence
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


def set_batch_sizes(
    training,
    *,
    seq_len: int,
    local_batch_size: int,
    global_batch_size: int | None = None,
) -> None:
    """Set token budgets from sequence counts.

    ``local_batch_size`` is the number of sequences in one microbatch on one
    data-parallel rank. ``global_batch_size`` is the number of sequences in one
    optimizer step across data-parallel ranks and gradient accumulation. Omit
    it to leave the trainer's default global token count.
    """
    if seq_len <= 0 or local_batch_size <= 0:
        raise ValueError("seq_len and local_batch_size must be positive")
    training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len
    if global_batch_size is not None:
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        training.num_tokens_per_train_step = global_batch_size * seq_len


def _row_positions(sequence: TextSequence, *, max_context_length: int) -> TextSequence:
    positions = np.arange(len(sequence.input_ids), dtype=np.int64) % max_context_length
    return replace(sequence, positions=positions)


@dataclass(frozen=True, kw_only=True, slots=True)
class RowPositionsPackingConfig:
    """Concat-then-split packing with positions counted from each row start.

    ``ConcatThenSplitPackingConfig`` restarts positions at every document, which
    matches flex/varlen document masking. SDPA attends causally across documents
    within each ``max_context_length`` row, so RoPE positions count from the row
    start instead, like Megatron's ``reset_position_ids=False``.
    """

    dataset: DatasetConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        if context.num_tokens_per_microbatch % context.max_context_length != 0:
            raise ValueError(
                "row positions require num_tokens_per_microbatch "
                f"({context.num_tokens_per_microbatch}) to be a multiple of "
                f"max_context_length ({context.max_context_length})"
            )
        packed = ConcatThenSplitPackingConfig(dataset=self.dataset).build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        return packed.map(
            partial(_row_positions, max_context_length=context.max_context_length)
        )
