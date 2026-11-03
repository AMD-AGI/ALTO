# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Sequence-batch sizes for TorchTitan token budgets."""


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
