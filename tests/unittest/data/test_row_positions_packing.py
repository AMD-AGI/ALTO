# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from grain.python import ReadOptions

from torchtitan.components.data.megatron import megatron_dataset_config
from torchtitan.components.data.megatron.indexed_datasets import _IndexWriter
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy

from alto.models.batch import RowPositionsPackingConfig
from alto.models.llama3 import config_registry


def _write_indexed(prefix: str, sequences: list[np.ndarray]) -> None:
    tokens = np.concatenate(sequences).astype(np.int32)
    with open(prefix + ".bin", "wb") as handle:
        handle.write(tokens.tobytes())
    with _IndexWriter(prefix + ".idx", np.int32) as writer:
        writer.write(
            [len(sequence) for sequence in sequences],
            None,
            list(range(len(sequences) + 1)),
        )


def _context(*, num_tokens: int, context_length: int) -> DatasetBuildContext:
    return DatasetBuildContext(
        tokenizer=None,
        max_context_length=context_length,
        num_tokens_per_microbatch=num_tokens,
        read_options=ReadOptions(),
    )


def _policy() -> DatasetIterationPolicy:
    return DatasetIterationPolicy(
        seed=0,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
        streaming_shuffle_buffer_size=1,
    )


def test_row_positions_ignore_document_boundaries(tmp_path):
    prefix = str(tmp_path / "docs")
    # 5 + 11 input tokens: the first document ends at offset 5 of the first row.
    _write_indexed(
        prefix,
        [np.arange(10, 16, dtype=np.int32), np.arange(20, 32, dtype=np.int32)],
    )
    documents = megatron_dataset_config(prefix + ".idx")
    context = _context(num_tokens=8, context_length=4)
    baseline = list(
        ConcatThenSplitPackingConfig(dataset=documents).build(
            context=context, dataset_iteration_policy=_policy()
        )
    )
    rows = list(
        RowPositionsPackingConfig(dataset=documents).build(
            context=context, dataset_iteration_policy=_policy()
        )
    )

    assert len(rows) == 2
    assert len(baseline) == len(rows)
    mid_row_resets = [
        index
        for row in baseline
        for index in np.flatnonzero(np.asarray(row.positions) == 0)
        if index % 4 != 0
    ]
    assert mid_row_resets
    for row, base in zip(rows, baseline, strict=True):
        np.testing.assert_array_equal(row.positions, np.arange(8) % 4)
        np.testing.assert_array_equal(row.input_ids, base.input_ids)
        np.testing.assert_array_equal(row.labels, base.labels)
        np.testing.assert_array_equal(row.padding_mask, base.padding_mask)


def test_row_positions_reject_partial_rows():
    config = RowPositionsPackingConfig(dataset=megatron_dataset_config("missing.idx"))
    with pytest.raises(ValueError, match="multiple of max_context_length"):
        config.build(
            context=_context(num_tokens=6, context_length=4),
            dataset_iteration_policy=_policy(),
        )


@pytest.mark.parametrize("name", config_registry.__all__)
def test_llama3_sdpa_configs_use_row_positions(name):
    config = getattr(config_registry, name)()
    assert isinstance(config.dataloader.dataset, RowPositionsPackingConfig)
    if config.validator is not None:
        assert isinstance(config.validator.dataloader.dataset, RowPositionsPackingConfig)
