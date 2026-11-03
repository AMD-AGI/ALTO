# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch
from grain.python import ReadOptions

from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.megatron import (
    megatron_dataset_config,
    megatron_packed_dataset,
)
from torchtitan.components.data.megatron.dp_split import MegatronDpSplitPackingConfig
from torchtitan.components.data.megatron.indexed_datasets import _IndexWriter
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


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


def _policy(*, rank: int, world: int, shuffle: bool = False, seed: int = 0):
    return DatasetIterationPolicy(
        seed=seed,
        shuffle=shuffle,
        repeat=False,
        dp_rank=rank,
        dp_world_size=world,
        streaming_shuffle_buffer_size=1,
    )


def _rows(dataset):
    return [dataset[index] for index in range(len(dataset))]


def test_megatron_split_stripes_flat_windows_across_files(tmp_path):
    first = [
        np.array([10, 11, 12, 13, 14], dtype=np.int32),
        np.array([20, 21, 22], dtype=np.int32),
    ]
    second = [np.array([30, 31, 32, 33, 34, 35], dtype=np.int32)]
    flat = np.concatenate(first + second)
    _write_indexed(str(tmp_path / "a"), first)
    _write_indexed(str(tmp_path / "b"), second)
    paths = f"{tmp_path / 'a'}.idx,{tmp_path / 'b'}.idx"

    config = megatron_packed_dataset(paths, split="megatron")
    assert isinstance(config, MegatronDpSplitPackingConfig)
    context = _context(num_tokens=4, context_length=4)
    rank0 = _rows(
        config.build(
            context=context, dataset_iteration_policy=_policy(rank=0, world=2)
        )
    )
    rank1 = _rows(
        config.build(
            context=context, dataset_iteration_policy=_policy(rank=1, world=2)
        )
    )

    assert len(rank0) == 1
    assert len(rank1) == 1
    np.testing.assert_array_equal(rank0[0].input_ids, flat[0:4])
    np.testing.assert_array_equal(rank0[0].labels, flat[1:5])
    np.testing.assert_array_equal(rank1[0].input_ids, flat[4:8])
    np.testing.assert_array_equal(rank1[0].labels, flat[5:9])
    np.testing.assert_array_equal(rank0[0].positions, np.arange(4))


def test_megatron_split_restarts_positions_on_context_length(tmp_path):
    prefix = tmp_path / "one"
    _write_indexed(str(prefix), [np.arange(20, dtype=np.int32)])
    config = MegatronDpSplitPackingConfig(
        dataset=megatron_dataset_config(str(prefix) + ".idx")
    )
    rows = _rows(
        config.build(
            context=_context(num_tokens=8, context_length=4),
            dataset_iteration_policy=_policy(rank=0, world=1),
        )
    )
    np.testing.assert_array_equal(rows[0].input_ids, np.arange(8))
    np.testing.assert_array_equal(rows[0].labels, np.arange(1, 9))
    np.testing.assert_array_equal(rows[0].positions, [0, 1, 2, 3, 0, 1, 2, 3])


def test_megatron_split_shuffles_one_shared_order(tmp_path):
    prefix = tmp_path / "shuffle"
    _write_indexed(str(prefix), [np.arange(17, dtype=np.int32)])
    config = MegatronDpSplitPackingConfig(
        dataset=megatron_dataset_config(str(prefix) + ".idx")
    )
    context = _context(num_tokens=4, context_length=4)
    rank0 = _rows(
        config.build(
            context=context,
            dataset_iteration_policy=_policy(rank=0, world=2, shuffle=True, seed=7),
        )
    )
    rank1 = _rows(
        config.build(
            context=context,
            dataset_iteration_policy=_policy(rank=1, world=2, shuffle=True, seed=7),
        )
    )

    order = np.arange(4, dtype=np.int64)
    np.random.RandomState(7).shuffle(order)
    flat = np.arange(17)
    for rows, rank in ((rank0, 0), (rank1, 1)):
        assert len(rows) == 2
        for local, window in enumerate(order[rank::2]):
            start = int(window) * 4
            np.testing.assert_array_equal(rows[local].input_ids, flat[start : start + 4])


def test_loader_checkpoint_resumes_the_same_window(tmp_path):
    prefix = tmp_path / "resume"
    _write_indexed(str(prefix), [np.arange(40, dtype=np.int32)])
    dataset = MegatronDpSplitPackingConfig(
        dataset=megatron_dataset_config(str(prefix) + ".idx")
    )

    def loader():
        return GrainDataLoader.Config(
            dataset=dataset,
            shuffle=False,
            repeat=True,
            num_prefetch_microbatches=1,
        ).build(
            dp_world_size=2,
            dp_rank=1,
            tokenizer=None,
            max_context_length=4,
            num_tokens_per_microbatch=4,
        )

    source = loader()
    iterator = iter(source)
    next(iterator)
    next(iterator)
    state = source.state_dict()
    expected = next(iterator).input
    source.close()

    resumed = loader()
    resumed.load_state_dict(state)
    seen = next(iter(resumed)).input
    resumed.close()
    torch.testing.assert_close(seen, expected)


def test_torchtitan_split_keeps_document_packing():
    dataset = megatron_packed_dataset("missing.idx", split="torchtitan")
    assert isinstance(dataset, ConcatThenSplitPackingConfig)


def test_megatron_split_rejects_weights():
    with pytest.raises(ValueError, match="blend weights"):
        megatron_packed_dataset("missing.idx", weights=[1.0], split="megatron")
