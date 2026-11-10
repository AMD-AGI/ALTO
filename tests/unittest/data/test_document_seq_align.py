# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import grain.python as grain
import numpy as np
import torch

from alto.models.gpt_oss.config_registry import (
    gpt_oss_20b_mxfp8_attn,
    gpt_oss_20b_mxfp8_attn_aligned,
)
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.megatron import PretokenizedProcessor
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common.attention import create_varlen_metadata_for_document


class _Tokens:
    def __init__(self, sequences: list[TextSequence]) -> None:
        self._sequences = sequences

    def build(self, *, context, dataset_iteration_policy):
        del context, dataset_iteration_policy
        return grain.MapDataset.source(self._sequences)


def _context(*, num_tokens: int, context_length: int):
    return DatasetBuildContext(
        tokenizer=None,
        max_context_length=context_length,
        num_tokens_per_microbatch=num_tokens,
        read_options=grain.ReadOptions(),
    )


def _policy():
    return DatasetIterationPolicy(
        seed=0,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
        streaming_shuffle_buffer_size=1,
    )


def _document(length: int, *, seq_align: int | None) -> TextSequence:
    processor = PretokenizedProcessor(
        PretokenizedProcessor.Config(seq_align=seq_align),
        context=_context(num_tokens=64, context_length=128),
    )
    tokens = np.arange(length, dtype=np.int64)
    return processor({"tokens": tokens}, np.random.default_rng(0))


def _boundaries(row: TextSequence) -> torch.Tensor:
    metadata = create_varlen_metadata_for_document(
        torch.as_tensor(row.positions),
        padding_mask=torch.as_tensor(row.padding_mask),
    )
    return metadata.cu_seq_q


def test_pretokenized_processor_pads_document_to_quant_block():
    sequence = _document(41, seq_align=32)
    assert len(sequence.input_ids) == 64
    assert np.all(sequence.labels[40:] == IGNORE_INDEX)
    assert np.array_equal(
        sequence.padding_mask, np.array([False] * 40 + [True] * 24)
    )

    unchanged = _document(33, seq_align=None)
    assert len(unchanged.input_ids) == 32
    assert unchanged.padding_mask is None


def test_packed_rows_keep_aligned_document_boundaries():
    sequences = [_document(41, seq_align=32), _document(50, seq_align=32)]
    assert [len(sequence.input_ids) for sequence in sequences] == [64, 64]
    dataset = ConcatThenSplitPackingConfig(dataset=_Tokens(sequences)).build(
        context=_context(num_tokens=96, context_length=128),
        dataset_iteration_policy=_policy(),
    )
    rows = list(dataset)
    assert len(rows) == 1
    boundaries = _boundaries(rows[0])
    assert torch.equal(boundaries, torch.tensor([0, 64, 96], dtype=torch.int32))
    assert torch.all(boundaries % 32 == 0)


def test_mxfp8_attn_document_alignment_is_a_separate_config():
    baseline = gpt_oss_20b_mxfp8_attn()
    aligned = gpt_oss_20b_mxfp8_attn_aligned()

    assert baseline.dataloader.dataset.dataset.processor.seq_align is None
    assert baseline.validator.dataloader.dataset.dataset.processor.seq_align is None
    assert aligned.dataloader.dataset.dataset.processor.seq_align == 32
    assert aligned.validator.dataloader.dataset.dataset.processor.seq_align == 32
    assert aligned.dump_folder != baseline.dump_folder

    for name in ("training", "optim", "parallelism", "loss", "checkpointer", "debug", "model"):
        assert getattr(aligned, name) == getattr(baseline, name), name
    assert aligned.model_converters == baseline.model_converters
