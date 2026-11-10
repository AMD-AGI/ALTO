# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import grain.python as grain
import numpy as np
import torch

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.megatron import (
    PretokenizedProcessor,
    megatron_packed_dataset,
)
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.text_datasets import DATASETS, TextProcessor
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common.attention import create_varlen_metadata_for_document

from alto.models.gpt_oss.config_registry import gpt_oss_debugmodel, gpt_oss_debugmodel_lpt


class _Tokens:
    def __init__(self, sequences: list[TextSequence]) -> None:
        self._sequences = sequences

    def build(self, *, context, dataset_iteration_policy):
        del context, dataset_iteration_policy
        return grain.MapDataset.source(self._sequences)


class _Tokenizer:
    def encode(self, text, add_bos=True, add_eos=True):
        del text, add_bos, add_eos
        return list(range(41))


def _context(*, num_tokens: int, context_length: int, max_num_documents=None):
    return DatasetBuildContext(
        tokenizer=None,
        max_context_length=context_length,
        num_tokens_per_microbatch=num_tokens,
        read_options=grain.ReadOptions(),
        max_num_documents=max_num_documents,
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


def _processor(seq_align):
    return PretokenizedProcessor(
        PretokenizedProcessor.Config(seq_align=seq_align),
        context=_context(num_tokens=64, context_length=128),
    )


def _document(length: int, *, seq_align: int | None) -> TextSequence:
    tokens = np.arange(length, dtype=np.int64)
    return _processor(seq_align)({"tokens": tokens}, np.random.default_rng(0))


def _boundaries(row: TextSequence) -> torch.Tensor:
    metadata = create_varlen_metadata_for_document(
        torch.as_tensor(row.positions),
        padding_mask=torch.as_tensor(row.padding_mask),
    )
    return metadata.cu_seq_q


def test_pretokenized_processor_pads_document_to_quant_block():
    sequence = _document(41, seq_align=32)
    assert len(sequence.input_ids) == 64
    assert np.array_equal(sequence.input_ids[:40], np.arange(40))
    assert np.all(sequence.input_ids[40:] == 0)
    assert np.array_equal(sequence.labels[:40], np.arange(1, 41))
    assert np.all(sequence.labels[40:] == IGNORE_INDEX)
    assert np.array_equal(
        sequence.padding_mask, np.array([False] * 40 + [True] * 24)
    )

    unchanged = _document(33, seq_align=None)
    assert len(unchanged.input_ids) == 32
    assert unchanged.padding_mask is None


def test_text_processor_uses_the_same_alignment():
    processor = TextProcessor(
        TextProcessor.Config(seq_align=32),
        context=DatasetBuildContext(
            tokenizer=_Tokenizer(),
            max_context_length=128,
            num_tokens_per_microbatch=128,
            read_options=grain.ReadOptions(),
        ),
    )
    sequence = processor({"text": "unused"}, np.random.default_rng(0))
    assert len(sequence.input_ids) == 64
    assert np.all(sequence.padding_mask[40:])


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

    aware = ConcatThenSplitPackingConfig(dataset=_Tokens(sequences)).build(
        context=_context(num_tokens=96, context_length=128, max_num_documents=8),
        dataset_iteration_policy=_policy(),
    )
    aware_rows = list(aware)
    assert len(aware_rows) == 2
    assert torch.equal(_boundaries(aware_rows[0]), torch.tensor([0, 64, 96], dtype=torch.int32))
    assert torch.equal(_boundaries(aware_rows[1]), torch.tensor([0, 32, 96], dtype=torch.int32))


def test_lpt_configs_enable_document_alignment():
    lpt = gpt_oss_debugmodel_lpt()
    assert lpt.dataloader.dataset.dataset.processor.seq_align == 32
    stock = gpt_oss_debugmodel()
    assert stock.dataloader.dataset.dataset.processor.seq_align is None
    assert DATASETS["c4_test"].processor.seq_align is None

    packed = megatron_packed_dataset("missing.idx", seq_align=32)
    assert packed.dataset.processor.seq_align == 32
