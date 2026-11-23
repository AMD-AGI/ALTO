"""Adapt FlexAttention ``BlockMask`` tables for the MXFP8 attention kernels.

Flex already owns mask semantics.  This module deliberately does not infer
causal/window behavior: it only materializes the token masks for the partial
blocks named by Flex's sparse tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
from torch.nn.attention.flex_attention import BlockMask

BLOCK_SIZE = 128


@dataclass(frozen=True)
class MXFP8BlockMask:
    """Kernel-ready BlockMask tensors.

    ``kv_partial_mask`` and ``q_partial_mask`` are dense only in their first
    dimension.  Their index is the flattened ``[B, H, q_block, slot]`` (or
    corresponding Q-table) slot, including invalid slots.  Keeping that shape
    makes Triton's lookup a few integer operations rather than an indirection.
    """

    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    full_kv_num_blocks: torch.Tensor
    full_kv_indices: torch.Tensor
    q_num_blocks: torch.Tensor
    q_indices: torch.Tensor
    full_q_num_blocks: torch.Tensor
    full_q_indices: torch.Tensor
    kv_partial_mask: torch.Tensor
    q_partial_mask: torch.Tensor
    seqlen_q: int
    seqlen_kv: int


_CACHE: WeakKeyDictionary[BlockMask, MXFP8BlockMask] = WeakKeyDictionary()


def _require_table(
    name: str, value: torch.Tensor | None, *, like: torch.Tensor | None = None
) -> torch.Tensor:
    if value is None:
        if like is None:
            raise ValueError(f"BlockMask must provide {name}; construct it with FlexAttention.")
        return torch.empty((*like.shape[:-1], 0), dtype=like.dtype, device=like.device)
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"BlockMask {name} must contain integer block indices, got {value.dtype}.")
    if value.ndim != 4:
        raise ValueError(f"BlockMask {name} must be [B, H, blocks, slots], got {tuple(value.shape)}.")
    return value.contiguous()


def _require_counts(
    name: str, value: torch.Tensor | None, indices: torch.Tensor, *, like: torch.Tensor | None = None
) -> torch.Tensor:
    if value is None:
        if like is None:
            raise ValueError(f"BlockMask must provide {name}; construct it with FlexAttention.")
        return torch.zeros(like.shape[:-1], dtype=like.dtype, device=like.device)
    if value.shape != indices.shape[:-1]:
        raise ValueError(f"BlockMask {name} shape {tuple(value.shape)} does not match {tuple(indices.shape[:-1])}.")
    return value.contiguous()


def _materialize_partial(
    block_mask: BlockMask,
    num_blocks: torch.Tensor,
    indices: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    *,
    index_is_q_block: bool,
) -> torch.Tensor:
    """Materialize only valid partial slots into flattened-slot storage."""
    batch, heads, q_blocks, slots = indices.shape
    device = indices.device
    result = torch.zeros(
        (batch * heads * q_blocks * slots, q_block_size, kv_block_size),
        dtype=torch.bool,
        device=device,
    )
    slot = torch.arange(slots, device=device)
    valid = slot[None, None, None, :] < num_blocks[..., None]
    b, h, q_block, block_slot = valid.nonzero(as_tuple=True)
    if not b.numel():
        return result

    indexed_block = indices[b, h, q_block, block_slot]
    if index_is_q_block:
        q_positions = indexed_block[:, None] * q_block_size + torch.arange(q_block_size, device=device)[None, :]
        kv_positions = q_block[:, None] * kv_block_size + torch.arange(kv_block_size, device=device)[None, :]
    else:
        q_positions = q_block[:, None] * q_block_size + torch.arange(q_block_size, device=device)[None, :]
        kv_positions = indexed_block[:, None] * kv_block_size + torch.arange(kv_block_size, device=device)[None, :]
    masks = block_mask.mask_mod(
        b[:, None, None],
        h[:, None, None],
        q_positions[:, :, None],
        kv_positions[:, None, :],
    ).to(dtype=torch.bool)
    flat_slot = ((b * heads + h) * q_blocks + q_block) * slots + block_slot
    result[flat_slot] = masks
    return result


def prepare_block_mask(block_mask: BlockMask) -> MXFP8BlockMask:
    """Validate and convert a Flex ``BlockMask`` to MXFP8 kernel inputs."""
    cached = _CACHE.get(block_mask)
    if cached is not None:
        return cached

    if tuple(block_mask.BLOCK_SIZE) != (BLOCK_SIZE, BLOCK_SIZE):
        raise ValueError(
            f"MXFP8 BlockMask requires BLOCK_SIZE={(BLOCK_SIZE, BLOCK_SIZE)}, "
            f"got {tuple(block_mask.BLOCK_SIZE)}."
        )

    kv_indices = _require_table("kv_indices", block_mask.kv_indices)
    q_indices = _require_table("q_indices", block_mask.q_indices)
    full_kv_indices = _require_table("full_kv_indices", block_mask.full_kv_indices, like=kv_indices)
    full_q_indices = _require_table("full_q_indices", block_mask.full_q_indices, like=q_indices)
    kv_num_blocks = _require_counts("kv_num_blocks", block_mask.kv_num_blocks, kv_indices)
    q_num_blocks = _require_counts("q_num_blocks", block_mask.q_num_blocks, q_indices)
    full_kv_num_blocks = _require_counts(
        "full_kv_num_blocks", block_mask.full_kv_num_blocks, full_kv_indices, like=kv_indices
    )
    full_q_num_blocks = _require_counts(
        "full_q_num_blocks", block_mask.full_q_num_blocks, full_q_indices, like=q_indices
    )

    prepared = MXFP8BlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        kv_partial_mask=_materialize_partial(
            block_mask, kv_num_blocks, kv_indices, BLOCK_SIZE, BLOCK_SIZE, index_is_q_block=False
        ),
        q_partial_mask=_materialize_partial(
            block_mask, q_num_blocks, q_indices, BLOCK_SIZE, BLOCK_SIZE, index_is_q_block=True
        ),
        seqlen_q=int(block_mask.shape[-2]),
        seqlen_kv=int(block_mask.shape[-1]),
    )
    _CACHE[block_mask] = prepared
    return prepared


def blockmask_attention_fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prepared: MXFP8BlockMask,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 oracle for the table walk. NOT MXFP8 and not a production path.

    It applies no quantization at all, so its numbers are higher precision than
    any kernel result: use it only to check that the full/partial tables are
    being read correctly. It never builds an ``[B, H, S, S]`` mask, and the
    Python-level block loop makes it far too slow for training.
    """
    batch, heads_q, seqlen_q, _ = q.shape
    _, heads_kv, seqlen_kv, _ = k.shape
    if heads_q % heads_kv:
        raise ValueError(f"query heads ({heads_q}) must be divisible by KV heads ({heads_kv}).")
    if prepared.kv_indices.shape[0] not in (1, batch):
        raise ValueError("BlockMask batch dimension must be 1 or match q.")
    if prepared.kv_indices.shape[1] not in (1, heads_q):
        raise ValueError("BlockMask head dimension must be 1 or match query heads.")

    group_size = heads_q // heads_kv
    # Keep an autograd edge even when every query tile is empty.
    zero = (q.sum() + k.sum() + v.sum()) * 0
    output = torch.zeros((batch, heads_q, seqlen_q, v.shape[-1]), dtype=q.dtype, device=q.device) + zero
    lse = torch.zeros((batch, heads_q, seqlen_q), dtype=torch.float32, device=q.device)
    q_blocks = (seqlen_q + BLOCK_SIZE - 1) // BLOCK_SIZE

    for batch_idx in range(batch):
        table_batch = min(batch_idx, prepared.kv_indices.shape[0] - 1)
        for head_idx in range(heads_q):
            table_head = min(head_idx, prepared.kv_indices.shape[1] - 1)
            kv_head = head_idx // group_size
            for q_block in range(q_blocks):
                q_start = q_block * BLOCK_SIZE
                q_end = min(q_start + BLOCK_SIZE, seqlen_q)
                full_count = int(prepared.full_kv_num_blocks[table_batch, table_head, q_block])
                partial_count = int(prepared.kv_num_blocks[table_batch, table_head, q_block])
                key_parts: list[torch.Tensor] = []
                value_parts: list[torch.Tensor] = []
                mask_parts: list[torch.Tensor] = []

                for slot in range(full_count):
                    kv_block = int(prepared.full_kv_indices[table_batch, table_head, q_block, slot])
                    start = kv_block * BLOCK_SIZE
                    end = min(start + BLOCK_SIZE, seqlen_kv)
                    key_parts.append(k[batch_idx, kv_head, start:end])
                    value_parts.append(v[batch_idx, kv_head, start:end])
                    mask_parts.append(torch.ones((q_end - q_start, end - start), dtype=torch.bool, device=q.device))
                for slot in range(partial_count):
                    kv_block = int(prepared.kv_indices[table_batch, table_head, q_block, slot])
                    start = kv_block * BLOCK_SIZE
                    end = min(start + BLOCK_SIZE, seqlen_kv)
                    flat_slot = ((table_batch * prepared.kv_indices.shape[1] + table_head) *
                                 prepared.kv_indices.shape[2] + q_block) * prepared.kv_indices.shape[3] + slot
                    key_parts.append(k[batch_idx, kv_head, start:end])
                    value_parts.append(v[batch_idx, kv_head, start:end])
                    mask_parts.append(prepared.kv_partial_mask[flat_slot, :q_end - q_start, :end - start])

                if not key_parts:
                    continue
                keys = torch.cat(key_parts, dim=0)
                values = torch.cat(value_parts, dim=0)
                allowed = torch.cat(mask_parts, dim=1)
                scores = (q[batch_idx, head_idx, q_start:q_end].float() @ keys.float().transpose(0, 1)) * sm_scale
                scores = scores.masked_fill(~allowed, float("-inf"))
                row_lse = torch.logsumexp(scores, dim=-1)
                probabilities = torch.softmax(scores, dim=-1)
                probabilities = torch.nan_to_num(probabilities)
                output[batch_idx, head_idx, q_start:q_end] = (probabilities @ values.float()).to(q.dtype)
                lse[batch_idx, head_idx, q_start:q_end] = torch.nan_to_num(row_lse)
    return output, lse
