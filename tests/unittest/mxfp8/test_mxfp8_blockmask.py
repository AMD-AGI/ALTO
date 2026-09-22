"""Tests for the MXFP8 Flex BlockMask adapter and the kernel path it feeds.

Everything except ``test_blockmask_kernel_matches_reference`` runs on CPU: the
table walk and the mask materialization are plain PyTorch, so they can be
validated before CDNA4 hardware is in the loop.
"""

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from alto.kernels.mxfp8.blockmask import blockmask_attention_fp32_reference, prepare_block_mask
from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8

from .utils import (
    calc_cossim,
    calc_snr,
    mxfp8_attention_forward_reference,
    mxfp8_blockmask_forward_reference,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required.")


def _causal_window(window: int):
    def mask_mod(batch, head, q_idx, kv_idx):
        return (kv_idx <= q_idx) & ((q_idx - kv_idx) < window)

    return mask_mod


def _dense_mask(mask_mod, batch: int, heads: int, q_len: int, kv_len: int, device: str):
    b = torch.arange(batch, device=device)[:, None, None, None]
    h = torch.arange(heads, device=device)[None, :, None, None]
    q = torch.arange(q_len, device=device)[None, None, :, None]
    kv = torch.arange(kv_len, device=device)[None, None, None, :]
    return mask_mod(b, h, q, kv)


def test_prepare_blockmask_materializes_partial_slots_and_caches():
    mask_mod = _causal_window(1)
    block_mask = create_block_mask(mask_mod, 1, 1, 256, 256, device="cpu", BLOCK_SIZE=128)

    prepared = prepare_block_mask(block_mask)
    assert prepare_block_mask(block_mask) is prepared
    assert prepared.kv_partial_mask.shape[1:] == (128, 128)
    assert prepared.q_partial_mask.shape[1:] == (128, 128)

    for q_block in range(prepared.kv_indices.shape[2]):
        count = int(prepared.kv_num_blocks[0, 0, q_block])
        for slot in range(count):
            kv_block = int(prepared.kv_indices[0, 0, q_block, slot])
            flat_slot = (q_block * prepared.kv_indices.shape[3]) + slot
            expected = _dense_mask(mask_mod, 1, 1, 256, 256, "cpu")[0, 0,
                q_block * 128:(q_block + 1) * 128, kv_block * 128:(kv_block + 1) * 128]
            assert torch.equal(prepared.kv_partial_mask[flat_slot], expected)

    for kv_block in range(prepared.q_indices.shape[2]):
        count = int(prepared.q_num_blocks[0, 0, kv_block])
        for slot in range(count):
            q_block = int(prepared.q_indices[0, 0, kv_block, slot])
            flat_slot = (kv_block * prepared.q_indices.shape[3]) + slot
            expected = _dense_mask(mask_mod, 1, 1, 256, 256, "cpu")[0, 0,
                q_block * 128:(q_block + 1) * 128, kv_block * 128:(kv_block + 1) * 128]
            assert torch.equal(prepared.q_partial_mask[flat_slot], expected)


def test_sparse_blockmask_matches_dense_window_and_gqa_gradients():
    torch.manual_seed(7)
    mask_mod = _causal_window(128)
    block_mask = create_block_mask(mask_mod, 1, 1, 256, 256, device="cpu", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)

    q = torch.randn(1, 2, 256, 32, requires_grad=True)
    k = torch.randn(1, 1, 256, 32, requires_grad=True)
    v = torch.randn(1, 1, 256, 32, requires_grad=True)
    scale = 32**-0.5
    output, lse = blockmask_attention_fp32_reference(q, k, v, prepared, scale)

    dense = _dense_mask(mask_mod, 1, 2, 256, 256, "cpu")
    scores = (q @ k.repeat_interleave(2, dim=1).transpose(-1, -2)) * scale
    dense_lse = torch.logsumexp(scores.masked_fill(~dense, float("-inf")), dim=-1)
    expected = torch.softmax(scores.masked_fill(~dense, float("-inf")), dim=-1) @ v.repeat_interleave(2, dim=1)
    assert torch.allclose(output, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(lse, dense_lse, atol=2e-5, rtol=2e-5)

    do = torch.randn_like(output)
    got_grads = torch.autograd.grad((output * do).sum(), (q, k, v), retain_graph=True)
    ref_grads = torch.autograd.grad((expected * do).sum(), (q, k, v))
    for got, expected_grad in zip(got_grads, ref_grads):
        assert torch.allclose(got, expected_grad, atol=2e-5, rtol=2e-5)


def test_sparse_blockmask_empty_query_rows_are_zero():
    def no_tokens(batch, head, q_idx, kv_idx):
        return torch.zeros_like(q_idx + kv_idx, dtype=torch.bool)

    block_mask = create_block_mask(no_tokens, 1, 1, 128, 128, device="cpu", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 1, 128, 32, requires_grad=True)
    k = torch.randn(1, 1, 128, 32, requires_grad=True)
    v = torch.randn(1, 1, 128, 32, requires_grad=True)
    output, lse = blockmask_attention_fp32_reference(q, k, v, prepared, 32**-0.5)

    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(lse) == 0
    dq, dk, dv = torch.autograd.grad(output.sum(), (q, k, v), allow_unused=True)
    assert torch.count_nonzero(dq) == 0
    assert torch.count_nonzero(dk) == 0
    assert torch.count_nonzero(dv) == 0


def _call_kernel(q, k, v, prepared, **overrides):
    kwargs = dict(
        bias=None,
        alibi_slopes=None,
        sm_scale=q.shape[-1]**-0.5,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=q.shape[2],
        max_seqlens_k=k.shape[2],
        causal=False,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
        block_mask=prepared,
    )
    kwargs.update(overrides)
    return triton_attention_mxfp8(q, k, v, **kwargs)


@pytest.mark.parametrize("overrides, message", [
    ({"causal": True}, "gets causality from the mask tables"),
    ({"layout": "thd"}, "requires layout 'bhsd'"),
    ({"dropout_p": 0.1}, "does not support dropout"),
    ({"return_scores": True}, "does not return scores"),
])
def test_blockmask_entry_rejects_inputs_it_cannot_honor(overrides, message):
    """Anything the table walk ignores must fail loudly, not be silently dropped."""
    block_mask = create_block_mask(_causal_window(128), 1, 1, 128, 128, device="cpu", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 2, 128, 64)
    k = torch.randn(1, 2, 128, 64)
    v = torch.randn(1, 2, 128, 64)
    with pytest.raises(ValueError, match=message):
        _call_kernel(q, k, v, prepared, **overrides)


def test_blockmask_entry_rejects_sequence_length_mismatch():
    block_mask = create_block_mask(_causal_window(128), 1, 1, 256, 256, device="cpu", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 2, 128, 64)
    k = torch.randn(1, 2, 128, 64)
    v = torch.randn(1, 2, 128, 64)
    with pytest.raises(ValueError, match="BlockMask was built for"):
        _call_kernel(q, k, v, prepared)


@cuda_only
@pytest.mark.parametrize("window", [1, 128, 256, 1024])
@pytest.mark.parametrize("num_head_q, num_head_kv", [(4, 4), (8, 2)])
def test_blockmask_kernel_matches_reference(window, num_head_q, num_head_kv):
    """Kernel vs a reference that quantizes identically and walks the same order.

    Both sides see the same mxfp8 operands and the same full-then-partial block
    sequence, so a gap here is a Triton port bug (index tables, partial mask
    addressing, empty-row handling) rather than quantization error.
    """
    torch.manual_seed(1234)
    seqlen, head_dim = 512, 128
    block_mask = create_block_mask(_causal_window(window), None, None, seqlen, seqlen, device="cuda", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)

    q = torch.randn(2, num_head_q, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, num_head_kv, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, num_head_kv, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)

    o_kernel, lse_kernel, _ = _call_kernel(q.contiguous(), k.contiguous(), v.contiguous(), prepared)
    o_ref, lse_ref = mxfp8_blockmask_forward_reference(q, k, v, prepared, head_dim**-0.5)

    assert calc_cossim(o_ref, o_kernel) > 0.99
    assert calc_snr(o_ref, o_kernel) > 30
    assert torch.allclose(lse_kernel.float(), lse_ref.float(), atol=2e-2, rtol=2e-2)


@cuda_only
def test_blockmask_kernel_empty_query_tile_is_zero():
    """An empty table must use the kernel early exit, not create -inf or NaN."""
    def no_tokens(batch, head, q_idx, kv_idx):
        return torch.zeros_like(q_idx + kv_idx, dtype=torch.bool)

    block_mask = create_block_mask(no_tokens, None, None, 128, 128, device="cuda", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, lse, _ = _call_kernel(q, k, v, prepared)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(lse) == 0


@cuda_only
def test_blockmask_kernel_partial_tile_empty_rows_are_zero():
    """Rows empty inside a nonempty partial tile must not poison the softmax."""
    def alternating_diagonal(batch, head, q_idx, kv_idx):
        return (q_idx % 2 == 0) & (q_idx == kv_idx)

    block_mask = create_block_mask(alternating_diagonal, None, None, 256, 256, device="cuda", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, lse, _ = _call_kernel(q, k, v, prepared)
    dense_mask = _dense_mask(alternating_diagonal, 1, 2, 256, 256, "cuda")
    expected, expected_lse = mxfp8_attention_forward_reference(
        q, k, v, 64**-0.5, causal=False, mask=dense_mask
    )

    assert torch.count_nonzero(output[:, :, 1::2]) == 0
    assert torch.count_nonzero(lse[:, :, 1::2]) == 0
    assert torch.allclose(output, expected, atol=2e-2, rtol=2e-2)
    assert torch.allclose(lse, expected_lse, atol=2e-2, rtol=2e-2)


@cuda_only
def test_blockmask_kernel_uses_per_batch_per_head_tables():
    """Non-broadcast B/H dimensions must use their real strides, not stride zero."""
    def shifted_diagonal(batch, head, q_idx, kv_idx):
        return kv_idx == ((q_idx + batch + head) % 256)

    block_mask = create_block_mask(shifted_diagonal, 2, 4, 256, 256, device="cuda", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, lse, _ = _call_kernel(q, k, v, prepared)
    dense_mask = _dense_mask(shifted_diagonal, 2, 4, 256, 256, "cuda")
    expected, expected_lse = mxfp8_attention_forward_reference(
        q, k, v, 64**-0.5, causal=False, mask=dense_mask
    )

    assert calc_cossim(expected, output) > 0.99
    assert torch.allclose(lse, expected_lse, atol=2e-2, rtol=2e-2)


@cuda_only
def test_blockmask_backward_refuses_to_guess():
    """Forward reads the tables; backward still walks causal ranges, so it must raise."""
    block_mask = create_block_mask(_causal_window(256), None, None, 256, 256, device="cuda", BLOCK_SIZE=128)
    prepared = prepare_block_mask(block_mask)
    q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    o = _call_kernel(q, k, v, prepared)[0]
    with pytest.raises(NotImplementedError, match="MXFP8 BlockMask backward"):
        o.sum().backward()
