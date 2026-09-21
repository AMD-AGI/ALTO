"""CPU-valid semantic tests for the MXFP8 Flex BlockMask adapter."""

import torch
from torch.nn.attention.flex_attention import create_block_mask

from alto.kernels.mxfp8.blockmask import blockmask_attention_fp32_reference, prepare_block_mask


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
