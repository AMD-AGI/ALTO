import torch

from torchtitan.models.llama3.model.model import apply_rotary_emb as original_apply_rotary_emb

__all__ = ["apply_rotary_emb"]


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    xq = xq.reshape(
        *xq.shape[:-1],
        2,
        head_dim // 2,
    ).transpose(-1, -2).reshape(
        *xq.shape[:-1],
        head_dim,
    ).contiguous()
    xk = xk.reshape(
        *xk.shape[:-1],
        2,
        head_dim // 2,
    ).transpose(-1, -2).reshape(
        *xk.shape[:-1],
        head_dim,
    ).contiguous()
    return original_apply_rotary_emb(xq, xk, freqs_cis)
