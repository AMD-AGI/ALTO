# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from torchtitan.models.common.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
)

from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from alto.kernels.mxfp8.blockmask import prepare_block_mask
from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8
from .config import TrainingOpConfig

__all__ = ["LPFlexAttentionWrapper", "LPScaledDotProductAttentionWrapper"]


class LPScaledDotProductAttentionWrapper(ScaledDotProductAttentionWrapper):

    def __init__(self, config: TrainingOpConfig):
        super().__init__()
        self.config = config
        self.attn_func = None

        if isinstance(config, TrainingOpConfig) and config.precision == "mxfp4":
            self.attn_func = triton_attention_mxfp4
        elif isinstance(config, TrainingOpConfig) and config.precision == "mxfp8_e4m3":
            self.attn_func = triton_attention_mxfp8
        else:
            raise ValueError(f"Unsupported SDPA config: {config}")

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}[{self.config}]"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
    ):
        batch, num_head_q, seqlen_q, head_dim_qk = q.shape
        batch_k, num_head_kv, seqlen_kv, head_dim_qk_k = k.shape
        batch_v, num_head_kv_v, seqlen_kv_v, head_dim_v = v.shape

        assert batch == batch_k == batch_v
        assert num_head_kv == num_head_kv_v
        assert head_dim_qk == head_dim_qk_k
        assert seqlen_kv == seqlen_kv_v
        assert self.attn_func is not None

        sm_scale = head_dim_qk**(-0.5) if scale is None else scale
        o = self.attn_func(
            q,
            k,
            v,
            bias=None,
            alibi_slopes=None,
            sm_scale=sm_scale,
            dropout_p=0.0,
            cu_seqlens_q=0,
            cu_seqlens_k=0,
            max_seqlens_q=seqlen_q,
            max_seqlens_k=seqlen_kv,
            causal=is_causal,
            return_scores=False,
            use_exp2=True,
            layout="bhsd",
        )[0]
        return o


class LPFlexAttentionWrapper(FlexAttentionWrapper):
    """MXFP8 implementation of FlexAttention's public wrapper contract.

    The argument validation and BlockMask preparation are in place, but the
    kernel behind them is not: calling this raises until the Triton BlockMask
    loops land.
    """

    def __init__(self, config: TrainingOpConfig):
        torch.nn.Module.__init__(self)
        if not isinstance(config, TrainingOpConfig) or config.precision != "mxfp8_e4m3":
            raise ValueError(f"FlexAttention only supports mxfp8_e4m3, got {config}")
        self.config = config

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}[{self.config}]"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        score_mod=None,
        block_mask=None,
        scale: float | None = None,
        return_lse: bool = False,
        enable_gqa: bool = False,
    ):
        if score_mod is not None:
            raise ValueError("MXFP8 FlexAttention accepts mask semantics only through BlockMask.")
        if block_mask is None:
            raise ValueError("MXFP8 FlexAttention requires a Flex BlockMask.")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("MXFP8 FlexAttention requires q, k, v in bhsd layout.")
        if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("q, k, and v batch dimensions must match.")
        if q.shape[1] % k.shape[1]:
            raise ValueError("query head count must be divisible by KV head count.")
        if enable_gqa is False and q.shape[1] != k.shape[1]:
            raise ValueError("GQA tensors require enable_gqa=True.")

        prepared = prepare_block_mask(block_mask)
        output, lse, _ = triton_attention_mxfp8(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            bias=None,
            alibi_slopes=None,
            sm_scale=q.shape[-1] ** -0.5 if scale is None else scale,
            dropout_p=0.0,
            cu_seqlens_q=0,
            cu_seqlens_k=0,
            max_seqlens_q=q.shape[-2],
            max_seqlens_k=k.shape[-2],
            causal=False,
            return_scores=False,
            use_exp2=True,
            layout="bhsd",
            block_mask=prepared,
        )
        return (output, lse) if return_lse else output
