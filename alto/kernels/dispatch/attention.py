# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import ScaledDotProductInnerAttention

from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from .config import TrainingOpConfig

__all__ = ["LPScaledDotProductAttention"]


class LPScaledDotProductAttention(ScaledDotProductInnerAttention):
    """Low-precision SDPA inner attention.

    Must be selected in the model config (not swapped into a built model):
    attention metadata is keyed by the configured inner attention class.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductInnerAttention.Config):
        training_op_config: TrainingOpConfig

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config.training_op_config
        if self.config.precision == "mxfp4":
            self.attn_func = triton_attention_mxfp4
        else:
            raise ValueError(f"Unsupported SDPA config: {self.config}")

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}[{self.config}]"

    def attention_BLHK(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        *,
        scale: float | None,
        enable_gqa: bool,
        is_causal: bool,
    ) -> torch.Tensor:
        q, k, v = (
            q_BLHK.transpose(1, 2),
            k_BLHK.transpose(1, 2),
            v_BLHV.transpose(1, 2),
        )
        batch, num_head_q, seqlen_q, head_dim_qk = q.shape
        batch_k, num_head_kv, seqlen_kv, head_dim_qk_k = k.shape
        batch_v, num_head_kv_v, seqlen_kv_v, head_dim_v = v.shape

        assert batch == batch_k == batch_v
        assert num_head_kv == num_head_kv_v
        assert head_dim_qk == head_dim_qk_k
        assert seqlen_kv == seqlen_kv_v

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
        return o.transpose(1, 2)
