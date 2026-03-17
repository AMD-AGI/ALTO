import torch
from torchtitan.models.common.attention import (
    ScaledDotProductAttentionWrapper)

from modeloptimizer.kernels.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from .config import TrainingOpConfig

__all__ = ["LPScaledDotProductAttentionWrapper"]


class LPScaledDotProductAttentionWrapper(ScaledDotProductAttentionWrapper):

    def __init__(self, config: TrainingOpConfig):
        super().__init__()
        self.config = config
        self.attn_func = None

        if isinstance(config, TrainingOpConfig) and config.precision == "mxfp4":
            self.attn_func = triton_attention_mxfp4
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
