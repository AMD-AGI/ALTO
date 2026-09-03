"""MXFP4/MXFP8 directional Linear training without double quantization."""
import torch

from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.fp4.mxfp4.mxfp_quantization import BLOCK_SIZE_DEFAULT, is_cdna4

_FP8 = {"mxfp8_e4m3", "mxfp8_e5m2"}


def _q(x, scheme, axis, is_2d, sr=False):
    if scheme == "mxfp4":
        return torch.ops.torchtitan.convert_to_mxfp4(
            x, axis=axis, is_2d_block=is_2d, use_sr=sr
        )
    return torch.ops.alto.convert_to_mxfp8(
        x, block_size=BLOCK_SIZE_DEFAULT, mxfp_format=scheme.removeprefix("mxfp8_"),
        axis=axis, is_2d_block=is_2d, use_sr=sr,
    )


def _mm(a, as_, b, bs, scheme, a2d, b2d, out_dtype, ta=False, tb=False):
    if scheme == "mxfp4":
        kwargs = dict(use_2dblock_a=a2d, use_2dblock_b=b2d, output_dtype=out_dtype,
                      trans_a=ta, trans_b=tb)
        if ta:
            kwargs.update(k_pack_a=not a2d, k_pack_b=not b2d)
        elif not tb:
            kwargs["k_pack_b"] = not b2d
        return torch.ops.torchtitan.blockwise_mxfp4_gemm(a, as_, b, bs, **kwargs)
    return torch.ops.alto.blockwise_mxfp8_gemm(
        a, as_, b, bs, trans_a=ta, trans_b=tb, use_2dblock_a=a2d,
        use_2dblock_b=b2d, block_size=BLOCK_SIZE_DEFAULT, output_dtype=out_dtype,
    )


class DirectionalMXFPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, forward_scheme, backward_scheme, x2d, w2d, sr):
        if not is_cdna4():
            raise RuntimeError("Directional MXFP4/MXFP8 training requires CDNA4.")
        weight = unwrap_weight_wrapper(weight)
        shape, x = x.shape, x.reshape(-1, x.shape[-1])
        xf, xfs = _q(x, forward_scheme, -1, x2d)
        wf, wfs = _q(weight, forward_scheme, -1, w2d)
        y = _mm(xf, xfs, wf, wfs, forward_scheme, x2d, w2d, x.dtype, tb=True)

        # Materialize the backward representation directly from BF16.
        xb, xbs = _q(x, backward_scheme, -1 if x2d else 0, x2d)
        wb, wbs = _q(weight, backward_scheme, -1 if w2d else 0, w2d)
        ctx.save_for_backward(xb, xbs, wb, wbs)
        ctx.scheme, ctx.x2d, ctx.w2d, ctx.sr, ctx.dtype, ctx.shape = (
            backward_scheme, x2d, w2d, sr, x.dtype, shape)
        return y.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, go):
        xb, xbs, wb, wbs = ctx.saved_tensors
        go = go.reshape(-1, go.shape[-1])
        g, gs = _q(go, ctx.scheme, -1, ctx.x2d, ctx.sr)
        gm, gms = (g, gs) if ctx.x2d else _q(go, ctx.scheme, 0, False, ctx.sr)
        dx = _mm(g, gs, wb, wbs, ctx.scheme, ctx.x2d, ctx.w2d, ctx.dtype)
        dw = _mm(gm, gms, xb, xbs, ctx.scheme, ctx.x2d, ctx.x2d, ctx.dtype, ta=True)
        return dx.view(*ctx.shape), dw, None, None, None, None, None


def directional_mxfp_linear(x, weight, *, forward_scheme, backward_scheme,
                            use_2dblock_x, use_2dblock_w, use_sr_grad):
    return DirectionalMXFPLinearFunction.apply(
        x, weight, forward_scheme, backward_scheme, use_2dblock_x, use_2dblock_w, use_sr_grad)
