"""Dispatch-path NVFP4 training: 10K steps comparison.

Validates that the newly added NVFP4 dispatch hook works correctly by running
BF16, MXFP4 and NVFP4 training through the same __torch_function__ mechanism.

This script is intentionally self-contained and does NOT depend on torchao,
so it runs in environments where that package is not installed.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import copy
import json
import time
import torch
import torch.nn as nn

from torchtitan.models.llama3 import llama3_configs
from alto.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction
from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm


# ---------------------------------------------------------------------------
# Lightweight tensor subclasses that implement the dispatch mechanism
# ---------------------------------------------------------------------------

class NVFP4WeightTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, use_2dblock_x=False, use_2dblock_w=False,
                use_sr_grad=True, use_per_tensor_scale=False):
        t = torch.Tensor._make_subclass(cls, data)
        t._cfg = dict(
            use_2dblock_x=use_2dblock_x,
            use_2dblock_w=use_2dblock_w,
            use_sr_grad=use_sr_grad,
            use_per_tensor_scale=use_per_tensor_scale,
        )
        return t

    def detach(self):
        # nn.Parameter creation requires detach() to return the same subclass type.
        t = type(self)(self.as_subclass(torch.Tensor).detach(), **self._cfg)
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("linear", "mm.default", "matmul.default", "addmm.default"):
            trans_b = func.__name__ == "linear"
            if func.__name__ == "addmm.default":
                bias, A, B = args[0], args[1], args[2]
            else:
                A, B = args[0], args[1]
                bias = args[2] if len(args) > 2 else None
            assert isinstance(B, cls)
            c = B._cfg
            W = B if trans_b else B.T
            Y = NVFP4LinearFunction.apply(
                A,
                W.as_subclass(torch.Tensor),
                c["use_2dblock_x"],
                c["use_2dblock_w"],
                c["use_sr_grad"],
                c["use_per_tensor_scale"],
            )
            if bias is not None:
                Y = Y + bias
            return Y
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


class MXFP4WeightTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, use_2dblock_x=False, use_2dblock_w=False,
                use_sr_grad=True):
        t = torch.Tensor._make_subclass(cls, data)
        t._cfg = dict(
            use_2dblock_x=use_2dblock_x,
            use_2dblock_w=use_2dblock_w,
            use_sr_grad=use_sr_grad,
        )
        return t

    def detach(self):
        t = type(self)(self.as_subclass(torch.Tensor).detach(), **self._cfg)
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.__name__ in ("linear", "mm.default", "matmul.default", "addmm.default"):
            trans_b = func.__name__ == "linear"
            if func.__name__ == "addmm.default":
                bias, A, B = args[0], args[1], args[2]
            else:
                A, B = args[0], args[1]
                bias = args[2] if len(args) > 2 else None
            assert isinstance(B, cls)
            c = B._cfg
            W = B.as_subclass(torch.Tensor) if trans_b else B.as_subclass(torch.Tensor).T
            Y = _to_mxfp4_then_scaled_mm(
                A, W,
                use_2dblock_x=c["use_2dblock_x"],
                use_2dblock_w=c["use_2dblock_w"],
                use_sr_grad=c["use_sr_grad"],
                use_dge=False,
                use_hadamard=False,
            )
            if bias is not None:
                Y = Y + bias
            return Y
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Weight-wrapping helper
# ---------------------------------------------------------------------------

def wrap_weights(model, weight_cls, weight_kwargs, *, keep_tail_blocks_bf16=0):
    total_layers = len(model.layers) if hasattr(model, "layers") else 0

    def _should_wrap(fqn):
        if keep_tail_blocks_bf16 <= 0 or total_layers == 0:
            return True
        parts = fqn.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            return int(parts[1]) < total_layers - keep_tail_blocks_bf16
        return False   # embedding / output stay BF16

    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _should_wrap(fqn):
            continue
        p = module.weight
        module.weight = nn.Parameter(
            weight_cls(p.data, **weight_kwargs),
            requires_grad=p.requires_grad,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(model, steps, batch_size, seq_len, vocab_size, lr, device, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    t0 = time.time()
    for step in range(steps):
        torch.manual_seed(step)
        x   = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            model(x).view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 1000 == 0 or step == steps - 1:
            tail = losses[-200:] if len(losses) >= 200 else losses
            ta = sum(tail) / len(tail)
            print(f"[{label}] step {step:5d}/{steps}"
                  f"  loss={loss.item():.4f}  tail_avg={ta:.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    return losses


def summarize(losses):
    tail = losses[-200:]
    m = sum(tail) / len(tail)
    std = (sum((x - m) ** 2 for x in tail) / len(tail)) ** 0.5
    return {"tail_avg": m, "tail_std": std, "best": min(losses), "final": losses[-1]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda")
    torch.manual_seed(42)
    steps, batch_size, seq_len, lr = 10000, 4, 128, 3e-4

    cfg = llama3_configs["debugmodel"]
    base = cfg.build().to(device).bfloat16()
    base.init_weights(buffer_device=device)
    vocab_size = cfg.vocab_size
    nparams = sum(p.numel() for p in base.parameters())

    print(f"Model  : llama3_debugmodel  ({nparams/1e6:.2f}M params, vocab={vocab_size})")
    print(f"Steps  : {steps}  batch={batch_size}  seq_len={seq_len}")
    print("-" * 65)
    print("  BF16   : standard BF16 GEMM")
    print("  MXFP4  : __torch_function__ dispatch -> _to_mxfp4_then_scaled_mm")
    print("  NVFP4  : __torch_function__ dispatch -> NVFP4LinearFunction (tail2 BF16)")
    print("=" * 65)

    model_bf16 = copy.deepcopy(base)

    model_mx = copy.deepcopy(base)
    wrap_weights(model_mx, MXFP4WeightTensor,
                 dict(use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=True),
                 keep_tail_blocks_bf16=0)

    model_nv = copy.deepcopy(base)
    wrap_weights(model_nv, NVFP4WeightTensor,
                 dict(use_2dblock_x=False, use_2dblock_w=False,
                      use_sr_grad=True, use_per_tensor_scale=False),
                 keep_tail_blocks_bf16=2)

    l_bf16 = train_loop(model_bf16, steps, batch_size, seq_len,
                        vocab_size, lr, device, "BF16  ")
    l_mx   = train_loop(model_mx,   steps, batch_size, seq_len,
                        vocab_size, lr, device, "MXFP4 ")
    l_nv   = train_loop(model_nv,   steps, batch_size, seq_len,
                        vocab_size, lr, device, "NVFP4 ")

    print("=" * 65)
    print("=== Final 10K Summary ===")
    bf_ta = summarize(l_bf16)["tail_avg"]
    mx_ta = summarize(l_mx)["tail_avg"]
    for name, losses in [
        ("BF16", l_bf16),
        ("MXFP4 (dispatch)", l_mx),
        ("NVFP4 (dispatch, tail2+SR)", l_nv),
    ]:
        s = summarize(losses)
        gap_bf  = s["tail_avg"] - bf_ta
        gap_mx  = s["tail_avg"] - mx_ta
        print(f"  {name:32s}"
              f"  tail_avg={s['tail_avg']:.4f}"
              f"  gap_bf16={gap_bf:+.4f}"
              f"  gap_mxfp4={gap_mx:+.4f}"
              f"  best={s['best']:.4f}"
              f"  final={s['final']:.4f}")

    with open("/tmp/dispatch_nvfp4_10k.json", "w") as f:
        json.dump({"bf16": l_bf16, "mxfp4_dispatch": l_mx,
                   "nvfp4_dispatch_tail2": l_nv}, f)
    print("Saved: /tmp/dispatch_nvfp4_10k.json")


if __name__ == "__main__":
    main()
