"""Route A: Minimal noise — 3 QDQ (no axis=0) + Forward SR for activation."""
import json, torch, torch.nn as nn, copy, time, sys
sys.stdout.reconfigure(line_buffering=True)

from torchtitan.models.llama3 import llama3_configs
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    convert_to_nvfp4, convert_from_nvfp4,
)


def _qdq(tensor, *, axis, use_sr=False, block_size=16):
    data_lp, scales, pts = convert_to_nvfp4(
        tensor, block_size=block_size, axis=axis, use_sr=use_sr,
    )
    return convert_from_nvfp4(
        data_lp, scales, output_dtype=tensor.dtype,
        block_size=block_size, axis=axis,
    )


@torch.compiler.allow_in_graph
class RouteAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        x_dq = _qdq(x_2d, axis=-1, use_sr=True)    # SR for activation
        w_dq = _qdq(weight, axis=-1, use_sr=False)   # RNE for weight

        y = x_dq @ w_dq.T

        ctx.save_for_backward(x_dq, w_dq)
        ctx.use_sr = True

        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])

        grad_dq = _qdq(grad_output, axis=-1, use_sr=True)

        grad_inputs = grad_dq @ w_dq
        grad_weights = grad_dq.T @ x_dq

        return grad_inputs.view(*original_shape[:-1], -1), grad_weights


class RouteALinear(nn.Linear):
    def forward(self, x):
        y = RouteAFunction.apply(x, self.weight)
        return y + self.bias if self.bias is not None else y


def replace_linear(model, cls):
    for name, m in model.named_children():
        if isinstance(m, nn.Linear):
            nl = cls(m.in_features, m.out_features, bias=m.bias is not None)
            nl.weight, nl.bias = m.weight, m.bias
            setattr(model, name, nl)
        else:
            replace_linear(m, cls)


def main():
    device = torch.device("cuda")
    torch.manual_seed(42)
    total_steps = 20000
    bs, seq = 4, 128

    cfg = llama3_configs["debugmodel"]
    model_base = cfg.build().to(device).bfloat16()
    model_base.init_weights(buffer_device=device)
    vocab = cfg.vocab_size
    print(f"Model: {sum(p.numel() for p in model_base.parameters())/1e6:.2f}M params")

    model = copy.deepcopy(model_base)
    replace_linear(model, RouteALinear)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    t0 = time.time()
    for s in range(total_steps):
        torch.manual_seed(s)
        x = torch.randint(0, vocab, (bs, seq), device=device)
        tgt = torch.randint(0, vocab, (bs, seq), device=device)

        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x).view(-1, vocab), tgt.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if s % 2000 == 0 or s == total_steps - 1:
            elapsed = time.time() - t0
            print(f"step {s:6d}: loss={loss.item():.4f}  [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"Done {total_steps} steps in {elapsed:.0f}s")

    with open("/tmp/route_a_20000.json", "w") as f:
        json.dump({"route_a": losses, "config": {"variant": "route_a", "steps": total_steps}}, f)
    print("JSON saved: /tmp/route_a_20000.json")


if __name__ == "__main__":
    main()
