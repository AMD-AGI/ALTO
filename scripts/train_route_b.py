"""Route B: Hardware-aligned — E4M3 scale rounding + per-tensor scale + Hadamard + full 6 QDQ."""
import json, math, torch, torch.nn as nn, copy, time, sys
sys.stdout.reconfigure(line_buffering=True)

from torchtitan.models.llama3 import llama3_configs
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    convert_to_nvfp4, convert_from_nvfp4,
)


# ---------------------------------------------------------------------------
# E4M3 scale rounding (simulate hardware FP8 scale storage)
# ---------------------------------------------------------------------------
def _round_scales_to_e4m3(scales: torch.Tensor) -> torch.Tensor:
    """Round float32 scales to the nearest E4M3 representable value.

    E4M3 has 3 mantissa bits → we zero out the lower 20 bits of the float32
    mantissa (23 - 3 = 20) with round-to-nearest.
    """
    s_int = scales.view(torch.int32)
    s_int = (s_int + (1 << 19)) & 0xFFF00000
    return s_int.view(torch.float32)


# ---------------------------------------------------------------------------
# Block-wise Hadamard transform
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[int, torch.Tensor] = {}

def _hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    if key not in _HADAMARD_CACHE:
        H = torch.ones(1, 1, device=device, dtype=dtype)
        size = 1
        while size < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
            size *= 2
        _HADAMARD_CACHE[key] = H / math.sqrt(n)
    return _HADAMARD_CACHE[key]


def _block_hadamard(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Apply block-wise Hadamard transform along axis=0 (rows)."""
    M, K = x.shape
    if M % block_size != 0:
        return x
    H = _hadamard_matrix(block_size, x.device, x.dtype)
    x = x.reshape(M // block_size, block_size, K)
    x = torch.bmm(H.unsqueeze(0).expand(x.shape[0], -1, -1), x)
    return x.reshape(M, K)


# ---------------------------------------------------------------------------
# Route B QDQ: per-tensor scale + E4M3 scale rounding
# ---------------------------------------------------------------------------
def _qdq_b(tensor, *, axis, use_sr=False, block_size=16):
    """QDQ with per-tensor scale enabled and E4M3 scale rounding."""
    data_lp, scales, pts = convert_to_nvfp4(
        tensor, block_size=block_size, axis=axis,
        dynamic_per_tensor_scale=True, use_sr=use_sr,
    )
    scales = _round_scales_to_e4m3(scales)
    return convert_from_nvfp4(
        data_lp, scales, output_dtype=tensor.dtype,
        block_size=block_size, axis=axis,
        per_tensor_scale=pts,
    )


# ---------------------------------------------------------------------------
# Autograd function — full 6 QDQ (same structure as current) + enhancements
# ---------------------------------------------------------------------------
@torch.compiler.allow_in_graph
class RouteBFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        x_dq = _qdq_b(x_2d, axis=-1)
        w_dq = _qdq_b(weight, axis=-1)
        y = x_dq @ w_dq.T

        # axis=0 QDQ with Hadamard on activation
        x_had = _block_hadamard(x_2d, block_size=16)
        x_dq_axis0 = _qdq_b(x_had, axis=0)
        w_dq_axis0 = _qdq_b(weight, axis=0)

        ctx.save_for_backward(x_dq_axis0, w_dq_axis0)
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])

        grad_dq = _qdq_b(grad_output, axis=-1, use_sr=True)
        grad_m_dq = _qdq_b(grad_output, axis=0, use_sr=True)

        grad_inputs = grad_dq @ w_dq
        grad_weights = grad_m_dq.T @ x_dq

        return grad_inputs.view(*original_shape[:-1], -1), grad_weights


class RouteBLinear(nn.Linear):
    def forward(self, x):
        y = RouteBFunction.apply(x, self.weight)
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
    replace_linear(model, RouteBLinear)
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

    with open("/tmp/route_b_20000.json", "w") as f:
        json.dump({"route_b": losses, "config": {"variant": "route_b", "steps": total_steps}}, f)
    print("JSON saved: /tmp/route_b_20000.json")


if __name__ == "__main__":
    main()
