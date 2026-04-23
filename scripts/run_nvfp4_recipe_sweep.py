"""Quick recipe sweep for NVFP4 low-precision training.

Runs a small llama3 debugmodel training comparison across BF16, MXFP4-light,
and several NVFP4 recipes. Designed for fast 1K/3K-step validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from torchtitan.models.llama3 import llama3_configs

from alto.kernels.fp4.mxfp4.mxfp_quantization import (
    convert_from_mxfp4,
    convert_to_mxfp4,
)
from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    convert_from_nvfp4,
    convert_to_nvfp4,
)


@dataclass(frozen=True)
class Nvfp4Recipe:
    name: str
    use_axis0: bool
    fwd_x_sr: bool
    fwd_w_sr: bool
    grad_sr: bool
    pts_x: bool
    pts_w: bool
    pts_grad: bool
    tail_bf16_layers: int = 0
    wgrad_hadamard: bool = False
    weight_2d_only: bool = False


RECIPES = {
    "nvfp4_current": Nvfp4Recipe(
        name="nvfp4_current",
        use_axis0=True,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
    ),
    "nvfp4_6qdq_xsr": Nvfp4Recipe(
        name="nvfp4_6qdq_xsr",
        use_axis0=True,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
    ),
    "nvfp4_3qdq_rne": Nvfp4Recipe(
        name="nvfp4_3qdq_rne",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
    ),
    "nvfp4_route_a": Nvfp4Recipe(
        name="nvfp4_route_a",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
    ),
    "nvfp4_route_a_pts_xg": Nvfp4Recipe(
        name="nvfp4_route_a_pts_xg",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=True,
        pts_w=False,
        pts_grad=True,
    ),
    "nvfp4_route_a_pts_all": Nvfp4Recipe(
        name="nvfp4_route_a_pts_all",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=True,
        pts_w=True,
        pts_grad=True,
    ),
    "nvfp4_3qdq_rne_tail1": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail1",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=1,
    ),
    "nvfp4_3qdq_rne_tail2": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail2",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
    ),
    "nvfp4_3qdq_rne_tail3": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail3",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=3,
    ),
    "nvfp4_3qdq_rne_tail4": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail4",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=4,
    ),
    "nvfp4_3qdq_rne_tail1_h16": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail1_h16",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=1,
        wgrad_hadamard=True,
    ),
    "nvfp4_3qdq_rne_tail2_h16": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail2_h16",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
        wgrad_hadamard=True,
    ),
    "nvfp4_route_a_tail2": Nvfp4Recipe(
        name="nvfp4_route_a_tail2",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
    ),
    "nvfp4_route_a_tail3": Nvfp4Recipe(
        name="nvfp4_route_a_tail3",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=3,
    ),
    "nvfp4_route_a_tail4": Nvfp4Recipe(
        name="nvfp4_route_a_tail4",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=4,
    ),
    "nvfp4_3qdq_rne_tail2_w2d": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail2_w2d",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
        weight_2d_only=True,
    ),
    "nvfp4_3qdq_rne_tail3_w2d": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail3_w2d",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=3,
        weight_2d_only=True,
    ),
    "nvfp4_3qdq_rne_tail2_h16v2": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail2_h16v2",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
        wgrad_hadamard=True,
    ),
    "nvfp4_3qdq_rne_tail2_w2d_h16v2": Nvfp4Recipe(
        name="nvfp4_3qdq_rne_tail2_w2d_h16v2",
        use_axis0=False,
        fwd_x_sr=False,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=2,
        wgrad_hadamard=True,
        weight_2d_only=True,
    ),
    "nvfp4_route_a_tail4_w2d": Nvfp4Recipe(
        name="nvfp4_route_a_tail4_w2d",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=4,
        weight_2d_only=True,
    ),
    "nvfp4_route_a_tail4_h16v2": Nvfp4Recipe(
        name="nvfp4_route_a_tail4_h16v2",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=4,
        wgrad_hadamard=True,
    ),
    "nvfp4_route_a_tail4_w2d_h16v2": Nvfp4Recipe(
        name="nvfp4_route_a_tail4_w2d_h16v2",
        use_axis0=False,
        fwd_x_sr=True,
        fwd_w_sr=False,
        grad_sr=True,
        pts_x=False,
        pts_w=False,
        pts_grad=False,
        tail_bf16_layers=4,
        wgrad_hadamard=True,
        weight_2d_only=True,
    ),
}


_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def block_hadamard_left(
    tensor: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    rows, cols = tensor.shape
    if rows < block_size or rows % block_size != 0:
        return tensor

    key = (block_size, tensor.device, tensor.dtype)
    if key not in _HADAMARD_CACHE:
        h = torch.ones((1, 1), device=tensor.device, dtype=tensor.dtype)
        size = 1
        while size < block_size:
            h = torch.cat(
                (
                    torch.cat((h, h), dim=1),
                    torch.cat((h, -h), dim=1),
                ),
                dim=0,
            )
            size *= 2
        _HADAMARD_CACHE[key] = h / math.sqrt(block_size)

    h = _HADAMARD_CACHE[key]
    tiles = tensor.reshape(rows // block_size, block_size, cols)
    out = torch.bmm(h.unsqueeze(0).expand(tiles.shape[0], -1, -1), tiles)
    return out.reshape(rows, cols)


def qdq_nvfp4(
    tensor: torch.Tensor,
    *,
    axis: int,
    use_sr: bool,
    use_per_tensor_scale: bool,
    is_2d_block: bool = False,
    block_size: int = 16,
) -> torch.Tensor:
    data_lp, scales, pts = convert_to_nvfp4(
        tensor,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        dynamic_per_tensor_scale=use_per_tensor_scale,
        use_sr=use_sr,
    )
    return convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=tensor.dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        per_tensor_scale=pts if use_per_tensor_scale else None,
    )


class Nvfp4RecipeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        use_axis0: bool,
        fwd_x_sr: bool,
        fwd_w_sr: bool,
        grad_sr: bool,
        pts_x: bool,
        pts_w: bool,
        pts_grad: bool,
        wgrad_hadamard: bool,
        weight_2d_only: bool,
    ):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        x_dq = qdq_nvfp4(
            x_2d, axis=-1, use_sr=fwd_x_sr, use_per_tensor_scale=pts_x
        )
        w_dq = qdq_nvfp4(
            weight,
            axis=-1,
            use_sr=fwd_w_sr,
            use_per_tensor_scale=pts_w,
            is_2d_block=weight_2d_only,
        )
        y = x_dq @ w_dq.T

        if use_axis0:
            x_saved = qdq_nvfp4(
                x_2d, axis=0, use_sr=fwd_x_sr, use_per_tensor_scale=pts_x
            )
            w_saved = qdq_nvfp4(
                weight, axis=0, use_sr=fwd_w_sr, use_per_tensor_scale=pts_w
            )
        else:
            x_saved = x_dq
            w_saved = w_dq

        ctx.save_for_backward(x_saved, w_saved)
        ctx.x_hp = x_2d
        ctx.use_axis0 = use_axis0
        ctx.grad_sr = grad_sr
        ctx.pts_grad = pts_grad
        ctx.wgrad_hadamard = wgrad_hadamard
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_saved, w_saved = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])

        grad_dq = qdq_nvfp4(
            grad_output,
            axis=-1,
            use_sr=ctx.grad_sr,
            use_per_tensor_scale=ctx.pts_grad,
        )
        if ctx.use_axis0:
            grad_m_dq = qdq_nvfp4(
                grad_output,
                axis=0,
                use_sr=ctx.grad_sr,
                use_per_tensor_scale=ctx.pts_grad,
            )
        else:
            grad_m_dq = grad_dq

        grad_inputs = grad_dq @ w_saved
        if ctx.wgrad_hadamard:
            grad_w_in = block_hadamard_left(grad_output, block_size=16)
            x_w_in = block_hadamard_left(ctx.x_hp, block_size=16)
            grad_m_dq = qdq_nvfp4(
                grad_w_in,
                axis=0,
                use_sr=ctx.grad_sr,
                use_per_tensor_scale=ctx.pts_grad,
            )
            x_w_dq = qdq_nvfp4(
                x_w_in,
                axis=0,
                use_sr=False,
                use_per_tensor_scale=False,
            )
            grad_weights = grad_m_dq.T @ x_w_dq
        else:
            grad_weights = grad_m_dq.T @ x_saved
        return (
            grad_inputs.view(*original_shape[:-1], -1),
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MXFP4BF16Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor):
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        x_lp, x_s = convert_to_mxfp4(x_2d, axis=-1, use_sr=False)
        x_dq = convert_from_mxfp4(x_lp, x_s, output_dtype=x.dtype, axis=-1)
        w_lp, w_s = convert_to_mxfp4(weight, axis=-1, use_sr=False)
        w_dq = convert_from_mxfp4(w_lp, w_s, output_dtype=weight.dtype, axis=-1)

        ctx.save_for_backward(x_dq, w_dq)
        y = x_dq @ w_dq.T
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])
        g_lp, g_s = convert_to_mxfp4(grad_output, axis=-1, use_sr=True)
        g_dq = convert_from_mxfp4(g_lp, g_s, output_dtype=grad_output.dtype, axis=-1)
        grad_inputs = g_dq @ w_dq
        grad_weights = g_dq.T @ x_dq
        return grad_inputs.view(*original_shape[:-1], -1), grad_weights


class BF16Linear(nn.Linear):
    pass


def make_nvfp4_linear(recipe: Nvfp4Recipe):
    class RecipeLinear(nn.Linear):
        def forward(self, x: torch.Tensor):
            y = Nvfp4RecipeFunction.apply(
                x,
                self.weight,
                recipe.use_axis0,
                recipe.fwd_x_sr,
                recipe.fwd_w_sr,
                recipe.grad_sr,
                recipe.pts_x,
                recipe.pts_w,
                recipe.pts_grad,
                recipe.wgrad_hadamard,
                recipe.weight_2d_only,
            )
            return y + self.bias if self.bias is not None else y

    RecipeLinear.__name__ = f"{recipe.name}_Linear"
    return RecipeLinear


class MXFP4BF16Linear(nn.Linear):
    def forward(self, x: torch.Tensor):
        y = MXFP4BF16Function.apply(x, self.weight)
        return y + self.bias if self.bias is not None else y


class MXFP4NativeLinear(nn.Linear):
    def forward(self, x: torch.Tensor):
        y = _to_mxfp4_then_scaled_mm(
            x,
            self.weight,
            use_2dblock_x=False,
            use_2dblock_w=False,
            use_sr_grad=True,
            use_dge=False,
            use_hadamard=False,
        )
        return y + self.bias if self.bias is not None else y


def replace_linear_modules(
    model: nn.Module,
    linear_cls: type[nn.Linear],
    *,
    keep_tail_layers_bf16: int = 0,
    total_layers: int | None = None,
    prefix: str = "",
) -> None:
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            keep_bf16 = False
            if keep_tail_layers_bf16 > 0 and total_layers is not None:
                parts = full_name.split(".")
                if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    keep_bf16 = layer_idx >= total_layers - keep_tail_layers_bf16
            if keep_bf16:
                continue
            new_linear = linear_cls(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            new_linear.weight = child.weight
            new_linear.bias = child.bias
            setattr(model, name, new_linear)
        else:
            replace_linear_modules(
                child,
                linear_cls,
                keep_tail_layers_bf16=keep_tail_layers_bf16,
                total_layers=total_layers,
                prefix=full_name,
            )


def build_models(
    base_model: nn.Module,
    recipe_names: list[str],
) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    total_layers = len(base_model.layers) if hasattr(base_model, "layers") else None
    for name in recipe_names:
        model = copy.deepcopy(base_model)
        if name == "bf16":
            pass
        elif name == "mxfp4_native":
            replace_linear_modules(model, MXFP4NativeLinear, total_layers=total_layers)
        elif name in {"mxfp4_light", "mxfp4_bf16"}:
            replace_linear_modules(model, MXFP4BF16Linear, total_layers=total_layers)
        else:
            recipe = RECIPES[name]
            replace_linear_modules(
                model,
                make_nvfp4_linear(recipe),
                keep_tail_layers_bf16=recipe.tail_bf16_layers,
                total_layers=total_layers,
            )
        models[name] = model
    return models


def summarize(losses: list[float], tail: int) -> dict[str, float]:
    arr = torch.tensor(losses, dtype=torch.float32)
    tail_arr = arr[-tail:]
    return {
        "final": float(arr[-1]),
        "tail_avg": float(tail_arr.mean()),
        "tail_std": float(tail_arr.std(unbiased=False)),
        "best": float(arr.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--recipes",
        type=str,
        default="bf16,mxfp4_bf16,nvfp4_current,nvfp4_6qdq_xsr,nvfp4_3qdq_rne,nvfp4_route_a,nvfp4_route_a_pts_xg,nvfp4_route_a_pts_all",
    )
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--output", type=str, default="/tmp/nvfp4_recipe_sweep.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(42)

    recipe_names = [item.strip() for item in args.recipes.split(",") if item.strip()]
    cfg = llama3_configs["debugmodel"]
    base_model = cfg.build().to(device).bfloat16()
    base_model.init_weights(buffer_device=device)
    vocab = cfg.vocab_size

    models = build_models(base_model, recipe_names)
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr)
        for name, model in models.items()
    }
    losses = {name: [] for name in recipe_names}

    print(
        f"Model: {sum(p.numel() for p in base_model.parameters())/1e6:.2f}M params | "
        f"recipes={recipe_names}",
        flush=True,
    )

    t0 = time.time()
    for step in range(args.steps):
        torch.manual_seed(step)
        x = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
        tgt = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)

        for name in recipe_names:
            model = models[name]
            optimizer = optimizers[name]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                model(x).view(-1, vocab),
                tgt.view(-1),
            )
            loss.backward()
            optimizer.step()
            losses[name].append(float(loss.item()))

        if step % 200 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            tail = min(args.tail, step + 1)
            metrics = []
            for name in recipe_names:
                tail_avg = sum(losses[name][-tail:]) / tail
                metrics.append(f"{name}={tail_avg:.4f}")
            print(
                f"step {step:5d} [{elapsed:.0f}s] tail{tail}: " + " ".join(metrics),
                flush=True,
            )

    summary = {name: summarize(losses[name], min(args.tail, len(losses[name]))) for name in recipe_names}
    bf16_tail = summary["bf16"]["tail_avg"] if "bf16" in summary else None
    mxfp4_key = "mxfp4_light" if "mxfp4_light" in summary else "mxfp4_bf16" if "mxfp4_bf16" in summary else None
    mxfp4_tail = summary[mxfp4_key]["tail_avg"] if mxfp4_key is not None else None

    ranking = []
    for name in recipe_names:
        item = dict(summary[name])
        if bf16_tail is not None:
            item["gap_vs_bf16"] = item["tail_avg"] - bf16_tail
        if mxfp4_tail is not None:
            item["gap_vs_mxfp4"] = item["tail_avg"] - mxfp4_tail
        ranking.append((name, item))

    ranking.sort(key=lambda kv: kv[1]["tail_avg"])
    print("\n=== Summary (sorted by tail_avg) ===", flush=True)
    for name, item in ranking:
        extras = []
        if "gap_vs_bf16" in item:
            extras.append(f"gap_bf16={item['gap_vs_bf16']:+.4f}")
        if "gap_vs_mxfp4" in item:
            extras.append(f"gap_mxfp4={item['gap_vs_mxfp4']:+.4f}")
        print(
            f"{name:22s} tail_avg={item['tail_avg']:.4f} "
            f"tail_std={item['tail_std']:.4f} best={item['best']:.4f} "
            + " ".join(extras),
            flush=True,
        )

    payload = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "recipes": recipe_names,
        "losses": losses,
        "summary": {name: item for name, item in ranking},
    }
    with open(args.output, "w") as f:
        json.dump(payload, f)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
