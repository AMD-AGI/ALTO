# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MXFP4 PNQ reference, invariant test, and optional kernel cross-validation.

Run the fast CPU invariant test with pytest:
    pytest -q tests/unittest/mxfp4/test_mxfp_pnq.py

Run the GPU cross-validation against the pre-PNQ worktree:
    python tests/unittest/mxfp4/test_mxfp_pnq.py --cross-validate \
        --create-baseline-worktree --causal
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USER", f"uid{os.getuid()}")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(tempfile.gettempdir()) / f"torchinductor-{os.getuid()}"))
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / f"triton-{os.getuid()}"))

import torch

ALTO_PNQ_PARENT = "2543fc04ce5be1052e64ecc68a2edf045c46d203"
MINDIE_CANDIDATE_COMMIT = "ff8ebdd1a67d20431803134f57870f75428e4ada"


def _install_quantization_test_stubs() -> None:
    """Load the MXFP4 PyTorch reference without importing optional kernels."""
    alto_root = Path(__file__).resolve().parents[3]
    package_paths = {
        "alto": alto_root / "alto",
        "alto.kernels": alto_root / "alto" / "kernels",
        "alto.kernels.fp4": alto_root / "alto" / "kernels" / "fp4",
        "alto.kernels.fp4.mxfp4": alto_root / "alto" / "kernels" / "fp4" / "mxfp4",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    quantization_name = "alto.kernels.fp4.mxfp4.mxfp_quantization"
    if quantization_name not in sys.modules:
        quantization = types.ModuleType(quantization_name)
        quantization.BLOCK_SIZE_DEFAULT = 32
        sys.modules[quantization_name] = quantization


_install_quantization_test_stubs()

try:
    from .utils import convert_from_mxfp4_pytorch, convert_to_mxfp4_pytorch
except ImportError:
    from utils import convert_from_mxfp4_pytorch, convert_to_mxfp4_pytorch


def _qdq_alto(x: torch.Tensor, *, is_2d_block: bool) -> torch.Tensor:
    packed, scales = convert_to_mxfp4_pytorch(x, axis=-1, is_2d_block=is_2d_block)
    return convert_from_mxfp4_pytorch(
        packed,
        scales,
        output_dtype=torch.float32,
        axis=-1,
        is_2d_block=is_2d_block,
    )


def _expand_kv(k: torch.Tensor, v: torch.Tensor, num_query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = k.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError(f"num_query_heads={num_query_heads} is not divisible by num_kv_heads={num_kv_heads}")
    group_size = num_query_heads // num_kv_heads
    return k.repeat_interleave(group_size, dim=1), v.repeat_interleave(group_size, dim=1)


def mxfp4_pnq_reference_bhsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    pnq: bool,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Run ALTO QDQ with the MindIE-SD-style tiled PNQ recurrence."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must use [batch, heads, sequence, dim]")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
        raise ValueError("batch size and KV-head counts must agree")
    if q.shape[-2] % 32 or k.shape[-2] % 32 or q.shape[-1] % 32 or v.shape[-1] % 32:
        raise ValueError("sequence lengths and dimensions must be divisible by 32")

    q_dq = _qdq_alto(q, is_2d_block=True).float()
    k_dq = _qdq_alto(k, is_2d_block=True).float()
    v_dq = _qdq_alto(v, is_2d_block=True).float()
    k_dq, v_dq = _expand_kv(k_dq, v_dq, q.shape[1])

    batch, heads, seqlen_q, head_dim = q.shape
    output = torch.empty(batch, heads, seqlen_q, v.shape[-1], dtype=torch.float32, device=q.device)
    sm_scale = head_dim**-0.5
    seqlen_k = k.shape[-2]

    for batch_idx in range(batch):
        for head_idx in range(heads):
            for query_start in range(0, seqlen_q, block_m):
                query_end = min(query_start + block_m, seqlen_q)
                query = q_dq[batch_idx, head_idx, query_start:query_end]
                running_max = torch.full((query_end - query_start,), -torch.inf, device=q.device)
                running_sum = torch.zeros_like(running_max)
                accumulator = torch.zeros(query_end - query_start, v.shape[-1], device=q.device)

                for key_start in range(0, seqlen_k, block_n):
                    key_end = min(key_start + block_n, seqlen_k)
                    if causal and key_start >= query_end:
                        break

                    key = k_dq[batch_idx, head_idx, key_start:key_end]
                    value = v_dq[batch_idx, head_idx, key_start:key_end]
                    scores = query @ key.transpose(0, 1) * sm_scale
                    if causal:
                        query_positions = torch.arange(query_start, query_end, device=q.device) + (seqlen_k - seqlen_q)
                        key_positions = torch.arange(key_start, key_end, device=q.device)
                        scores = scores.masked_fill(key_positions.unsqueeze(0) > query_positions.unsqueeze(1), -torch.inf)

                    next_max = torch.maximum(running_max, scores.amax(dim=1))
                    alpha = torch.exp(running_max - next_max)
                    probabilities = torch.nan_to_num(torch.exp(scores - next_max.unsqueeze(1)), nan=0.0)
                    quantized_probabilities = _qdq_alto(probabilities, is_2d_block=False)

                    accumulator = accumulator * alpha.unsqueeze(1) + quantized_probabilities @ value
                    normalizer = quantized_probabilities if pnq else probabilities
                    running_sum = running_sum * alpha + normalizer.sum(dim=1)
                    running_max = next_max

                output[batch_idx, head_idx, query_start:query_end] = accumulator / running_sum.unsqueeze(1)

    return output.to(q.dtype)


def fp32_reference_bhsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    k, v = _expand_kv(k, v, q.shape[1])
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        is_causal=causal,
        scale=q.shape[-1] ** -0.5,
    ).to(q.dtype)


def test_pnq_preserves_constant_value_attention():
    """For V=1, PNQ must preserve the attention probability-mass invariant."""
    torch.manual_seed(1234)
    q = torch.randn(1, 32, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16)
    v = torch.ones(1, 8, 64, 64, dtype=torch.bfloat16)

    output_pnq = mxfp4_pnq_reference_bhsd(q, k, v, causal=True, pnq=True)
    output_without_pnq = mxfp4_pnq_reference_bhsd(q, k, v, causal=True, pnq=False)

    assert torch.allclose(output_pnq, torch.ones_like(output_pnq), atol=0.0, rtol=0.0)
    assert not torch.allclose(output_without_pnq, torch.ones_like(output_without_pnq), atol=1e-2, rtol=0.0)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        text=True,
    ).strip()


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    diff = actual - reference
    return {
        "mae": float(diff.abs().mean()),
        "max_abs_error": float(diff.abs().max()),
        "relative_l2": float(diff.norm() / reference.norm().clamp_min(1e-12)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(reference.flatten(), actual.flatten(), dim=0)),
        "norm_ratio": float(actual.norm() / reference.norm().clamp_min(1e-12)),
    }


def _install_alto_stub(alto_root: Path) -> None:
    for name in list(sys.modules):
        if name == "alto" or name.startswith("alto."):
            sys.modules.pop(name)
    sys.path.insert(0, str(alto_root))
    stub = types.ModuleType("alto")
    stub.__path__ = [str(alto_root / "alto")]
    sys.modules["alto"] = stub


def _run_alto_worker(alto_root: Path, input_path: Path, output_path: Path) -> None:
    _install_alto_stub(alto_root)
    sys.modules.pop("alto.kernels.fp4.mxfp4.mxfp_quantization", None)
    attention = importlib.import_module(
        "alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4"
    ).triton_attention_mxfp4
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    q, k, v = (payload[name].to("cuda") for name in ("q", "k", "v"))
    output = attention(
        q,
        k,
        v,
        bias=None,
        alibi_slopes=None,
        sm_scale=float(payload["sm_scale"]),
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=q.shape[-2],
        max_seqlens_k=k.shape[-2],
        causal=bool(payload["causal"]),
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]
    torch.save(output.cpu(), output_path)


def _invoke_worker(script: Path, alto_root: Path, input_path: Path, output_path: Path) -> torch.Tensor:
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-alto-worker",
            "--alto-root",
            str(alto_root),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )
    return torch.load(output_path, map_location="cpu", weights_only=True)


def _ensure_baseline_worktree(alto_root: Path, baseline_root: Path) -> None:
    if baseline_root.exists():
        if _git(baseline_root, "rev-parse", "HEAD") == ALTO_PNQ_PARENT:
            return
        raise RuntimeError(f"{baseline_root} is not the required pre-PNQ worktree")
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={alto_root}",
            "-C",
            str(alto_root),
            "worktree",
            "add",
            "--detach",
            str(baseline_root),
            ALTO_PNQ_PARENT,
        ],
        check=True,
    )


def _case_inputs(args: argparse.Namespace, value_mode: str) -> dict[str, torch.Tensor | float | bool]:
    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.query_heads, args.seqlen_q, args.head_dim, dtype=torch.bfloat16)
    k = torch.randn(args.batch, args.kv_heads, args.seqlen_k, args.head_dim, dtype=torch.bfloat16)
    v = torch.ones(args.batch, args.kv_heads, args.seqlen_k, args.value_dim, dtype=torch.bfloat16)
    if value_mode != "ones":
        v = torch.randn_like(v)
        if value_mode == "biased":
            v += 2.0
    return {"q": q, "k": k, "v": v, "sm_scale": args.head_dim**-0.5, "causal": args.causal}


def _write_report(output_dir: Path, metadata: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# MXFP4 PNQ 交叉验证结果",
        "",
        "MindIE-SD-style recurrence with ALTO QDQ; this report is generated and not tracked.",
        "",
        "| case | implementation | MAE vs FP32 | max abs | relative L2 | cosine | norm ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metric = row["metrics"]
        lines.append(
            f"| {row['case']} | {row['implementation']} | {metric['mae']:.6g} | "
            f"{metric['max_abs_error']:.6g} | {metric['relative_l2']:.6g} | "
            f"{metric['cosine_similarity']:.8f} | {metric['norm_ratio']:.8f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def _cross_validate(args: argparse.Namespace) -> None:
    alto_root = args.alto_root.resolve()
    baseline_root = args.baseline_alto_root.resolve() if args.baseline_alto_root else None
    if args.create_baseline_worktree:
        baseline_root = baseline_root or Path(tempfile.gettempdir()) / "alto-mxfp4-pnq-baseline"
        _ensure_baseline_worktree(alto_root, baseline_root)
    if baseline_root is None:
        raise ValueError("pass --baseline-alto-root or --create-baseline-worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("cross-validation requires a ROCm/CUDA PyTorch environment")

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="alto-mxfp4-pnq-"))
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alto_pnq_commit": _git(alto_root, "rev-parse", "HEAD"),
        "alto_no_pnq_commit": _git(baseline_root, "rev-parse", "HEAD"),
        "mindiesd_candidate_commit": MINDIE_CANDIDATE_COMMIT,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "config": {
            "batch": args.batch,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "seqlen_q": args.seqlen_q,
            "seqlen_k": args.seqlen_k,
            "head_dim": args.head_dim,
            "value_dim": args.value_dim,
            "causal": args.causal,
            "qkv_quantization": "ALTO 2D MXFP4 PyTorch reference",
            "p_quantization": "ALTO q=7, 1D E2M1 PyTorch reference",
        },
    }
    rows = []
    for value_mode in ("random", "biased", "ones"):
        payload = _case_inputs(args, value_mode)
        input_path = output_dir / f"{value_mode}_input.pt"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, input_path)
        fp32 = fp32_reference_bhsd(payload["q"], payload["k"], payload["v"], causal=args.causal)
        outputs = {
            "MindIE-derived reference without PNQ": mxfp4_pnq_reference_bhsd(
                payload["q"], payload["k"], payload["v"], causal=args.causal, pnq=False
            ),
            "MindIE-derived reference with PNQ": mxfp4_pnq_reference_bhsd(
                payload["q"], payload["k"], payload["v"], causal=args.causal, pnq=True
            ),
            "ALTO without PNQ": _invoke_worker(Path(__file__), baseline_root, input_path, output_dir / f"{value_mode}_no_pnq.pt"),
            "ALTO with PNQ": _invoke_worker(Path(__file__), alto_root, input_path, output_dir / f"{value_mode}_pnq.pt"),
        }
        rows.extend({"case": value_mode, "implementation": name, "metrics": _metrics(fp32, output)} for name, output in outputs.items())

    _write_report(output_dir, metadata, rows)
    print(f"wrote {output_dir / 'report.md'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--alto-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--baseline-alto-root", type=Path)
    parser.add_argument("--create-baseline-worktree", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seqlen-q", type=int, default=64)
    parser.add_argument("--seqlen-k", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--run-alto-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        parser.error("--query-heads must be divisible by --kv-heads")
    for name in ("seqlen_q", "seqlen_k", "head_dim", "value_dim"):
        if getattr(args, name) % 32:
            parser.error(f"--{name.replace('_', '-')} must be divisible by 32")
    return args


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.run_alto_worker:
        _run_alto_worker(arguments.alto_root.resolve(), arguments.input.resolve(), arguments.output.resolve())
    elif arguments.cross_validate:
        _cross_validate(arguments)
