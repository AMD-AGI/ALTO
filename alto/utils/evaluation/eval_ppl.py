# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Perplexity evaluation on C4 (val), WikiText-2, and PTB (val).

Usage:
    python -m alto.utils.evaluation.eval_ppl \
        --model_path /path/to/hf_model \
        --datasets c4 wikitext2 ptb \
        --seqlen 2048 \
        --dtype bfloat16
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Dataset loaders – each returns a single 1-D LongTensor of concatenated
# token ids ready for strided PPL evaluation.
# ---------------------------------------------------------------------------

def _load_c4(tokenizer, seqlen: int, n_samples: int = 1100, seed: int = 0,
             cache_dir: str | None = None) -> torch.Tensor:
    ds = load_dataset(
        "allenai/c4", "en",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed)

    texts = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        texts.append(sample["text"])

    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    return enc.input_ids[0]


def _load_wikitext2(tokenizer, seqlen: int,
                    cache_dir: str | None = None) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                      split="test", cache_dir=cache_dir)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids[0]


def _load_ptb(tokenizer, seqlen: int,
              cache_dir: str | None = None) -> torch.Tensor:
    ds = load_dataset("ptb_text_only", "penn_treebank",
                      split="validation", cache_dir=cache_dir,
                      trust_remote_code=True)
    text = "\n\n".join(ds["sentence"])
    enc = tokenizer(text, return_tensors="pt")
    return enc.input_ids[0]


DATASET_LOADERS = {
    "c4": _load_c4,
    "wikitext2": _load_wikitext2,
    "ptb": _load_ptb,
}


# ---------------------------------------------------------------------------
# Strided PPL evaluation (non-overlapping windows)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    seqlen: int,
    device: torch.device,
    batch_size: int = 1,
) -> float:
    """Compute perplexity with non-overlapping stride equal to seqlen."""
    n_tokens = input_ids.numel()
    n_windows = n_tokens // seqlen
    input_ids = input_ids[: n_windows * seqlen].view(n_windows, seqlen)

    loss_fn = CrossEntropyLoss(reduction="none")
    total_nll = 0.0
    total_count = 0

    for start in tqdm(range(0, n_windows, batch_size), desc="eval", leave=False):
        batch = input_ids[start : start + batch_size].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        total_nll += nll.sum().item()
        total_count += nll.numel()

    return math.exp(total_nll / total_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPL evaluation (C4 / WikiText-2 / PTB)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--datasets", nargs="+", default=["c4", "wikitext2", "ptb"],
                        choices=list(DATASET_LOADERS.keys()),
                        help="Datasets to evaluate on")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of windows per forward pass)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"],
                        help="Model weight dtype")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs (>1 enables device_map=auto)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace datasets cache directory")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional JSON file to save results")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading model from {args.model_path} (dtype={args.dtype}) ...")
    load_kwargs = dict(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.num_gpus > 1:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": args.device}

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()

    device = next(model.parameters()).device

    results: dict[str, float] = {}
    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  |  seqlen: {args.seqlen}")
        print(f"{'='*60}")

        loader = DATASET_LOADERS[ds_name]
        input_ids = loader(tokenizer, args.seqlen, cache_dir=args.cache_dir)

        n_tokens = input_ids.numel()
        n_windows = n_tokens // args.seqlen
        print(f"  Tokens: {n_tokens:,}  |  Windows: {n_windows}")

        ppl = evaluate_ppl(model, input_ids, args.seqlen, device, args.batch_size)
        results[ds_name] = round(ppl, 4)
        print(f"  PPL: {ppl:.4f}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for ds_name, ppl in results.items():
        print(f"  {ds_name:>12s}:  {ppl:.4f}")

    if args.output_path:
        out = {
            "model_path": args.model_path,
            "seqlen": args.seqlen,
            "dtype": args.dtype,
            "results": results,
        }
        os.makedirs(Path(args.output_path).parent, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
