# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch/torchtitan licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

from typing import Optional
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import argparse

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter

from compressed_tensors import (
    SparsityCompressionConfig,
    CompressionFormat,
    ModelCompressor,
)

from torchtitan.components.checkpointer import ModelWrapper
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.loader import ConfigLoader
from torchtitan.trainer import Trainer

from alto.components.converter import ModelOptConverter
from alto.utils.compression.sparsity_metadata_config import SparsityConfigMetadata


@contextmanager
def patch_finfo():

    orig_finfo_func = torch.finfo

    def patched_finfo(dtype):
        if dtype in (
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.uint16,
                torch.int32,
                torch.uint32,
                torch.int64,
                torch.uint64,
        ):
            return torch.iinfo(dtype)
        else:
            return orig_finfo_func(dtype)

    torch.finfo = patched_finfo
    yield
    torch.finfo = orig_finfo_func


def get_model_compressor(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    model_converters: ModelConvertersContainer,
    sparsity_config: Optional[SparsityCompressionConfig] = None,
    quantization_format: Optional[str] = None,
    save_compressed: bool = True,
    disable_sparse_compression: bool = False,
):
    # modified from https://github.com/vllm-project/llm-compressor/blob/4ce1bdfc197ccd95e2be0297bfa4a7c8d7d9a614/src/llmcompressor/transformers/compression/compressed_tensors_utils.py#L152-L242

    converter = next(
        (c for c in model_converters.converters if isinstance(c, ModelOptConverter)),
        None,
    )
    if converter is None:
        return None
    modifiers = converter.recipe.modifiers

    if sparsity_config is None:
        sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(
            model,
            check_only_modifiers=True,
            modifiers=modifiers,
        )
        if sparsity_structure is not None:
            sparsity_config = SparsityConfigMetadata.from_pretrained(
                model,
                state_dict=state_dict,
                compress=save_compressed,
                quantization_format=quantization_format,
                disable_sparse_compression=disable_sparse_compression,
                sparsity_structure=sparsity_structure,
            )
        else:
            sparsity_config = None
    else:
        if sparsity_config.sparsity_structure is None:
            print(
                "SparsityConfigMetadata provided without indicating ",
                "the sparsity structure. Sparisty will be inferred from the model. "
                "Consider providing the structure to skip this step ",
            )
            sparsity_config.sparsity_structure = (SparsityConfigMetadata.infer_sparsity_structure(model))

    if not save_compressed:
        if quantization_format not in (None, CompressionFormat.dense.value):
            raise ValueError("A quantizatiom format was provided but "
                             "save_compressed is set to False. "
                             "A compression format can only be applied when "
                             "saving the model compressed")
        quantization_format = CompressionFormat.dense.value

    print(f"sparsity_config: {sparsity_config}")
    print(f"quantization_format: {quantization_format}")

    return ModelCompressor.from_pretrained_model(
        model,
        sparsity_config_or_format=sparsity_config,
        quantization_format=quantization_format,
    )


def _strip_quantization_config(output_dir: str):
    """Remove quantization_config from config.json so the exported model
    loads as a plain HF model without triggering quantizer logic."""
    import os
    cfg_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


@torch.inference_mode()
def convert_to_hf(
    config: Trainer.Config,
    input_dir: str,
    output_dir: str,
    hf_assets_path: str,
    export_dtype: str,
    save_compressed: bool = True,
    disable_sparse_compression: bool = False,
):
    # load model and model args so that we can get the state dict shape
    model_config = copy.deepcopy(config.model)

    model_converters = config.model_converters.build(
        parallelism_context=None,
        model_compile_enabled=False,
    )
    model_converters.convert_config(model_config)

    with torch.device("cpu"):
        model = model_config.build()
        model_converters.convert(model)
    model_converters.post_initialization([model])
    model_converters.pre_step([model])
    model_converters.finalize([model])
    wrapped_model = ModelWrapper(model)

    adapter_cls = type(model).state_dict_adapter_cls
    assert (
        adapter_cls
        is not None), "trying to convert checkpoint from DCP to HF safetensors format, but the model has no state dict adapter."
    sd_adapter = adapter_cls(model_config, hf_assets_path)

    state_dict = wrapped_model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    if save_compressed:
        compressor = get_model_compressor(
            model,
            state_dict,
            model_converters,
            save_compressed=save_compressed,
            disable_sparse_compression=disable_sparse_compression,
        )
        if compressor is not None:
            if compressor.quantization_config is not None:
                compressor.quantization_config.ignore = sd_adapter.map_ignore_list_to_hf(
                    compressor.quantization_config.ignore)
            if compressor.sparsity_config is not None:
                compressor.sparsity_config.ignore = sd_adapter.map_ignore_list_to_hf(compressor.sparsity_config.ignore)

            state_dict = compressor.compress(model)
            compressor.update_config(output_dir)

    hf_state_dict = sd_adapter.to_hf(state_dict)
    sd_adapter.update_storage_plan(hf_state_dict)
    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) if v.is_floating_point() else v for k, v in hf_state_dict.items()}

    with patch_finfo():
        dcp.save(
            hf_state_dict,
            storage_writer=storage_writer,
        )

    if not save_compressed:
        _strip_quantization_config(output_dir)


def eval_tasks(model_dir: str, tasks: list[str]):
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model_dir,
        },
        tasks=tasks,
        device="cuda:0",
        batch_size=1,
        log_samples=False,
    )
    print(results["results"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument("module", type=str, help="e.g. alto.models.llama3.config_registry")
    parser.add_argument("config", type=str, help="e.g. llama3_8b_gptq")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument(
        "--disable_sparse_compression",
        action="store_true",
        help="Disable sparse compression (default: False)",
    )
    parser.add_argument(
        "--save_uncompressed",
        action="store_true",
        help="Save uncompressed checkpoint (default: False)",
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip checkpoint export (default: False)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to evaluate (default: [])",
        default=[],
    )
    args = parser.parse_args()

    config = ConfigLoader().load(["--module", args.module, "--config", args.config])

    dump_folder = Path(config.dump_folder)
    hf_assets_path = Path(config.hf_assets_path)
    checkpoint_folder = config.checkpointer.folder if config.checkpointer is not None else "checkpoint"
    input_dir = dump_folder / checkpoint_folder / f"step-{config.training.steps}"
    output_dir = dump_folder / "hf"

    if not args.skip_export:
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["*.json", "*.py"]:
            for extra_file in hf_assets_path.glob(pattern):
                shutil.copy(extra_file, output_dir)

        convert_to_hf(
            config,
            input_dir.as_posix(),
            output_dir.as_posix(),
            hf_assets_path.as_posix(),
            args.export_dtype,
            save_compressed=not args.save_uncompressed,
            disable_sparse_compression=args.disable_sparse_compression,
        )
        sharded_output_dir = output_dir / "sharded"
        shutil.rmtree(sharded_output_dir, ignore_errors=True)

    if args.tasks:
        eval_tasks(output_dir.as_posix(), args.tasks)
