# Modifications Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import argparse
from pathlib import Path
import shutil
from contextlib import contextmanager

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter

from compressed_tensors import (
    SparsityCompressionConfig,
    CompressionFormat,
    ModelCompressor,
)

from torchtitan.config.manager import ConfigManager
from torchtitan.config.job_config import JobConfig
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.protocols.model_converter import (
    build_model_converters,
    ModelConvertersContainer,
)

from modeloptimizer.components.converter import ModelOptConverter
from modeloptimizer.utils.compression.sparsity_metadata_config import SparsityConfigMetadata


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


def hot_fix_for_tied_word_embeddings(model: torch.nn.Module,
                                     compressor: ModelCompressor):
    if not getattr(model, "tie_word_embeddings", False):
        return
    # in the current impl, lm_head is not a linear layer,
    # so we need to add it to the ignore list manually
    if compressor.quantization_config is not None:
        compressor.quantization_config.ignore.append("lm_head")
    if compressor.sparsity_config is not None:
        compressor.sparsity_config.ignore.append("lm_head")


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

    # get the first modifier of ModelOptConverter class from model_converters
    converter = next(
        (c for c in model_converters.converters
         if isinstance(c, ModelOptConverter)),
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
            sparsity_config.sparsity_structure = (
                SparsityConfigMetadata.infer_sparsity_structure(model))

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


@torch.inference_mode()
def convert_to_hf(
    job_config: JobConfig,
    input_dir: str,
    output_dir: str,
    model_name: str,
    model_flavor: str,
    hf_assets_path: str,
    export_dtype: str,
    save_compressed: bool = True,
    disable_sparse_compression: bool = False,
):
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    model_converters = build_model_converters(job_config, None)

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
        model_converters.convert(model)
    model_converters.post_initialization(model)
    model_converters.pre_step(model)
    model_converters.finalize(model)
    wrapped_model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = wrapped_model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # print(model)
    # print(model.layers['0'].feed_forward.w1.quantization_scheme)
    # print(model.layers['0'].feed_forward.w1.quantization_status)

    compressor = get_model_compressor(
        model,
        state_dict,
        model_converters,
        save_compressed=save_compressed,
        disable_sparse_compression=disable_sparse_compression,
    )
    if compressor is not None:
        hot_fix_for_tied_word_embeddings(model, compressor)
        state_dict = compressor.compress(model)

        # TODO: update layer name mapping in the "ignore" entry
        compressor.update_config(output_dir)

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {
            k: v.to(target_dtype) if v.is_floating_point() else v
            for k, v in hf_state_dict.items()
        }

    with patch_finfo():
        dcp.save(
            hf_state_dict,
            storage_writer=storage_writer,
        )


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
    parser = argparse.ArgumentParser(
        description="Convert DCP weights to HF format.")
    parser.add_argument("config_file",
                        type=Path,
                        help="Path to job config file.")
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
        help="Tasks to evaluate (default: ['wikitext'])",
        default=["wikitext"],
    )
    args = parser.parse_args()

    config_manager = ConfigManager()
    job_config: JobConfig = config_manager.parse_args(
        ["--job.config_file", args.config_file.as_posix()])

    dump_folder = Path(job_config.job.dump_folder)
    hf_assets_path = Path(job_config.model.hf_assets_path)
    input_dir = dump_folder / "checkpoint" / f"step-{job_config.training.steps}"
    output_dir = dump_folder / "hf"

    model_name = job_config.model.name
    model_flavor = job_config.model.flavor
    
    if not args.skip_export:
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ["*.json", "*.py"]:
            for extra_file in hf_assets_path.glob(pattern):
                shutil.copy(extra_file, output_dir)

        convert_to_hf(
            job_config,
            input_dir.as_posix(),
            output_dir.as_posix(),
            model_name,
            model_flavor,
            hf_assets_path.as_posix(),
            args.export_dtype,
            save_compressed=not args.save_uncompressed,
            disable_sparse_compression=args.disable_sparse_compression,
        )
        sharded_output_dir = output_dir / "sharded"
        shutil.rmtree(sharded_output_dir, ignore_errors=True)

    if args.tasks:
        eval_tasks(output_dir.as_posix(), args.tasks)
