import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group

from src.optimization.quantization import *
from src.optimization.sparsification import *
from src.data.base_dataset import BaseDataset
from src.models import *
from src.utils.utils import (deploy_all_modality, get_modality, seed_all, mkdirs)
from src.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY

from src.eval import prefill_perplexity, decode_perplexity, EvalModel
from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.utils import make_table

def main(config):
    model = MODEL_REGISTRY[config.model.type](config)

    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    if int(os.environ['RANK']) == 0:
        if config.get("eval_perplexity") and "pretrained" in config.eval_perplexity.eval_checkpoints:
            for eval_type in config.eval_perplexity.eval_type:
                for data_config in config.eval_perplexity.data_configs:
                    testdata = BaseDataset(model.get_tokenizer(), data_config, model.batch_process)
                    testenc = testdata.get_input_ids()
                    if eval_type == 'prefill':
                        res = prefill_perplexity(model, testenc, config.eval_perplexity.seq_len, config.eval_perplexity.bs)
                        logger.info(f'EVAL: Prefill Perplexity on {data_config.dataset.name} is {res}')
                    elif eval_type == 'decode':
                        res = decode_perplexity(model, testenc, config.eval_perplexity.seq_len, config.eval_perplexity.bs)
                        logger.info(f'EVAL: Decode Perplexity on {data_config.dataset.name} is {res}')
                    else:
                        raise NotImplementedError("Perplexity evaluation for other types is not supported.")

        if config.get("eval_downstream") and "pretrained" in config.eval_downstream.eval_checkpoints:
            lmeval_model = EvalModel(model, batch_size=config.eval_downstream.batch_size)
            for task_name, num_fewshot in zip(config.eval_downstream.task_names, config.eval_downstream.num_fewshots):
                num_fewshot = None if num_fewshot == "None" else num_fewshot
                single_task_result = evaluator.simple_evaluate(
                    model=lmeval_model,
                    tasks=[task_name,],
                    num_fewshot=num_fewshot
                )
                logger.info("\n\n" + make_table(single_task_result))

    blockwise_opts = []
    for optimization_type, optimization_config in config["optimization"].items():
        if not config.get('calib', False):
            blockwise_opt = ALGO_REGISTRY[config['optimization'][optimization_type]['method']](
                model,
                optimization_config,
                input=None,
                padding_mask=None,
                config=config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()
        else:
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data, padding_mask = dataset.get_calib_dataset()
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[config['optimization'][optimization_type]['method']](
                model,
                optimization_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )

            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()

    if int(os.environ['RANK']) == 0:
        if config.get("eval_perplexity") and "pretrained" in config.eval_perplexity.eval_checkpoints:
            for eval_type in config.eval_perplexity.eval_type:
                for data_config in config.eval_perplexity.data_configs:
                    testdata = BaseDataset(model.get_tokenizer(), data_config, model.batch_process)
                    testenc = testdata.get_input_ids()
                    if eval_type == 'prefill':
                        res = prefill_perplexity(model, testenc, config.eval_perplexity.seq_len, config.eval_perplexity.bs)
                        logger.info(f'EVAL: Prefill Perplexity on {data_config.dataset.name} is {res}')
                    elif eval_type == 'decode':
                        res = decode_perplexity(model, testenc, config.eval_perplexity.seq_len, config.eval_perplexity.bs)
                        logger.info(f'EVAL: Decode Perplexity on {data_config.dataset.name} is {res}')
                    else:
                        raise NotImplementedError("Perplexity evaluation for other types is not supported.")

        if config.get("eval_downstream") and "pretrained" in config.eval_downstream.eval_checkpoints:
            lmeval_model = EvalModel(model, batch_size=config.eval_downstream.batch_size)
            for task_name, num_fewshot in zip(config.eval_downstream.task_names, config.eval_downstream.num_fewshots):
                num_fewshot = None if num_fewshot == "None" else num_fewshot
                single_task_result = evaluator.simple_evaluate(
                    model=lmeval_model,
                    tasks=[task_name,],
                    num_fewshot=num_fewshot
                )
                logger.info("\n\n" + make_table(single_task_result))
    dist.barrier()


if __name__ == '__main__':
    rank = int(os.environ.get('RANK', '0'))
    logger.remove()
    if rank == 0:
        logger.add(sys.stdout, level='INFO')
    src_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/group/ossdphi_algo_scratch_13/guanchen/AMD-Model-Optimizer/configs/sparsification/llama-sparsegpt.yml")
    parser.add_argument('--task_id', type=str, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    if int(os.environ['RANK']) != 0:
        logger.remove()

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')

    logger.info(f'WORLD_SIZE : {int(os.environ["WORLD_SIZE"])}')

    seed_all(config.base.seed + int(os.environ['RANK']))

    # Ensure only the main process creates directories
    if int(os.environ['RANK']) == 0:
        pass# TODO: deal with save llm

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config)

    destroy_process_group()

    src_end_time = time.time()
    src_duration_time = src_end_time - src_start_time
    logger.info(f'src_duration_time: {src_duration_time} s')
    logger.info('--- src finished ---')