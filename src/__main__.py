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

from src.models import *
from src.optimization import *
from src.data import BaseDataset
from src.eval import calc_evaluate
from src.utils import seed_all, mkdirs, ALGO_REGISTRY, MODEL_REGISTRY


def main(config):
    model = MODEL_REGISTRY[config.model.type](config)

    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')
    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    calc_evaluate(model, config, 'pretrained')

    for optimization_type, optimization_config in config['optimization'].items():
        if not config.get('calib', False):
            blockwise_optimizer = ALGO_REGISTRY[config['optimization'][optimization_type]['method']](
                model,
                optimization_config,
                input=None,
            )
            blockwise_optimizer.optimize()
            dist.barrier()
        else:
            dataset = BaseDataset(
                model.get_tokenizer(), config.calib, model.batch_process
            )
            calib_data = dataset.get_calib_dataset()
            import pdb; pdb.set_trace()
            model.collect_first_block_input(calib_data)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_optimizer = ALGO_REGISTRY[config['optimization'][optimization_type]['method']](
                model,
                optimization_config,
                model.get_first_block_input(),
            )
            blockwise_optimizer.optimize()
            dist.barrier()

    calc_evaluate(model, config, 'transformed')

    dist.barrier()


if __name__ == '__main__':
    if int(os.environ['RANK']) != 0:
        logger.remove()

    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config yaml path, e.g., llama-wanda.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)
    seed_all(config.base.seed + int(os.environ['RANK']))

    if int(os.environ['RANK']) == 0:
        pass  # TODO: mkdirs for storing compressed models

    dist.barrier()

    src_start_time = time.time()
    main(config)
    src_end_time = time.time()
    src_duration_time = src_end_time - src_start_time
    logger.info(f'src_duration_time: {src_duration_time} s')
    logger.info('--- src finished ---')

    destroy_process_group()