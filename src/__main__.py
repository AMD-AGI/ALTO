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

# from lm_eval.utils import make_table
# from lm_eval import evaluator

def main(config):
    model = MODEL_REGISTRY[config.model.type](config)

    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    # eval_list = get_eval_list(model, config)
    # eval_model(model, None, eval_list, eval_pos='pretrain')

    # if config.get("eval_downstream") and "pretrain" in config.eval_downstream.eval_pos:
    #     lmeval_model = srcFakeQuantizedModel(model, batch_size=config.eval_downstream.batch_size)
    #     for task_name, num_fewshot in zip(config.eval_downstream.task_names, config.eval_downstream.num_fewshots):
    #         num_fewshot = None if num_fewshot == "None" else num_fewshot
    #         single_task_result = evaluator.simple_evaluate(
    #             model=lmeval_model,
    #             tasks=[task_name,],
    #             num_fewshot=num_fewshot
    #         )
    #         logger.info("\n\n" + make_table(single_task_result))

    blockwise_opts = []
    modalities, modality_configs = get_modality(config)

    for modality, modality_config in zip(modalities, modality_configs):
        model.set_modality(modality)
        if not config.get('calib', False):
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
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
            import pdb; pdb.set_trace()
            calib_data, padding_mask = dataset.get_calib_dataset()
            import pdb; pdb.set_trace()
            model.collect_first_block_input(calib_data, padding_mask)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            blockwise_opt = ALGO_REGISTRY[modality_config.method](
                model,
                modality_config,
                model.get_first_block_input(),
                model.get_padding_mask(),
                config,
            )
            blockwise_opt.run_block_loop()
            blockwise_opts.append(blockwise_opt)
            dist.barrier()

    # eval_model(model, blockwise_opts, eval_list, eval_pos='transformed')
    # if int(os.environ['RANK']) == 0:
    #     if 'save' in config and config.save.get('save_trans', False):
    #         blockwise_opt.save_model(save_trans_path)

    #     if 'save' in config and config.save.get('save_trtllm', False):
    #         blockwise_opt.save_model(save_trtllm_trans_path)
    #         from src.utils.export_trtllm import cvt_trtllm_engine

    #         cvt_trtllm_engine(
    #             save_trtllm_trans_path,
    #             save_trtllm_engine_path,
    #             config.save.get('trtllm_cfg'),
    #         )

    #     eval_model(model, blockwise_opts, eval_list, eval_pos='fake_quant')

    #     if config.get("eval_downstream") and "fake_quant" in config.eval_downstream.eval_pos:
    #         lmeval_model = srcFakeQuantizedModel(model, batch_size=config.eval_downstream.batch_size)
    #         for task_name, num_fewshot in zip(config.eval_downstream.task_names, config.eval_downstream.num_fewshots):
    #             num_fewshot = None if num_fewshot == "None" else num_fewshot
    #             single_task_result = evaluator.simple_evaluate(
    #                 model=lmeval_model,
    #                 tasks=[task_name,],
    #                 num_fewshot=num_fewshot
    #             )
    #             logger.info("\n\n" + make_table(single_task_result))


        if 'save' in config and config.save.get('save_fake', False):
            deploy_all_modality(blockwise_opts, 'fake_quant')
            blockwise_opt.save_model(save_fake_path)

        if 'save' in config:
            if (
                config.save.get('save_vllm', False)
                or config.save.get('save_sgl', False)
                or config.save.get('save_lightllm', False)
            ):
                for modality_config in modality_configs:
                    w, a = modality_config.weight, modality_config.get('act')

                    if isinstance(w.bit, str):
                        assert w.symmetric, 'Only symmetric quant is supported.'
                        assert w.bit in ['e4m3', 'e3m4'], 'Supported quant: w8a16.'
                        if a:
                            assert (
                                w.symmetric and a.symmetric
                            ), 'Only symmetric quant is supported.'
                            assert (
                                w.bit == a.bit
                                and w.bit in ['e4m3', 'e5m2']
                                and a.bit in ['e4m3', 'e5m2']
                            ), 'Only WA FP8 quant is supported'
                    else:
                        assert w.symmetric, 'Only symmetric quant is supported.'
                        assert w.bit in [4, 8], 'Supported quant: w4a16, w8a16, w8a8.'
                        if a:
                            assert a.symmetric, 'Only symmetric quant is supported.'
                            assert a.bit == 8, 'Supported quant: w4a16, w8a16, w8a8.'

                if config.save.get('save_vllm', False):
                    deploy_all_modality(blockwise_opts, 'vllm_quant')
                elif config.save.get('save_lightllm', False):
                    deploy_all_modality(blockwise_opts, 'lightllm_quant')
                elif config.save.get('save_sgl', False):
                    deploy_all_modality(blockwise_opts, 'sgl_quant')

                blockwise_opt.save_model(save_quant_path)
                update_vllm_quant_config(blockwise_opt.model, config, save_quant_path)

            elif config.save.get('save_autoawq', False):
                for modality_config in modality_configs:
                    assert (
                        modality_config.weight.bit in [4] and 'act' not in modality_config
                    ), 'AutoAWQ supports only 4-bit weight-only quantization.'
                    assert (
                        not modality_config.weight.symmetric
                    ), 'Only asymmetric quant is supported.'

                deploy_all_modality(blockwise_opts, 'autoawq_quant')
                blockwise_opt.save_model(save_quant_path)
                update_autoawq_quant_config(config, save_quant_path)

            elif config.save.get('save_mlcllm', False):
                for modality_config in modality_configs:
                    assert (
                        modality_config.weight.bit in [4] and 'act' not in modality_config
                    ), 'MlcLLM supports only 4-bit weight-only quantization.'
                    assert (
                        not modality_config.weight.symmetric
                    ), 'Only asymmetric quant is supported.'

                deploy_all_modality(blockwise_opts, 'mlcllm_quant')
                blockwise_opt.save_model(save_quant_path)
                update_autoawq_quant_config(config, save_quant_path)

            elif config.save.get('save_lightx2v', False):
                deploy_all_modality(blockwise_opts, 'lightx2v_quant')
                blockwise_opt.save_model(save_quant_path)

        if 'opencompass' in config:
            assert config.save.get('save_trans', False)
            cfg_path = config['opencompass']['cfg_path']
            output_path = config['opencompass']['output_path']
            eval_model_path = os.path.abspath(save_trans_path)
            opencompass_cmd = (
                f'opencompass {cfg_path} -w {output_path} '
                f'--src_cfg {args.config} '
                f'--src_eval_mode quant '
                f'--src_model_path {eval_model_path}'
            )
            logger.info(f'opencompass_cmd : {opencompass_cmd}')
            os.system(opencompass_cmd)
    dist.barrier()


if __name__ == '__main__':
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
        pass
        if 'save' in config:
            if config.save.get('save_trans', False):
                save_trans_path = os.path.join(config.save.save_path, 'transformed_model')
                mkdirs(save_trans_path)
            if config.save.get('save_trtllm', False):
                save_trtllm_trans_path = os.path.join(config.save.save_path, 'trtllm_transformed_model')
                mkdirs(save_trtllm_trans_path)
                save_trtllm_engine_path = os.path.join(config.save.save_path, 'trtllm_engine')
                mkdirs(save_trtllm_engine_path)
            if config.save.get('save_vllm', False):
                save_quant_path = os.path.join(config.save.save_path, 'vllm_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_lightllm', False):
                save_quant_path = os.path.join(config.save.save_path, 'lightllm_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_sgl', False):
                save_quant_path = os.path.join(config.save.save_path, 'sgl_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_autoawq', False):
                save_quant_path = os.path.join(config.save.save_path, 'autoawq_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_mlcllm', False):
                save_quant_path = os.path.join(config.save.save_path, 'mlcllm_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_lightx2v', False):
                save_quant_path = os.path.join(config.save.save_path, 'lightx2v_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_fake', False):
                save_fake_path = os.path.join(config.save.save_path, 'fake_quant_model')
                mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config)

    destroy_process_group()

    src_end_time = time.time()
    src_duration_time = src_end_time - src_start_time
    logger.info(f'src_duration_time: {src_duration_time} s')
    logger.info('--- src finished ---')