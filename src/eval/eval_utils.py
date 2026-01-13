import os
from loguru import logger

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.utils import make_table

from src.data.base_dataset import BaseDataset
from .ppl_eval import prefill_perplexity, decode_perplexity, prefill_perplexity_offload, decode_perplexity_offload
from .lmeval_utils import EvalModel


def calc_evaluate(model, config, eval_checkpoint='pretrained'):
    if int(os.environ['RANK']) == 0:
        if config.get("eval") and config.eval.get("eval_perplexity") and eval_checkpoint in config.eval.eval_checkpoints:
            for eval_type in config.eval.eval_perplexity.eval_type:
                for data_config in config.eval.eval_perplexity.data_configs:
                    testdata = BaseDataset(model.get_tokenizer(), data_config, model.batch_process)
                    testenc = testdata.get_input_ids()
                    if eval_type == 'prefill':
                        if config.eval.eval_perplexity.get('offload', False):
                            res = prefill_perplexity_offload(
                                model, 
                                testenc, 
                                data_config.processing.seq_len, 
                                data_config.processing.bs,
                                'cuda'
                            )
                        else:
                            res = prefill_perplexity(
                                model, 
                                testenc, 
                                data_config.processing.seq_len, 
                                data_config.processing.bs,
                                'cuda'
                            )
                        logger.info(f'EVAL: Prefill Perplexity on {data_config.dataset.name} is {res}')
                    elif eval_type == 'decode':
                        if config.eval.eval_perplexity.get('offload', False):
                            res = decode_perplexity_offload(
                                model, 
                                testenc, 
                                data_config.processing.seq_len, 
                                data_config.processing.bs, 
                                data_config.processing.num_samples, 
                                data_config.processing.max_num_eval_tokens
                            )
                        else:
                            res = decode_perplexity(
                                model, 
                                testenc, 
                                data_config.processing.seq_len, 
                                data_config.processing.bs, 
                                data_config.processing.num_samples, 
                                data_config.processing.max_num_eval_tokens
                            )
                        logger.info(f'EVAL: Decode Perplexity on {data_config.dataset.name} is {res}')
                    else:
                        raise NotImplementedError("Perplexity evaluation for other types is not supported.")

        if config.get("eval") and config.eval.get("eval_downstream") and eval_checkpoint in config.eval.eval_checkpoints:
            lmeval_model = EvalModel(model, batch_size=config.eval.eval_downstream.batch_size)
            for task_name, num_fewshot in zip(config.eval.eval_downstream.task_names, config.eval.eval_downstream.num_fewshots):
                num_fewshot = None if num_fewshot == "None" else num_fewshot
                single_task_result = evaluator.simple_evaluate(
                    model=lmeval_model,
                    tasks=[task_name,],
                    num_fewshot=num_fewshot
                )
                logger.info("\n\n" + make_table(single_task_result))