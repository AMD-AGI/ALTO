import os
from loguru import logger

from lm_eval import evaluator, utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.utils import make_table

from .ppl_eval import prefill_perplexity, decode_perplexity
from .lmeval_utils import EvalModel


def 
    if int(os.environ['RANK']) == 0:
        if config.get("eval_perplexity") and "pretrained" in config.eval_perplexity.eval_checkpoints:
            for eval_type in config.eval_perplexity.eval_type:
                testdata = BaseDataset(model.get_tokenizer(), config.eval_perplexity, model.batch_process)
                testenc = testdata.get_input_ids()
                res = prefill_perplexity(model, testenc, config.eval_perplexity.seq_len, config.eval_perplexity.bs)
                logger.info(f'EVAL: Perplexity on {config.eval_perplexity.dataset.name} is {res}')

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