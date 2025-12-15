import json
import os
from abc import ABCMeta

import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from PIL import Image
from torch.nn import functional as F

from .preproc_factory import PREPROC_REGISTRY


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, tokenizer, data_config, batch_process=None):
        # data_config
        logger.info(f'data_config : {data_config}')
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        
        # calib dataset config
        dataset_config     = data_config.get('dataset', {})
        self.dataset_name  = dataset_config['name']
        self.subset_name   = dataset_config.get('subset_name', None)
        self.data_files    = dataset_config.get('data_files', None)
        self.split         = dataset_config.get('split', 'test')
        self.download      = dataset_config.get('download', True)
        self.download_mode = dataset_config.get('download_mode', 'reuse_dataset_if_exists')
        self.cache_dir     = dataset_config.get('cache_dir', None)
        self.revision      = dataset_config.get('revision', None)
        self.key           = dataset_config.get('hf_context_key', 'text')
        self.dataset_path  = dataset_config.get('path', None)
        assert self.dataset_path is not None or self.download == True
        
        # calib processing config
        processing_config        = data_config.get('processing', {})
        self.apply_chat_template = processing_config.get('apply_chat_template', False)
        self.n_samples           = processing_config.get('n_samples', 1)
        self.calib_bs            = processing_config.get('bs', 1)
        self.seq_len             = processing_config.get('seq_len', 1024)
        self.preproc             = processing_config.get('preproc', 'calib_truncated_jointdoc_random')
        self.preproc_kwargs      = processing_config.get('preproc_kwargs', {})
        self.seed                = processing_config.get('seed', 0)
        
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.download:
            self.calib_dataset = load_dataset(
                path=self.dataset_name,
                name=self.subset_name,
                data_files=self.data_files,
                split=self.split,
                cache_dir=self.cache_dir,
                download_mode=self.download_mode,
                revision=self.revision
            )
        else:
            self.calib_dataset = load_from_disk(self.dataset_path)

    def get_input_ids(self):
        txts = self.calib_dataset
        preproc = PREPROC_REGISTRY[self.preproc]
        preproc_param_dict = {
            'calib_dataset': txts,
            'tokenizer': self.tokenizer,
            'n_samples': self.n_samples,
            'seq_len': self.seq_len,
            'hf_context_key': self.key
        }
        return preproc(**preproc_param_dict, **self.preproc_kwargs)
    
    def get_calib_model_inputs(self, samples):
        samples = self.get_input_ids()
        calib_model_inputs = []
        if self.calib_bs == -1:
            batch = torch.cat(samples, dim=0)
            calib_model_inputs.append({'input_ids': batch})
        elif self.calib_bs == 1:
            for i in range(len(samples)):
                calib_model_inputs.append({'input_ids': samples[i]})
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                batch = torch.cat(batch, dim=0)
                calib_model_inputs.append({'input_ids': batch})
        return calib_model_inputs

    def get_calib_dataset(self):
        samples = self.calib_dataset[
            int(os.environ['RANK'])::int(os.environ['WORLD_SIZE'])
        ]
        calib_model_inputs = self.get_calib_model_inputs(samples)
        logger.info(f'len(calib_model_inputs) : {len(calib_model_inputs)}')
        return calib_model_inputs
