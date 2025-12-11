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
        self.calib_dataset_name  = data_config['dataset']['name']
        self.calib_subset_name   = data_config['dataset'].get('subset_name', None)
        self.calib_data_files    = data_config['dataset'].get('data_files', None)
        self.calib_split         = data_config['dataset'].get('split', 'test')
        self.calib_download      = data_config['dataset'].get('download', True)
        self.calib_download_mode = data_config['dataset'].get('download_mode', 'reuse_dataset_if_exists')
        self.calib_cache_dir     = data_config['dataset'].get('cache_dir', None)
        self.calib_revision      = data_config['dataset'].get('revision', None)
        self.calib_key           = data_config['dataset'].get('hf_context_key', 'text')
        self.calib_dataset_path  = data_config['dataset'].get('path', None)
        assert self.calib_dataset_path is not None or self.calib_download == True
        
        # calib processing config
        self.apply_chat_template = data_config['processing'].get('apply_chat_template', False)
        self.n_samples           = data_config['processing'].get('n_samples', None)
        self.calib_bs            = data_config['processing'].get('bs', 1)
        self.seq_len             = data_config['processing'].get('seq_len', None)
        self.preproc             = data_config['processing'].get('preproc', False)
        self.preproc_kwargs      = data_config['processing'].get('preproc_kwargs', {})
        self.seed                = data_config['processing'].get('seed', 0)
        
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.calib_download:
            self.calib_dataset = load_dataset(
                path=self.calib_dataset_name,
                name=self.calib_subset_name,
                data_files=self.calib_data_files,
                split=self.calib_split,
                cache_dir=self.calib_cache_dir,
                download_mode=self.calib_download_mode,
                revision=self.calib_revision
            )
        else:
            self.calib_dataset = load_from_disk(self.calib_dataset_path)

    def get_input_ids(self):
        txts = self.calib_dataset
        preproc = PREPROC_REGISTRY[self.preproc]
        preproc_param_dict = {
            'calib_dataset': txts,
            'tokenizer': self.tokenizer,
            'n_samples': self.n_samples,
            'seq_len': self.seq_len,
            'hf_context_key': self.calib_key
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
