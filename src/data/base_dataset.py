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
    def __init__(self, tokenizer, calib_cfg, batch_process=None):
        # calib_cfg
        logger.info(f'calib_cfg : {calib_cfg}')
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        # calib dataset config
        self.calib_dataset_name  = calib_cfg['dataset']['name']
        self.calib_subset_name   = calib_cfg['dataset'].get('subset_name', None)
        self.calib_data_files    = calib_cfg['dataset'].get('data_files', None)
        self.calib_split         = calib_cfg['dataset'].get('split', 'train')
        self.calib_download      = calib_cfg['dataset'].get('download', True)
        self.calib_download_mode = calib_cfg['dataset'].get('download_mode', 'reuse_dataset_if_exists')
        self.calib_cache_dir     = calib_cfg['dataset'].get('cache_dir', None)
        self.calib_revision      = calib_cfg['dataset'].get('revision', None)
        self.calib_key           = calib_cfg['dataset'].get('hf_context_key', 'text')
        self.calib_dataset_path  = calib_cfg['dataset'].get('path', None)
        assert self.calib_dataset_path is not None or self.calib_download == True
        
        # calib processing config
        self.padding             = calib_cfg['processing'].get('padding', False)
        self.apply_chat_template = calib_cfg['processing'].get('apply_chat_template', False)
        self.n_samples           = calib_cfg['processing'].get('n_samples', None)
        self.calib_bs            = calib_cfg['processing']['bs']
        self.seq_len             = calib_cfg['processing'].get('seq_len', None)
        self.preproc             = calib_cfg['processing'].get('preproc', False)
        self.seed                = calib_cfg['processing']['seed']
        
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
            if self.calib_dataset_name == 'custom':
                self.calib_dataset = self.get_custom_dataset(self.calib_dataset_path)
            else:
                self.calib_dataset = load_from_disk(self.calib_dataset_path)

    def get_calib_model_inputs(self, samples):
        if not self.padding:
            if self.calib_dataset_name == 'custom_txt':
                txts = self.batch_process(
                    samples,
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                    return_inputs=False,
                )
            else:
                txts = self.calib_dataset
            preproc = PREPROC_REGISTRY[self.preproc]
            preproc_param_dict = {
                'calib_dataset': txts,
                'tokenizer': self.tokenizer,
                'n_samples': self.n_samples,
                'seq_len': self.seq_len,
            }
            if self.preproc == 'txt_general_preproc':
                preproc_param_dict['key'] = self.calib_key
            samples = preproc(**preproc_param_dict)
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
        else:
            assert (
                self.calib_dataset_name == 'custom_txt'
                or self.calib_dataset_name == 'custom_mm'
            )
            calib_model_inputs = self.get_batch_process(samples)
        return calib_model_inputs

    def get_batch_process(self, samples):
        calib_model_inputs = []
        if self.calib_bs == -1:
            calib_model_inputs.append(
                self.batch_process(
                    samples,
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                )
            )
        elif self.calib_bs == 1:
            calib_model_inputs = [
                self.batch_process(
                    [sample],
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                )
                for sample in samples
            ]
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_model_inputs.append(
                    self.batch_process(
                        batch,
                        calib_or_eval='calib',
                        apply_chat_template=self.apply_chat_template,
                    )
                )
        return calib_model_inputs

    def get_calib_dataset(self):
        samples = self.calib_dataset[
            int(os.environ['RANK'])::int(os.environ['WORLD_SIZE'])
        ]
        calib_model_inputs = self.get_calib_model_inputs(samples)
        logger.info(f'len(calib_model_inputs) : {len(calib_model_inputs)}')
        if self.padding:
            padding_mask = [
                calib_model_input['attention_mask']
                for calib_model_input in calib_model_inputs
            ]
        else:
            padding_mask = None
        return calib_model_inputs, padding_mask

    def get_custom_dataset(self, custom_dataset_path):
        audio_img_qa_json = os.path.join(custom_dataset_path, 'samples.json')
        fp = open(audio_img_qa_json)
        custom_data_samples = json.load(fp)
        for idx in range(len(custom_data_samples)):
            if 'audio' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['audio'], list):
                    for audio_idx in range(len(custom_data_samples[idx]['audio'])):
                        custom_data_samples[idx]['audio'][audio_idx] = os.path.join(
                            custom_dataset_path, custom_data_samples[idx]['audio'][audio_idx]
                        )
                else:
                    custom_data_samples[idx]['audio'] = os.path.join(
                        custom_dataset_path, custom_data_samples[idx]['audio']
                    )
            else:
                custom_data_samples[idx]['audio'] = None
            if 'image' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['image'], list):
                    for img_idx in range(len(custom_data_samples[idx]['image'])):
                        custom_data_samples[idx]['image'][img_idx] = os.path.join(
                            custom_dataset_path, custom_data_samples[idx]['image'][img_idx]
                        )
                else:
                    custom_data_samples[idx]['image'] = os.path.join(
                        custom_dataset_path, custom_data_samples[idx]['image']
                    )
            else:
                custom_data_samples[idx]['image'] = None
            if 'question' not in custom_data_samples[idx]:
                custom_data_samples[idx]['question'] = ''
            if 'answer' not in custom_data_samples[idx]:
                custom_data_samples[idx]['answer'] = ''
            if 'prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['prompt'] = ''
            if 'negative_prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['negative_prompt'] = ''
        return custom_data_samples