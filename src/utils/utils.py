import os
import random
import shutil

import numpy as np
import torch
from loguru import logger


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(x, device, *, non_blocking: bool = False):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, torch.nn.Module):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device, non_blocking=non_blocking) for v in x)
    return x


def module_device(m):
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    b = next(m.buffers(), None)
    if b is not None:
        return b.device
    raise RuntimeError(f'{m.__class__.__name__} has no parameters or buffers; cannot infer device.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception(f'{path} existed before. Need check.')


def copy_files(source_dir, target_dir, substring):
    for filename in os.listdir(source_dir):
        if substring in filename:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.copy(source_file, target_file)
            logger.info(f'Copied {filename} to {target_dir}')


def print_important_package_version():
    from importlib.metadata import version
    logger.info(f"torch : {version('torch')}")
    logger.info(f"transformers : {version('transformers')}")
    logger.info(f"tokenizers : {version('tokenizers')}")
    logger.info(f"huggingface-hub : {version('huggingface-hub')}")
    logger.info(f"datasets : {version('datasets')}")