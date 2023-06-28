import torch, os, sys, random, json
import numpy as np


def check_device() -> bool:
    return torch.mps.is_available()


def check_library(checker: bool) -> tuple:
    """
    1) checker == True
        - current device is mps
    2) checker == False
        - current device is cuda with cudnn
    """
    if not checker:
        _is_built = torch.backends.cudnn.is_available()
        _is_enable = torch.backends.cudnn.enabledtorch.backends.cudnn.enabled
        version = torch.backends.cudnn.version()
        device = (_is_built, _is_enable, version)
        return device


def class2dict(cfg) -> dict:
    return dict((name, getattr(cfg, name)) for name in dir(cfg) if not name.startswith('__'))


def all_type_seed(cfg, checker: bool) -> None:
    # python & torch seed
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)  # python Seed
    random.seed(cfg.seed)  # random module Seed
    np.random.seed(cfg.seed)  # numpy module Seed
    torch.manual_seed(cfg.seed)  # Pytorch CPU Random Seed Maker

    # device == cuda
    if not checker:
        torch.cuda.manual_seed(cfg.seed)  # Pytorch GPU Random Seed Maker
        torch.cuda.manual_seed_all(cfg.seed)  # Pytorch Multi Core GPU Random Seed Maker
        # torch.cudnn seed
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # devide == mps
    else:
        torch.mps.manual_seed(cfg.seed)


def seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


