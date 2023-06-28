import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from configuration import CFG


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """
    wrapper function for endless data_folder loader.
    """
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def sync_config(json_config: json) -> None:
    """ Pipeline Options """
    CFG.train, CFG.test = json_config.pipeline_setting.train, json_config.pipeline_setting.test
    CFG.checkpoint_dir = json_config.pipeline_setting.checkpoint_dir
    CFG.resume, CFG.state_dict = json_config.pipeline_setting.resume, json_config.pipeline_setting.state_dict
    CFG.name = json_config.pipeline_setting.name
    CFG.loop = json_config.pipeline_setting.loop
    CFG.dataset = json_config.pipeline_setting.dataset
    CFG.model_arch = json_config.pipeline_setting.model_arch
    CFG.model = json_config.pipeline_setting.model
    CFG.pooling = json_config.pipeline_setting.pooling

    """ Common Options """
    CFG.wandb = json_config.common_settings.wandb
    CFG.optuna = json_config.common_settings.optuna
    CFG.competition = json_config.common_settings.competition
    CFG.seed = json_config.common_settings.seed
    CFG.n_gpu = json_config.common_settings.n_gpu
    CFG.gpu_id = json_config.common_settings.gpu_id
    CFG.num_workers = json_config.common_settings.num_workers

    """ Data Options """
    CFG.n_folds = json_config.data_settings.n_folds
    CFG.max_len = json_config.data_settings.max_len
    CFG.epochs = json_config.data_settings.epochs
    CFG.batch_size = json_config.data_settings.batch_size

    """ Gradient Options """
    CFG.amp_scaler = json_config.gradient_settings.amp_scaler
    CFG.gradient_checkpoint = json_config.gradient_settings.gradient_checkpoint
    CFG.clipping_grad = json_config.gradient_settings.clipping_grad
    CFG.max_grad_norm = json_config.gradient_settings.max_grad_norm

    """ Loss Options """
    CFG.loss_fn = json_config.loss_options.loss_fn
    CFG.val_loss_fn = json_config.loss_options.val_loss_fn
    CFG.reduction = json_config.loss_options.reduction

    """ Metrics Options """
    CFG.metrics = json_config.metrics_options.metrics

    """ Optimizer Options """
    CFG.optimizer = json_config.optimizer_options.optimizer
    CFG.llrd = json_config.optimizer_options.llrd
    CFG.layerwise_lr = json_config.optimizer_options.layerwise_lr
    CFG.layerwise_lr_decay = json_config.optimizer_options.layerwise_lr_decay
    CFG.layerwise_weight_decay = json_config.optimizer_options.layerwise_weight_decay
    CFG.layerwise_adam_epsilon = json_config.optimizer_options.layerwise_adam_epsilon
    CFG.layerwise_use_bertadam = json_config.optimizer_options.layerwise_use_bertadam
    CFG.betas = json_config.optimizer_options.betas

    """ Scheduler Options """
    CFG.scheduler = json_config.scheduler_options.scheduler
    CFG.batch_scheduler = json_config.scheduler_options.batch_scheduler
    CFG.num_cycles = json_config.scheduler_options.num_cycles
    CFG.warmup_ratio = json_config.scheduler_options.warmup_ratio

    """ SWA Options """
    CFG.swa = json_config.swa_options.swa
    CFG.swa_lr = json_config.swa_options.swa_lr
    CFG.anneal_epochs = json_config.swa_options.anneal_epochs
    CFG.anneal_strategy = json_config.swa_options.anneal_strategy

    """ model_utils """
    CFG.stop_mode = json_config.model_utils.stop_mode
    CFG.reinit = json_config.model_utils.reinit
    CFG.num_freeze = json_config.model_utils.num_freeze
    CFG.num_reinit = json_config.model_utils.num_reinit
    CFG.awp = json_config.model_utils.awp
    CFG.nth_awp_start_epoch = json_config.model_utils.nth_awp_start_epoch
    CFG.awp_eps = json_config.model_utils.awp_eps
    CFG.awp_lr = json_config.model_utils.awp_lr
