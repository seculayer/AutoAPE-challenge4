import re
import numpy as np
import torch
import transformers
from torch import Tensor
from dataclasses import dataclass


def get_optimizer_grouped_parameters(model, layerwise_lr, layerwise_weight_decay, layerwise_lr_decay):
    """ Grouped Version: Layer-wise learning rate decay """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                     "weight_decay": 0.0,
                                     "lr": layerwise_lr,
                                     }, ]
    # initialize lrs for every layer
    layers = [model.model.embeddings] + list(model.model.encoder.layer)
    layers.reverse()
    lr = layerwise_lr
    for layer in layers:
        optimizer_grouped_parameters += [
            {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": layerwise_weight_decay,
             "lr": lr,
             },
            {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": lr,
             },]
        lr *= layerwise_lr_decay
    return optimizer_grouped_parameters


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def collate(inputs):
    """ Descending sort inputs by length of sequence """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def get_swa_scheduler(cfg, optimizer):
    """  SWA Scheduler """
    swa_scheduler = getattr(torch.optim.swa_utils, 'SWALR')(
        optimizer,
        swa_lr=cfg.swa_lr,
        anneal_epochs=cfg.anneal_epochs,
        anneal_strategy=cfg.anneal_strategy
    )
    return swa_scheduler


def get_scheduler(cfg, optimizer, len_train: int):
    """ Select Scheduler Function """
    scheduler_dict = {
        'cosine_annealing': 'get_cosine_with_hard_restarts_schedule_with_warmup',
        'cosine': 'get_cosine_schedule_with_warmup',
        'linear': 'get_linear_schedule_with_warmup'
    }
    lr_scheduler = getattr(transformers, scheduler_dict[cfg.scheduler])(
        optimizer,
        num_warmup_steps=int(len_train/cfg.batch_size * cfg.epochs/cfg.n_gradient_accumulation_steps) * cfg.warmup_ratio,
        num_training_steps=int(len_train/cfg.batch_size * cfg.epochs/cfg.n_gradient_accumulation_steps),
        num_cycles=cfg.num_cycles
    )
    return lr_scheduler


def get_name(cfg) -> str:
    """ get name of model """
    try:
        name = cfg.model.replace('/', '-')
    except ValueError:
        name = cfg.model
    return name


class AWP:
    """ Adversarial Weight Perturbation """
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        awp: bool,
        adv_param: str = "weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.awp = awp
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs: dict, label):
        with torch.cuda.amp.autocast(enabled=self.awp):
            self._save()
            self._attack_step()
            y_preds = self.model(inputs)
            adv_loss = self.criterion(
                y_preds.view(-1, 1), label.view(-1, 1))
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess(pseudo_label):
    """ for post processing to teacher model's prediction(pseudo label) """
    label_dict = torch.arange(1, 5.5, 0.5)
    pseudo_label.squeeze()
    for instance in pseudo_label:
        for idx in range(len(instance)):
            instance[idx] = label_dict[(torch.abs(label_dict - instance[idx]) == min(torch.abs(label_dict - instance[idx]))).nonzero(as_tuple=False)]
    return pseudo_label


class EarlyStopping(object):
    """
    Monitor a metric and stop training when it stops improving.

    Args:
        mode: 'min' for loss base val_score for loss, 'max' for metric base val_score
        patience: number of checks with no improvement, default = 3
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement. default = 0.0
        detect_anomaly: When set ``True``, stops training when the monitor becomes NaN or infinite, etc
                        default = True
    """
    def __init__(self, mode: str, patience: int = 3, min_delta: float = 0.0, detect_anomaly: bool = True):
        self.mode = mode
        self.early_stop = False
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.detect_anomaly = detect_anomaly
        self.val_score = -np.inf

        if self.mode == 'min':
            self.val_score = np.inf

    def detecting_anomaly(self) -> None:
        """ Detecting Trainer's Error and Stop train loop """
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        return

    def __call__(self, score: any) -> None:
        """ When call by Trainer Loop, Check Trainer need to early stopping """
        if self.mode == 'min':
            if self.val_score >= score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.mode == 'max':
            if score >= self.val_score:
                self.counter = 0
                self.val_score = score
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print('Early STOP')
