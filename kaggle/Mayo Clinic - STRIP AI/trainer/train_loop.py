import gc
import wandb
import torch
import numpy as np
from tqdm.auto import tqdm
from configuration import CFG
from trainer import *
from trainer.trainer_utils import get_name
from utils.helper import class2dict

g = torch.Generator()
g.manual_seed(CFG.seed)


def train_loop(cfg: any) -> None:
    """ Base Trainer Loop Function """
    fold_list = [i for i in range(cfg.n_folds)]
    for fold in tqdm(fold_list):
        print(f'============== {fold}th Fold Train & Validation ==============')
        wandb.init(
            project=cfg.name,
            name=f'[{cfg.model_arch}]' + f'fold{fold}/' + cfg.model,
            config=class2dict(cfg),
            group=f'{cfg.pooling}/max_length_{cfg.max_len}/{cfg.model}',
            job_type='train',
            entity="qcqced"
        )
        early_stopping = EarlyStopping(mode=cfg.stop_mode)
        early_stopping.detecting_anomaly()

        val_score_max, fold_swa_loss = np.inf, []
        train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
        loader_train, loader_valid, train = train_input.make_batch(fold)
        model, criterion, val_criterion, optimizer, lr_scheduler = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss, grad_norm, lr = train_input.train_fn(
                loader_train, model, criterion, optimizer, lr_scheduler, epoch
            )
            valid_loss, score = train_input.valid_fn(
                loader_valid, model, val_criterion
            )
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Valid Loss': valid_loss,
                '<epoch> Gradient Norm': grad_norm,
                '<epoch> lr': lr
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Loss: {np.round(valid_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Gradient Norm: {np.round(grad_norm, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] lr: {lr}')
            if val_score_max <= score:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {score:.4f}) Save Parameter')
                print(f'Best Score: {score}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}fold{fold}_{cfg.pooling}_{cfg.max_len}_{get_name(cfg)}_state_dict.pth')
                val_score_max = score

            # Check if Trainer need to Early Stop
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                break
            del train_loss, valid_loss, grad_norm, lr
            gc.collect(), torch.cuda.empty_cache()
        wandb.finish()
