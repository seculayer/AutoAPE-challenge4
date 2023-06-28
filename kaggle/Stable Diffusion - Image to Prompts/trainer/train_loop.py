import gc
import wandb
from configuration import CFG
from trainer import *
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
            name=f'SD2_fold{fold}/' + cfg.model,
            config=class2dict(cfg),
            group=f'SD2_{cfg.image_pooling}/{cfg.model}',
            job_type='train',
            entity="qcqced"
        )
        early_stopping = EarlyStopping(mode=cfg.stop_mode, patience=2)
        early_stopping.detecting_anomaly()

        val_score_max = -np.inf
        train_input = getattr(trainer, cfg.name)(cfg, g)  # init object
        loader_train, loader_valid, train = train_input.make_batch(fold)
        model, style_model, text_encoder, criterion, val_metrics, optimizer, lr_scheduler = train_input.model_setting(len(train))

        for epoch in range(cfg.epochs):
            print(f'[{epoch + 1}/{cfg.epochs}] Train & Validation')
            train_loss = train_input.train_fn(
                loader_train, model, style_model, text_encoder, criterion, optimizer, lr_scheduler
            )
            valid_metric = train_input.valid_fn(
                loader_valid, model, style_model, text_encoder, val_metrics
            )
            wandb.log({
                '<epoch> Train Loss': train_loss,
                '<epoch> Valid Metric': valid_metric,
            })
            print(f'[{epoch + 1}/{cfg.epochs}] Train Loss: {np.round(train_loss, 4)}')
            print(f'[{epoch + 1}/{cfg.epochs}] Valid Metric: {np.round(valid_metric, 4)}')
            if val_score_max <= valid_metric:
                print(f'[Update] Valid Score : ({val_score_max:.4f} => {valid_metric:.4f}) Save Parameter')
                print(f'Best Score: {valid_metric}')
                torch.save(model.state_dict(),
                           f'{cfg.checkpoint_dir}fold{fold}_{cfg.image_pooling}_{get_name(cfg)}_state_dict.pth')
                val_score_max = valid_metric

            # Check if Trainer need to Early Stop
            early_stopping(valid_metric)
            if early_stopping.early_stop:
                break
            del train_loss, valid_metric
            gc.collect(), torch.cuda.empty_cache()

        del model, style_model, loader_train, loader_valid, train  # delete for next fold
        gc.collect(), torch.cuda.empty_cache()
        wandb.finish()
