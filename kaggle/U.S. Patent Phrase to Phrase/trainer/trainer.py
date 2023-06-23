import ast
import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.metric as model_metric
import model.model as model_arch
from torch import Tensor, inference_mode
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from dataset_class.text_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *
from functools import reduce


class UPPPMTrainer:
    """ For Token Classification Training Class """
    def __init__(self, cfg, generator):
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/token_classification/Fold4_UPPPM_train_df.csv')
        self.tokenizer = self.cfg.tokenizer
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int):
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid, is_valid=True)
        """ need to import pandas """
        tmp_valid, valid_labels = valid.explode('scores').scores.to_list(), []
        for val_list in tmp_valid:
            for score in ast.literal_eval(val_list):
                valid_labels.append(float(score))
        valid_labels = np.array(valid_labels)

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train, valid_labels

    def model_setting(self, len_train: int):
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg, len(self.tokenizer))
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)
        swa_model = AveragedModel(model)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
        val_metrics = getattr(model_metric, self.cfg.metrics)()

        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )

        swa_scheduler = get_swa_scheduler(self.cfg, optimizer)
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )

        return model, swa_model, criterion, val_criterion, val_metrics, optimizer, lr_scheduler, swa_scheduler, awp

    # Step 3.1 Train & Validation Function
    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None,
                 swa_model=None, swa_start=None, swa_scheduler=None,):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        global_step, score_list = 0, []  # All Fold's average of mean F2-Score
        losses = AverageMeter()
        model.train()

        for step, (inputs, _, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # label to gpu
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds.view(-1, 1), labels.view(-1, 1))
                mask = (labels.view(-1, 1) != -1)
                loss = torch.masked_select(loss, mask).mean()  # reduction = mean
                losses.update(loss, batch_size)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()

                if epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                global_step += 1
                scheduler.step()
        train_loss = losses.avg.detach().cpu().numpy()
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, val_criterion, val_metrics, valid_labels):
        """ Validation Function """
        preds_list, valid_losses = [], AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, target_masks, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = val_criterion(preds.view(-1, 1), labels.view(-1, 1))
                mask = (labels.view(-1, 1) != -1)
                valid_loss = torch.masked_select(valid_loss, mask).mean()
                valid_losses.update(valid_loss, batch_size)

                y_preds = preds.sigmoid().to('cpu').numpy()
                anchorwise_preds = []
                for pred, target_mask, in zip(y_preds, target_masks):
                    prev_i = -1
                    targetwise_pred_scores = []
                    for i, (p, tm) in enumerate(zip(pred, target_mask)):
                        if tm != 0:
                            if i - 1 == prev_i:
                                targetwise_pred_scores[-1].append(p)
                            else:
                                targetwise_pred_scores.append([p])
                            prev_i = i
                    for targetwise_pred_score in targetwise_pred_scores:
                        anchorwise_preds.append(np.mean(targetwise_pred_score))
                preds_list.append(anchorwise_preds)
        epoch_score = val_metrics(valid_labels, np.array(reduce(lambda a, b: a + b, preds_list)))
        valid_loss = valid_losses.avg.detach().cpu().numpy()
        return valid_loss, epoch_score

    def swa_fn(self, loader_valid, swa_model, val_criterion, val_metrics, valid_labels):
        """ Validation Function by Stochastic Weight Averaging """
        swa_model.eval()
        swa_preds_list, swa_valid_losses = [], AverageMeter()

        with torch.no_grad():
            for step, (swa_inputs, target_masks, swa_labels) in enumerate(tqdm(loader_valid)):
                swa_inputs = collate(swa_inputs)

                for k, v in swa_inputs.items():
                    swa_inputs[k] = v.to(self.cfg.device)

                swa_labels = swa_labels.to(self.cfg.device)
                batch_size = swa_labels.size(0)
                swa_preds = swa_model(swa_inputs)
                swa_valid_loss = val_criterion(swa_preds.view(-1, 1), swa_labels.view(-1, 1))
                mask = (swa_labels.view(-1, 1) != -1)
                swa_valid_loss = torch.masked_select(swa_valid_loss, mask)
                swa_valid_loss = swa_valid_loss.mean()
                swa_valid_losses.update(swa_valid_loss, batch_size)

                swa_y_preds = swa_preds.sigmoid().to('cpu').numpy()

                anchorwise_preds = []
                for pred, target_mask, in zip(swa_y_preds, target_masks):
                    prev_i = -1
                    targetwise_pred_scores = []
                    for i, (p, tm) in enumerate(zip(pred, target_mask)):
                        if tm != 0:
                            if i - 1 == prev_i:
                                targetwise_pred_scores[-1].append(p)
                            else:
                                targetwise_pred_scores.append([p])
                            prev_i = i
                    for targetwise_pred_score in targetwise_pred_scores:
                        anchorwise_preds.append(np.mean(targetwise_pred_score))

                swa_preds_list.append(anchorwise_preds)
            swa_valid_score = val_metrics(valid_labels, np.array(reduce(lambda a, b: a + b, swa_preds_list)))
            del swa_preds_list, swa_y_preds, swa_labels, anchorwise_preds
            gc.collect()
            torch.cuda.empty_cache()

        swa_loss = swa_valid_losses.avg.detach().cpu().numpy()
        return swa_loss, swa_valid_score


class TestInput:
    """ Test Pipeline Function with Token & Sentence """
    @inference_mode()
    def inference_fn(self, test_loader, model, device):
        preds = []
        model.eval()
        model.to(device)

        for inputs, target_masks in test_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            y_preds = model(inputs)
            y_preds = y_preds.sigmoid().to('cpu').numpy()

            anchorwise_preds = []
            for pred, target_mask, in zip(y_preds, target_masks):
                prev_i = -1
                targetwise_pred_scores = []
                for i, (p, tm) in enumerate(zip(pred, target_mask)):
                    if tm != 0:
                        if i - 1 == prev_i:
                            targetwise_pred_scores[-1].append(p)
                        else:
                            targetwise_pred_scores.append([p])
                        prev_i = i
                for targetwise_pred_score in targetwise_pred_scores:
                    anchorwise_preds.append(np.mean(targetwise_pred_score))
            preds.append(anchorwise_preds)

        return preds

    @inference_mode()
    def sentence_inference_fn(self, test_loader, model, device, use_sigmoid=False):
        preds = []
        model.eval()
        model.to(device)
        tk0 = tqdm(test_loader, total=len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            y_preds = model(inputs)
            if use_sigmoid:
                preds.append(y_preds.sigmoid().to('cpu').numpy())
            else:
                preds.append(y_preds.to('cpu').numpy())
        predictions = np.concatenate(preds)
        return predictions


class OptunaTuner:
    """ For Hyper-Params Tuning Class by Optuna """
    def __init__(self, cfg, generator):
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/Base_Train/train_df.csv')
        self.tokenizer = self.cfg.tokenizer
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int):
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, self.tokenizer, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, self.tokenizer, valid)

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))
        model.to(self.cfg.device)
        swa_model = AveragedModel(model)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )

        swa_scheduler = get_swa_scheduler(self.cfg, optimizer)
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        awp = None
        if self.cfg.awp:
            awp = AWP(
                model,
                criterion,
                optimizer,
                self.cfg.awp,
                adv_lr=self.cfg.awp_lr,
                adv_eps=self.cfg.awp_eps
            )

        return model, swa_model, criterion, optimizer, lr_scheduler, swa_scheduler, awp, self.save_parameter

    # Step 3.1 Train & Validation Function
    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None,
                 swa_model=None, swa_start=None, swa_scheduler=None,):
        """ Training Function """
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        global_step, score_list = 0, []  # All Fold's average of mean F2-Score
        losses = AverageMeter()
        model.train()

        for step, (inputs, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # label to gpu
            batch_size = labels.size(0)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = model(inputs)
                loss = criterion(preds, labels)
                losses.update(loss, batch_size)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            if self.cfg.awp and epoch >= self.cfg.nth_awp_start_epoch:
                loss = awp.attack_backward(inputs, labels)
                scaler.scale(loss).backward()
                awp._restore()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()

                if epoch >= int(swa_start):
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                global_step += 1
                scheduler.step()
        train_loss = losses.avg.detach().cpu().numpy()
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    def valid_fn(self, loader_valid, model, criterion):
        """ Validation Function """
        valid_losses = AverageMeter()
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = model(inputs)
                valid_loss = criterion(preds, labels)
                valid_losses.update(valid_loss, batch_size)
        valid_loss = valid_losses.avg.detach().cpu().numpy()
        return valid_loss

    def swa_fn(self, loader_valid, swa_model, criterion):
        """ Validation Function by Stochastic Weight Averaging """
        swa_model.eval()
        swa_valid_losses = AverageMeter()

        with torch.no_grad():
            for step, (swa_inputs, swa_labels) in enumerate(tqdm(loader_valid)):
                swa_inputs = collate(swa_inputs)

                for k, v in swa_inputs.items():
                    swa_inputs[k] = v.to(self.cfg.device)

                swa_labels = swa_labels.to(self.cfg.device)
                batch_size = swa_labels.size(0)
                swa_preds = swa_model(swa_inputs)
                swa_valid_loss = criterion(swa_preds, swa_labels)
                swa_valid_losses.update(swa_valid_loss, batch_size)
        swa_loss = swa_valid_losses.avg.detach().cpu().numpy()
        return swa_loss