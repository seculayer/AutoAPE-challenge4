import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.model as model_arch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from dataset_class.text_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *


class FBPTrainer:
    """ For Baseline Single Model Training Class """
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
        val_criterion = getattr(model_loss, self.cfg.val_loss_fn)(self.cfg.reduction)
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

        return model, swa_model, criterion, val_criterion, optimizer, lr_scheduler, swa_scheduler, awp

    # Step 3.1 Train & Validation Function
    def train_fn(self, loader_train, model, criterion, optimizer, scheduler, epoch, awp=None,
                 swa_model=None, swa_start=None, swa_scheduler=None,):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
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

    def valid_fn(self, loader_valid, model, val_criterion):
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
                valid_loss = val_criterion(preds, labels)
                valid_losses.update(valid_loss, batch_size)
        valid_loss = valid_losses.avg.detach().cpu().numpy()
        return valid_loss

    def swa_fn(self, loader_valid, swa_model, val_criterion):
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
                swa_valid_loss = val_criterion(swa_preds, swa_labels)
                swa_valid_losses.update(swa_valid_loss, batch_size)
        swa_loss = swa_valid_losses.avg.detach().cpu().numpy()
        return swa_loss


class MPLTrainer:
    """
    Trainer Class for Meta Pseudo Label Pipeline

    Args:
        cfg: config settings from configuration.py
        generator: generator for meta pseudo label

    Teacher Model
        train/validation: Supervised Data => use FBPDataset class, need to split for validation
        inference: Unsupervised Data for make pseudo labels
        loss: supervised loss + student validation loss

    Student Model
        train: Unsupervised Data with pseudo labels
        validation: Supervised Data => use FBPDataset class, no split for validation
        loss:

    1) Label Update with end of each training steps
    2) Pseudo Labeling for Regression Task
    """
    def __init__(self, cfg, generator):
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.supervised_df = load_data('./dataset_class/data_folder/Base_Train/train_df.csv')
        self.pseudo_df = load_data('./dataset_class/data_folder/MPL/train_df.csv')
        self.tokenizer = self.cfg.tokenizer
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self):
        """ For Supervised Learning & Meta Pseudo Label Learning """
        s_train, p_train = self.supervised_df.reset_index(drop=True), self.pseudo_df.reset_index(drop=True)

        s_train_dataset = dataset_class.FBPDataset(self.cfg, self.tokenizer, s_train)
        p_train_dataset = dataset_class.MPLDataset(self.cfg, self.tokenizer, p_train)

        # Teacher's DataLoader
        s_loader_train = DataLoader(
            s_train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Student's DataLoader
        p_loader_train = DataLoader(
            p_train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        p_loader_valid = DataLoader(
            s_train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return s_loader_train, s_train, p_loader_train, p_loader_valid, p_train

    def model_setting(self, len_t_train: int, len_s_train: int):
        """ For Meta Pseudo Label """
        t_model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        s_model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        t_model.load_state_dict(torch.load(
                self.cfg.checkpoint_dir + 'MPL_Teacher_microsoft-deberta-v3-large_state_dict.pth',
                map_location=self.cfg.device
                ),
                strict=False
        )
        if self.cfg.resume:
            s_model.load_state_dict(torch.load(
                self.cfg.checkpoint_dir + self.cfg.state_dict,
                map_location=self.cfg.device
                )
            )
        t_model.to(self.cfg.device)
        s_model.to(self.cfg.device)

        """ For Teacher Model """
        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        t_grouped_optimizer_params = get_optimizer_grouped_parameters(
            t_model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        t_optimizer = getattr(transformers, self.cfg.optimizer)(
            params=t_grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )
        """ For Student Model """
        s_grouped_optimizer_params = get_optimizer_grouped_parameters(
            s_model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        s_optimizer = getattr(transformers, self.cfg.optimizer)(
            params=s_grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )

        t_scheduler = get_scheduler(self.cfg, t_optimizer, len_t_train)
        s_scheduler = get_scheduler(self.cfg, s_optimizer, len_s_train)

        return t_model, s_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler, self.save_parameter

    def train_fn(self, t_model, s_model, criterion, t_optimizer, s_optimizer, t_scheduler, s_scheduler,
                  s_loader_train, p_loader_train, s_valid_loss):
        """
        Meta Pseudo Label Training Function

        Args:
            t: teacher model
            s: student model
            s_valid_loss: student model's validation loss by original label data

        Paper:
            teacher loss: supervised task loss + Unsupervised task loss + student model's validation loss
            student loss: meta pseudo label loss

            but, we use only supervised task loss + meta pseudo label loss for teacher loss
        """
        torch.autograd.set_detect_anomaly(True)
        t_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        s_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        t_losses, s_losses = AverageMeter(), AverageMeter()
        t_model.train(), s_model.train()
        """ Supervised Training: make supervised task loss """
        for step, (inputs, labels) in enumerate(tqdm(s_loader_train)):
            t_optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # train to gpu
            labels = labels.to(self.cfg.device)  # label to gpu
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = t_model(inputs)
                t_loss = criterion(preds, labels)
                new_t_loss = t_loss + s_valid_loss.squeeze()  # final teacher loss, later will add unsupervised loss
                t_losses.update(new_t_loss, batch_size) # teacher & student model's batch must be same
            if self.cfg.n_gradient_accumulation_steps > 1:
                new_t_loss = new_t_loss / self.cfg.n_gradient_accumulation_steps

            t_scaler.scale(new_t_loss).backward()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                t_scaler.unscale_(t_optimizer)
                torch.nn.utils.clip_grad_norm(
                    t_model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                t_scaler.step(t_optimizer)
                t_scaler.update()
                t_scheduler.step()
        t_train_loss = t_losses.avg.detach().cpu().numpy()  # supervised loss + student model's validation loss

        """ Pseudo Label Training: make student model's validation loss """
        for p_step, inputs in enumerate(tqdm(p_loader_train)):
            s_optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # un-supervised dataset to gpu
            # with torch.no_grad():
            pseudo_label = t_model(inputs)  # make pseudo label
            pseudo_label = postprocess(pseudo_label.detach().cpu().squeeze())  # postprocess
            batch_size = pseudo_label.size(0)
            pseudo_label = pseudo_label.to(self.cfg.device)  # pseudo label to gpu
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                preds = s_model(inputs)
                s_loss = criterion(preds, pseudo_label)
                print(s_loss)
                s_losses.update(s_loss, batch_size)
            if self.cfg.n_gradient_accumulation_steps > 1:
                s_loss = s_loss / self.cfg.n_gradient_accumulation_steps

            s_scaler.scale(s_loss).backward()

            if self.cfg.clipping_grad and (p_step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                s_scaler.unscale_(s_optimizer)
                torch.nn.utils.clip_grad_norm(
                    s_model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                s_scaler.step(s_optimizer)
                s_scaler.update()
                s_scheduler.step()

        s_train_loss = s_losses.avg.detach().cpu().numpy()
        return t_train_loss, s_train_loss, t_scheduler.get_lr()[0], s_scheduler.get_lr()[0]

    def valid_fn(self, p_loader_valid, s_model, criterion):
        """ Validation Function for student model's validation loss """
        s_valid_losses = AverageMeter()
        s_model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(tqdm(p_loader_valid)):
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)
                labels = labels.to(self.cfg.device)
                batch_size = labels.size(0)
                preds = s_model(inputs)
                s_valid_loss = criterion(preds, labels)
                s_valid_losses.update(s_valid_loss, batch_size)
        s_valid_losses = s_valid_losses.avg.detach().cpu().numpy()
        return s_valid_loss, s_valid_losses


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