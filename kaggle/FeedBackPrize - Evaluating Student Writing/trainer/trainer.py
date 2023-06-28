import dataset_class.dataclass as dataset_class
import model.loss as model_loss
import model.metric as model_metric
import model.metric_learning as metric_learning
import model.model as model_arch
from torch.utils.data import DataLoader

from dataset_class import data_preprocessing
from dataset_class.data_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *
from tqdm.auto import tqdm


class DeBERTaTrainer:
    """
    Trainer Class for NER Task Pipeline (Stage 1), with backbone model named "DeBERTa-V3-Large"
    This class have 4 function:
        1) make_batch: make some input object related to batch (dataloader, dataframe)
        2) model_setting: make some input object related to model (model, criterion, optimizer, scheduler)
        3) train_fn: implement train stage per epoch
        4) valid_fn: implement valid stage per epoch
    Args:
        cfg: configuration.CFG
        generator: torch.Generator
    """
    def __init__(self, cfg: configuration.CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = get_name(self.cfg)
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/final_converted_train_df.csv')
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame, pd.DataFrame]:
        """ Make Batch Dataset for main train loop """
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid, is_train=False)
        # my_collate = MiniBatchCollate

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train, valid

    def model_setting(self, len_train: int):
        """ set train & validation options for main train loop """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))

        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
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
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)
        return model, criterion, val_metrics, optimizer, lr_scheduler

    # Train Function
    def train_fn(self, loader_train, model, criterion, optimizer, lr_scheduler, val_metrics):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        train_accuracy, train_recall, train_precision = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, (ids, inputs) in enumerate(tqdm(loader_train)):  # Maybe need to append
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # input_ids, attention_mask, token_type_ids to GPU
            labels = inputs.labels.view(-1)  # labels to GPU
            batch_size = self.cfg.batch_size
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                pred = model(inputs).view(-1, 15)
                loss = criterion(pred, labels)  # loss = criterion(pred.view(-1, 15), labels.view(-1))
            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.update(loss.detach(), batch_size)

            if self.cfg.clipping_grad and ((step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            """ Calculate Train Metrics """
            flat_pred = torch.argmax(pred, dim=1)  # shape (batch_size * seq_len,)

            active_label = labels != -100  # shape (batch_size, seq_len)
            labels = torch.masked_select(
                labels,
                active_label
            )
            pred = torch.masked_select(
                flat_pred,
                active_label
            )
            t_accuracy, t_recall, t_precision = val_metrics(
                pred.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            train_accuracy.update(t_accuracy, batch_size)
            train_recall.update(t_recall, batch_size)
            train_precision.update(t_precision, batch_size)

            gc.collect()

        train_loss = losses.avg.detach().cpu().numpy()
        return train_loss, train_accuracy.avg, train_recall.avg, train_precision.avg

    # Validation Function
    def valid_fn(self, loader_valid, model) -> tuple[list, list]:
        """ Validation Functions """
        ids_to_labels = data_preprocessing.ids2labels()
        val_ids_list, val_pred_list, val_label_list = [], [], []
        model.eval()
        with torch.no_grad():
            for step, (ids, inputs) in enumerate(tqdm(loader_valid)):  # Maybe need to append
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)  # prompt to GPU

                val_ids_list += ids  # make list for calculating cross validation score
                val_pred = model(inputs)  # inference for cross validation

                flat_val_pred = torch.argmax(val_pred, dim=-1).detach().cpu().numpy()
                predictions = []
                for k, text_pred in enumerate(flat_val_pred):
                    token_pred = [ids_to_labels[i] for i in text_pred]
                    prediction = []
                    word_ids = inputs['word_ids'][k].detach().cpu().numpy()
                    previous_word_idx = -1
                    for idx, word_idx in enumerate(word_ids):
                        if word_idx == -1:
                            pass
                        elif word_idx != previous_word_idx:
                            prediction.append(token_pred[idx])
                            previous_word_idx = word_idx
                    predictions.append(prediction)
                val_pred_list.extend(predictions)
        gc.collect()
        return val_ids_list, val_pred_list


class LongformerTrainer:
    """
    Trainer Class for NER Task Pipeline (Stage 1), with backbone model named "Longformer"
    This class have 4 function:
        1) make_batch: make some input object related to batch (dataloader, dataframe)
        2) model_setting: make some input object related to model (model, criterion, optimizer, scheduler)
        3) train_fn: implement train stage per epoch
        4) valid_fn: implement valid stage per epoch
    Args:
        cfg: configuration.CFG
        generator: torch.Generator
    """
    def __init__(self, cfg: configuration.CFG, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.model_name = get_name(self.cfg)
        self.generator = generator
        self.df = load_data('./dataset_class/data_folder/final_converted_train_df.csv')
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, pd.DataFrame, pd.DataFrame]:
        """ Make Batch Dataset for main train loop """
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid, is_train=False)
        # my_collate = MiniBatchCollate

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            # collate_fn=my_collate,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train, valid

    def model_setting(self, len_train: int):
        """ set train & validation options for main train loop """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))

        model.to(self.cfg.device)

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
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
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)
        return model, criterion, val_metrics, optimizer, lr_scheduler

    # Train Function
    def train_fn(self, loader_train, model, criterion, optimizer, lr_scheduler, val_metrics):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        train_accuracy, train_recall, train_precision = AverageMeter(), AverageMeter(), AverageMeter()
        model.train()
        for step, (ids, inputs) in enumerate(tqdm(loader_train)):  # Maybe need to append
            optimizer.zero_grad()
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(self.cfg.device)  # input_ids, attention_mask, token_type_ids to GPU
            labels = inputs.labels.view(-1)  # labels to GPU
            batch_size = self.cfg.batch_size
            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                loss, logit = model(inputs)
            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps
            scaler.scale(loss).backward()
            losses.update(loss.detach(), batch_size)

            if self.cfg.clipping_grad and ((step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            """ Calculate Train Metrics """
            pred = logit.view(-1, 15)  # model.num_labels == 15
            flat_pred = torch.argmax(pred, dim=1)  # shape (batch_size * seq_len,)

            active_label = labels != -100  # shape (batch_size, seq_len)
            labels = torch.masked_select(
                labels,
                active_label
            )
            pred = torch.masked_select(
                flat_pred,
                active_label
            )
            t_accuracy, t_recall, t_precision = val_metrics(
                pred.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            train_accuracy.update(t_accuracy, batch_size)
            train_recall.update(t_recall, batch_size)
            train_precision.update(t_precision, batch_size)

            gc.collect()

        train_loss = losses.avg.detach().cpu().numpy()
        return train_loss, train_accuracy.avg, train_recall.avg, train_precision.avg

    # Validation Function
    def valid_fn(self, loader_valid, model) -> tuple[list, list]:
        """ Validation Functions """
        ids_to_labels = data_preprocessing.ids2labels()
        val_ids_list, val_pred_list, val_label_list = [], [], []
        model.eval()
        with torch.no_grad():
            for step, (ids, inputs) in enumerate(tqdm(loader_valid)):  # Maybe need to append
                inputs = collate(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(self.cfg.device)  # prompt to GPU

                val_ids_list += ids  # make list for calculating cross validation score
                _, val_logit = model(inputs)  # inference for cross validation

                flat_val_pred = torch.argmax(val_logit, dim=-1).detach().cpu().numpy()
                predictions = []
                for k, text_pred in enumerate(flat_val_pred):
                    token_pred = [ids_to_labels[i] for i in text_pred]
                    prediction = []
                    word_ids = inputs['word_ids'][k].detach().cpu().numpy()
                    previous_word_idx = -1
                    for idx, word_idx in enumerate(word_ids):
                        if word_idx == -1:
                            pass
                        elif word_idx != previous_word_idx:
                            prediction.append(token_pred[idx])
                            previous_word_idx = word_idx
                    predictions.append(prediction)
                val_pred_list.extend(predictions)
        gc.collect()
        return val_ids_list, val_pred_list

