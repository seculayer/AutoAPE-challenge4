import torch
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel

import configuration
from model.model_utils import freeze, reinit_topk, check_nan, nan_filtering


class DictionaryWiseModel(nn.Module):
    """
    Model class for pair-wise(Margin Ranking), dict-wise(Multiple Negative Ranking) pipeline with DeBERTa-V3-Large
    Apply Pooling & Fully Connected Layer for each unique cell in one notebook_id
    Args:
        cfg: configuration.CFG
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/368997
    """
    def __init__(self, cfg: configuration.CFG):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 1)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict, position_list: Tensor) -> Tensor:
        """
        Return tensor for each subsequent cells in one notebook_id
        Example:
            pred = Tensor[subsequent cell1's logit, subsequent cell2's logit, ...]
        """
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        pred = torch.tensor([], device=self.cfg.device)
        for i in range(self.cfg.batch_size):
            """ Apply Pooling & Fully Connected Layer for each unique cell in batch (one notebook_id) """
            for idx in range(len(position_list[i])):
                src, end = position_list[i][idx]
                embedding = self.pooling(feature[i, src:end + 1, :].unsqueeze(dim=0))  # maybe don't need mask
                logit = self.fc(embedding)
                pred = torch.cat([pred, logit], dim=0)
        return pred


class PairwiseModel(nn.Module):
    """
    Model class for pair-wise(Margin Ranking), dict-wise(Multiple Negative Ranking) pipeline with DeBERTa-V3-Large
    Apply Pooling & Fully Connected Layer for each unique cell in one notebook_id

    For this Modeling Strategy, we use subsequent tokens for calculating embedding last_hidden_state,
    So when we forward last_hidden_state, we don't need to use mask for padding tokens
    Thus, if you want to apply mean pooling, you can easily implement it by using torch.mean() function

    Args:
        cfg: configuration.CFG
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/368997
    """
    def __init__(self, cfg: configuration.CFG):
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.model.resize_token_embeddings(len(self.cfg.tokenizer))
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 1)
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict, position_list: Tensor) -> list[Tensor]:
        """
        Return list of logit for each subsequent cells in one notebook_id
        Example:
            pred = list[subsequent cell1's logit, subsequent cell2's logit, ...]
        """
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        print(feature.shape)
        pred = []
        for i in range(self.cfg.batch_size):
            for idx in range(len(position_list[i])):
                src, end = position_list[i][idx]
                # embedding = self.pooling(feature[i, src:end + 1, :].unsqueeze(dim=0))  # check right index
                subsequent = feature[i, src:end + 1, :].unsqueeze(dim=0)
                if check_nan(subsequent):
                    print('warning nan')
                    subsequent = nan_filtering(subsequent)
                embedding = torch.mean(subsequent, dim=1)  # check right index
                logit = self.fc(embedding)
                pred.append(logit)
        print(pred)
        return pred
