import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

import configuration
from model.model_utils import freeze, reinit_topk


class DeBERTaModel(nn.Module):
    """
    Model class For NER Task Pipeline, in this class no pooling layer with backbone named "DeBERTa"
    This pipeline apply B.I.O Style, so the number of classes is 15 which is 7 unique classes original
    Each of 7 unique classes has sub 2 classes (B, I) => 14 classes
    And 1 class for O => 1 class
    14 + 1 = 15 classes
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: configuration.CFG) -> None:
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
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 15)  # BIO Style NER Task

    def feature(self, inputs_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs

    def forward(self, inputs) -> Tensor:
        """
        No Pooling Layer for word-level task
        Args:
            inputs: Dict type from AutoTokenizer
            => {input_ids, attention_mask, token_type_ids, offset_mapping, labels}
        """
        outputs = self.feature(
            inputs_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        logit = self.fc(outputs.last_hidden_state)
        return logit


class LongformerModel(nn.Module):
    """
    Model class For NER Task Pipeline, in this class no pooling layer with backbone named "Longformer"
    Longformer in huggingface is called by more specific named class, like as LongformerForTokenClassification
    AutoModelForTokenClassification class has already classifier layer, so we don't need to make fc layer
    Longformer with AutoModelForTokenClassification will be return blow those object:
        loss: Classification loss, torch.FloatTensor of shape (1,)
        logits: Classification scores (before SoftMax), torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels)
        hidden_states: List of hidden states at the output of each layer plus the initial embedding outputs
        attentions: List of attention weights after each layer
        global_attention: List of global attention weights after each layer
    This pipeline apply B.I.O Style, so the number of classes is 15 which is 7 unique classes original
    Each of 7 unique classes has sub 2 classes (B, I) => 14 classes
    And 1 class for O => 1 class
    14 + 1 = 15 classes
    Args:
        cfg: configuration.CFG
    Reference:
        https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/longformer#transformers.LongformerForTokenClassification
    """
    def __init__(self, cfg: configuration.CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.auto_cfg.num_labels = 15
        self.model = AutoModelForTokenClassification.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )

        if cfg.reinit:
            self._init_weights(self.model.classifier)
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

    def feature(self, inputs_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs

    def forward(self, inputs) -> tuple[Tensor, Tensor]:
        """
        No Pooling Layer for word-level task
        Args:
            inputs: Dict type from AutoTokenizer
            => {input_ids, attention_mask, token_type_ids, offset_mapping, labels}
        """
        outputs = self.feature(
            inputs_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        return outputs.loss, outputs.logits
