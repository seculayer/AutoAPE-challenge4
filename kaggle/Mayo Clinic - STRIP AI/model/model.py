import torch, timm
import torch.nn as nn
from torch import Tensor


class STRIPModel(nn.Module):
    """ Baseline Model Class for Classification """
    def __init__(self, cfg):
        super(STRIPModel, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            features_only=False,
        )
        self.fc = nn.Linear(self.model.config.hidden_size, 2)

        if cfg.reinit:
            self._init_weights(self.fc)

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
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

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        logit = self.fc(outputs)
        return logit
