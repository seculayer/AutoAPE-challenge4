import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


# WeightedLayerPooling: Use Intermediate Layer's Embedding
class WeightedLayerPooling(nn.Module):
    """
    For Weighted Layer Pooling Class
    In Original Paper, they use [CLS] token for classification task.
    But in common sense, Mean Pooling more good performance than CLS token Pooling
    So, we append Last part of this Pooling Method, Using CLS Token then Mean Pooling Embedding
    Args:
        auto_cfg: AutoConfig from model class member variable
        layer_start: start layer for pooling, default 4
        layer_weights: layer weights for pooling, default None
    """
    def __init__(self, auto_cfg, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = auto_cfg.num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (self.num_hidden_layers + 1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states, attention_mask) -> Tensor:
        all_layer_embedding = torch.stack(list(all_hidden_states), dim=0)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_average.size()).float()
        sum_embeddings = torch.sum(weighted_average * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # if lower than thres, replace value to threshold (parameter min)
        weighted_mean_embeddings = sum_embeddings / sum_mask
        return weighted_mean_embeddings


# Attention pooling
class AttentionPooling(nn.Module):
    """
    [Reference]
    <A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING>
    """
    def __init__(self, auto_cfg):
        super().__init__()
        self.attention = nn.Sequential(
           nn.Linear(auto_cfg.hidden_size, auto_cfg.hidden_size),
           nn.LayerNorm(auto_cfg.hidden_size),
           nn.GELU(),
           nn.Linear(auto_cfg.hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask) -> Tensor:
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


# Mean Pooling
class GEMPooling(nn.Module):
    """
    Generalized Mean Pooling for Natural Language Processing
    This class version of GEMPooling for NLP, Transfer from Computer Vision Task Code

    Mean Pooling <= GEMPooling <= Max Pooling
    Because of doing exponent to each token embeddings, GEMPooling is like as weight to more activation token

    In original paper, they use p=3, but in this class, we use p=4 because torch doesn't support pow calculation
    for negative value tensor, only for non-negative value in odd number exponent
    [Reference]
    https://paperswithcode.com/method/generalized-mean-pooling
    """
    def __init__(self, auto_cfg):
        super(GEMPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, attention_mask, p: int = 4) -> Tensor:
        """
        1) Expand Attention Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size]
        2) Sum Embeddings along max_len axis so now we have [batch_size, hidden_size]
        3) Sum Mask along max_len axis, This is done so that we can ignore padding tokens
        4) Average
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(
            torch.pow(last_hidden_state * input_mask_expanded, p), 1
        )
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        tmp_embeddings = sum_embeddings / sum_mask
        gem_embeddings = torch.pow(tmp_embeddings, 1/p)
        return gem_embeddings


# Mean Pooling
class MeanPooling(nn.Module):
    def __init__(self, auto_cfg):
        super(MeanPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, attention_mask) -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # if lower than threshold, replace value to threshold (parameter min)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# Max Pooling
class MaxPooling(nn.Module):
    def __init__(self, auto_cfg):
        super(MaxPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


# Min Pooling
class MinPooling(nn.Module):
    def __init__(self, auto_cfg):
        super(MinPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


# Convolution Pooling
class ConvPooling(nn.Module):
    """
    for filtering unwanted feature such as Toxicity Text, Negative Comment...etc
    kernel_size: similar as window size

    [Reference]
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """
    def __init__(self, feature_size: int, kernel_size: int, padding_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.convolution = nn.Sequential(
            nn.Conv1d(self.feature_size, 256, kernel_size=self.kernel_size, padding=self.padding_size),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=kernel_size, padding=padding_size),
        )

    def forward(self, last_hidden_state: Tensor) -> Tensor:
        embeddings = last_hidden_state.permute(0, 2, 1) # (batch_size, feature_size, seq_len)
        logit, _ = torch.max(self.convolution(embeddings), 2)
        return logit


# LSTM Pooling
class LSTMPooling(nn.Module):
    """
    [Reference]
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """
    def __int__(self, num_layers: int, hidden_size: int, hidden_dim_lstm):
        super().__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_dim_lstm,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states: list[Tensor]) -> Tensor:
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()\
                                    for layer_i in range(1, self.num_hidden_layers)], dim=1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
