import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .model_utils import zero_filtering, nan_filtering, check_nan


# Mean Pooling
class SubSequenceGEMPooling(nn.Module):
    """
    Generalized Mean Pooling for Natural Language Processing
    This class version of GEMPooling for NLP, Transfer from Computer Vision Task Code

    Mean Pooling <= GEMPooling <= Max Pooling
    Because of doing exponent to each token embeddings, GEMPooling is like as weight to more activation token

    In original paper, they use p=3, but in this class, we use p=4 because torch doesn't support pow calculation
    for negative value tensor, only for non-negative value in odd number exponent

    In this Competition, Code Cell & Markdown Cell's token statistic distribution is fine different
    GEMPooling can detect mean embedding with unique embedding like as max pooling
    This can be helpful for dividing code cell and markdown cell
    Reference:
        https://paperswithcode.com/method/generalized-mean-pooling
    """
    def __init__(self, auto_cfg) -> None:
        super(SubSequenceGEMPooling, self).__init__()

    @staticmethod
    def forward(last_hidden_state, p: float = 1) -> Tensor:
        """
        last_hidden_state.size: [1, cell_sequence, hidden_size]
        1) Pow last_hidden_state with p and then take a averaging
        2) pow sum_embeddings with 1/p
        """
        p_embeddings = zero_filtering(torch.pow(last_hidden_state, p))
        # Check NaN value in Embedding after applying torch.pow
        if check_nan(p_embeddings):
            p_embeddings = nan_filtering(p_embeddings)

        sum_embeddings = torch.mean(p_embeddings, dim=1)
        gem_embeddings = zero_filtering(torch.pow(sum_embeddings, 1. / p))

        # Check NaN value in Embedding after applying torch.pow
        if check_nan(gem_embeddings):
            gem_embeddings = nan_filtering(gem_embeddings)
        return gem_embeddings


# Mean Pooling
class GEMPooling(nn.Module):
    """
    Generalized Mean Pooling for Natural Language Processing
    This class version of GEMPooling for NLP, Transfer from Computer Vision Task Code
    In other words, GEMPooling == Lp-Norm Pooling
    Mean Pooling <= GEMPooling <= Max Pooling
    Because of doing exponent to each token embeddings, GEMPooling is like as weight to more activation token

    In original paper, they use p=3, but in this class, we use p=4 because torch doesn't support pow calculation
    for negative value tensor, only for non-negative value in odd number exponent

    In this Competition, Code Cell & Markdown Cell's token statistic distribution is fine different
    GEMPooling can detect mean embedding with unique embedding like as max pooling
    This can be helpful for dividing code cell and markdown cell
    Reference:
        https://paperswithcode.com/method/generalized-mean-pooling
    """
    def __init__(self, auto_cfg) -> None:
        super(GEMPooling, self).__init__()
        self.eps = 1e-6

    def forward(self, last_hidden_state, attention_mask, p: int = 4) -> Tensor:
        """
        1) Expand Attention Mask from [batch_size, max_len] to [batch_size, max_len, hidden_size]
            1-1) For remove padding token, padding token's attention mask is 0
        2) Sum Embeddings along max_len axis so now we have [batch_size, hidden_size]
        3) Sum Mask along max_len axis, This is done so that we can ignore padding tokens
            3-1) torch.clamp: If sum_mask is 0, it will be 1e-9
        4) Average
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(torch.pow(last_hidden_state * input_mask_expanded, p), 1) + self.eps
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        tmp_embeddings = sum_embeddings / sum_mask
        gem_embeddings = torch.pow(tmp_embeddings, 1/p) + self.eps
        return gem_embeddings
