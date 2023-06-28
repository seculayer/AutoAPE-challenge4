import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from torch import Tensor


class RMSELoss(nn.Module):
    def __init__(self, reduction, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps # If MSE == 0, We need eps

    def forward(self, yhat, y) -> Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    # num_scored => setting your number of metrics
    def __init__(self, reduction, num_scored=6):
        super().__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + (self.RMSE(yhat[:, i], y[:, i]) / self.num_scored)
        return score


# Weighted MCRMSE Loss => Apply different loss rate per target classes
class WeightMCRMSELoss(nn.Module):
    """
    Apply loss rate per target classes for using Meta Pseudo Labeling
    Weighted Loss can transfer original label data's distribution to pseudo label data
    [Reference]
    https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609
    """
    def __init__(self, reduction, num_scored=6):
        super().__init__()
        self.RMSE = RMSELoss(reduction=reduction)
        self.num_scored = num_scored
        self._loss_rate = torch.tensor([0.21, 0.16, 0.10, 0.16, 0.21, 0.16], dtype=torch.float32)

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score = score + torch.mul(self.RMSE(yhat[:, i], y[:, i]), self._loss_rate[i])
        return score


# Pearson Correlations Co-efficient Loss
class PearsonLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction
    @staticmethod
    def forward(y_pred, y_true) -> Tensor:
        x = y_pred.clone()
        y = y_true.clone()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr = torch.maximum(torch.minimum(corr, torch.tensor(1)), torch.tensor(-1))
        return torch.sub(torch.tensor(1), corr ** 2)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss

    [Reference]
    https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369793
    """
    def __init__(self, reduction, task_num=6) -> None:
        super(WeightedMSELoss, self).__init__()
        self.task_num = task_num
        self.smoothl1loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, y_pred, y_true, log_vars) -> float:
        loss = 0
        for i in range(self.task_num):
            precision = torch.exp(-log_vars[i])
            diff = self.smoothl1loss(y_pred[:, i], y_true[:, i])
            loss += torch.sum(precision * diff + log_vars[i], -1)
        loss = 0.5*loss
        return loss


class SmoothL1Loss(nn.Module):
    """ Smooth L1 Loss in Pytorch """
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.SmoothL1Loss(reduction=self.reduction)
        return criterion(y_pred, y_true)


# Cross-Entropy Loss
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.CrossEntropyLoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


# Binary Cross-Entropy Loss
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true) -> Tensor:
        criterion = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return criterion(y_pred, y_true)


class MarginRankingLoss(nn.Module):
    """
    This class to calculate loss values by comparing the rank of different two input's elements.
    Creates a criterion that measures the loss given inputs, Source code from pytorch official repository
    Forward Call Args:
        input1 (Tensor): Tensor of dimension 1D or 0D Tensor
        input2 (Tensor): Same dimension as input1, in this case, we put same tensor input1 & input2
        target (Tensor): Tensor of the same shape as input1 containing the rank value
    Reference:
        https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#margin_ranking_loss
        https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
    """
    def __init__(self, margin: float = 0.5, reduction: str = 'mean', size_average=True, reduce=True) -> None:
        super().__init__()
        self.__constants__ = ['margin', 'reduction']
        self.margin = margin
        self.reduction = reduction
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.margin_ranking_loss(
            input1,
            input2,
            target,
            margin=self.margin,
            reduction=self.reduction,
        )
