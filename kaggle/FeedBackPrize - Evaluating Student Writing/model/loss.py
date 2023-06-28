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


class FocalLoss(nn.Module):
    """
    FocalLoss
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()