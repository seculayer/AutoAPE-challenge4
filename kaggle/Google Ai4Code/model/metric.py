import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy import ndarray
from bisect import bisect


def accuracy(output, target) -> float:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3) -> float:
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[pred[:, i] == target]).item()
    return correct / len(target)


def pearson_score(y_true, y_pred) -> float:
    x, y = y_pred, y_true
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    cov = np.sum(vx * vy)
    corr = cov / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-12)
    return corr


def recall(y_true, y_pred) -> float:
    """ recall = tp / (tp + fn) """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    score = tp / (tp + fn)
    return round(score.mean(), 4)


def precision(y_true, y_pred) -> float:
    """ precision = tp / (tp + fp) """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    score = tp / (tp + fp)
    return round(score.mean(), 4)


def f_beta(y_true, y_pred, beta: float = 2) -> float:
    """
    f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    if you want to emphasize precision, set beta < 1, options: 0.3, 0.6
    if you want to emphasize recall, set beta > 1, options: 1.5, 2

    [Reference]
    https://blog.naver.com/PostView.naver?blogId=wideeyed&logNo=221531998840&parentCategoryNo=&categoryNo=2&
    """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    f_precision = tp / (tp + fp)
    f_recall = tp / (tp + fn)
    score = (1 + beta ** 2) * f_precision * f_recall / (beta ** 2 * f_precision + f_recall)
    return round(score.mean(), 4)


def map_k(y_true: any, y_pred: any, k: int) -> Tensor:
    """
    mAP@K: mean of average precision@k
    Args:
        y_true: string or int, must be sorted by descending probability and type must be same with y_pred
                (batch_size, labels)
        y_pred: string or int, must be sorted by Ranking and type must be same with y_true
                (batch_size, predictions)
        k: top k
    """
    score = np.array([1 / (pred[:k].index(label) + 1) for label, pred in zip(y_true, y_pred)])
    return round(score.mean(), 4)


class PearsonScore(nn.Module):
    """ Pearson Correlation Coefficient Score class"""
    def __init__(self):
        super(PearsonScore, self).__init__()

    @staticmethod
    def forward(y_true, y_pred) -> float:
        x, y = y_pred, y_true
        vx = x - np.mean(x)
        vy = y - np.mean(y)
        cov = np.sum(vx * vy)
        corr = cov / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-12)
        return corr


class CosineSimilarity(nn.Module):
    """
    Returns cosine similarity between `x_1` and `x_2`, computed along `dim`
    Source code from pytorch.org
    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
              and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`
    """
    def __init__(self, dim: int = 0, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return F.cosine_similarity(x1, x2, self.dim, self.eps)  # need to add mean for batch


class KendallTau(nn.Module):
    """
    Variation Version of Kendall Tau Score class for Kaggle Competition <GoogleAi4Code>
    This class calculate instance-level metric, not mini-batch level
    Reference:
        https://www.kaggle.com/code/ryanholbrook/competition-metric-kendall-tau-correlation/notebook
    """
    def __init__(self):
        super(KendallTau, self).__init__()

    @staticmethod
    def count_inversions(predicted_rank: list) -> float:
        """
        number of pairs (i,j) which is needed to swap for correct order,
        such that i < j but predicted_rank[i] > predicted_rank[j]
        Args:
            predicted_rank: list of predicted rank from model
        """
        inversions = 0
        sorted_so_far = []
        for i, u in enumerate(predicted_rank):
            j = bisect(sorted_so_far, u)
            inversions += i - j
            sorted_so_far.insert(j, u)
        return inversions

    def forward(self, predictions: list, ground_truth: list) -> float:
        """
        n is the number of cell in one unique notebook id,
        total_2max is worst case of inversions, when all cells are in reverse order So all of them need to swap
        Thus, this metric means that efficient & accuracy of predicted order from model
        T = 1 - 4 * number of predicted_order's swap / number of worst_case's swap
        Args:
            ground_truth: list of ground truth, labels
            predictions: list of predictions from model
        """
        total_inversions, total_2max = 0, 0
        for gt, pred in zip(ground_truth, predictions):
            ranks = []
            for x in pred:
                if x >= 0:
                    ranks.append(gt.index(round(x)))
                else:
                    ranks.append(gt.index(0))
            total_inversions += self.count_inversions(ranks)
            n = len(gt)
            total_2max += n * (n - 1)
        return 1 - 4 * total_inversions / total_2max
