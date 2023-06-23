import numpy as np
import torch
from torch import Tensor
from numpy import ndarray


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

