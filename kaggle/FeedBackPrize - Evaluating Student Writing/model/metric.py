import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch import Tensor


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
    """
    Actual positives that the model predicted to be positive
    Math:
        recall = tp / (tp + fn)
    """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    score = tp / (tp + fn)
    return round(score.mean(), 4)


def precision(y_true, y_pred) -> float:
    """
    Actual positives among the model's positive predictions
    Math:
        precision = tp / (tp + fp)
    """
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    score = tp / (tp + fp)
    return round(score.mean(), 4)


def f_beta(y_true, y_pred, beta: float = 1) -> float:
    """
    F-beta score, in this competition, beta is 1 (micro f1 score)
    Element Explanation:
        tp: true positive
        fp: false positive
        tn: true negative
        fn: false negative
        if true ~, prediction == ground truth,
        if false ~, prediction != ground truth, ~ is prediction not ground truth value
    Math:
        f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        if you want to emphasize precision, set beta < 1, options: 0.3, 0.6
        if you want to emphasize recall, set beta > 1, options: 1.5, 2
    Reference:
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
    Mean average precision for top-k
    Args:
        y_true: string or int, must be sorted by descending probability and type must be same with y_pred
                (batch_size, labels)
        y_pred: string or int, must be sorted by Ranking and type must be same with y_true
                (batch_size, predictions)
        k: top k
    """
    score = np.array([1 / (pred[:k].index(label) + 1) for label, pred in zip(y_true, y_pred)])
    return round(score.mean(), 4)


class ConfusionMatrixMetrics(nn.Module):
    """
    This class is calculating metrics from confusion matrix, such as Accuracy, Precision, Recall by sklearn.metric
    Return:
        accuracy: (tp + tn) / (tp + tn + fp + fn)
        recall: tp / (tp + fn), average = 'micro'
        precision: tp / (tp + fp), average = 'micro'
    """
    def __init__(self):
        super(ConfusionMatrixMetrics, self).__init__()

    @staticmethod
    def forward(y_pred, y_true) -> tuple[float, float, float]:
        accuracy_metric = accuracy_score(y_true, y_pred)
        recall_metric = recall_score(y_true, y_pred, average='micro')
        precision_metric = precision_score(y_true, y_pred, average='micro')
        return accuracy_metric, recall_metric, precision_metric


class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    @staticmethod
    def forward(y_pred, y_true) -> float:
        """
        Actual positives that the model predicted to be positive
        Math:
            recall = tp / (tp + fn)
        """
        y_true = y_true.apply(lambda x: set(x.split()))
        y_pred = y_pred.apply(lambda x: set(x.split()))
        tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
        fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
        score = tp / (tp + fn)
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


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def calculate_f1(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """
    Function for scoring for competition
    Step 1:
        Make dataframe all ground truths and predictions for a given class are compared
    Step 2:
        If the overlap between the ground truth and prediction is >= 0.5 (Recall),
        and the overlap between the prediction and the ground truth >= 0.5 (Precision),
        In other words, prediction will be accepted 'True Positive',
        when Precision & Recall greater than 0.5
        the prediction is a match and considered a true positive.
        If multiple matches exist, the match with the highest pair of overlaps is taken.
        And then count number of Potential True Positive ids
    Step 3:
        Any unmatched ground truths are false negatives and any unmatched predictions are false positives.
        And then count number of Potential False Positives
    Step 4.
        Calculate Micro F1-Score for Cross Validation
    Reference:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']].reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']].reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score
