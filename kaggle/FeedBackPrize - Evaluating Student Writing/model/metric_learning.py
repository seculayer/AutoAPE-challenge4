import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Dict
from torch import Tensor
import sentence_transformers
from sentence_transformers.util import cos_sim
from .loss import CrossEntropyLoss


# Base Class for select Distance Metrics which is used in Metric Learning
class SelectDistances(nn.Module):
    """
    Select Distance Metrics
    Args:
        metrics: select distance metrics do you want,
                 - options: euclidean, manhattan, cosine
                 - PairwiseDistance: Manhattan(p=1), Euclidean(p=2)
                 - type: you must pass str type
    """
    def __init__(self, metrics: str) -> None:
        super().__init__()
        self.metrics = metrics

    def select_distance(self, x: Tensor, y: Tensor) -> Tensor:
        if self.metrics == 'cosine':
            distance_metric = 1 - F.cosine_similarity(x, y)  # Cosine Distance
        elif self.metrics == 'euclidean':
            distance_metric = F.pairwise_distance(x, y, p=2)
        else:
            distance_metric = F.pairwise_distance(x, y, p=1)  # L1, Manhattan
        return distance_metric


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss which is basic method of Metric Learning
    Closer distance between data points in intra-class, more longer distance between data points in inter-class

    Distance:
        Euclidean Distance: sqrt(sum((x1 - x2)**2))
        Cosine Distance: 1 - torch.nn.function.cos_sim(x1, x2)

    Examples:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)

    Args:
        margin: margin value meaning for Area of intra class(positive area), default 1.0
        metric: standard of distance metrics, default cosine distance

    References:
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
    """
    def __init__(self, metric: str = 'cosine', margin: int = 1.0) -> None:
        super(ContrastiveLoss, self).__init__()
        self.distance = SelectDistances(metric)  # Options: euclidean, manhattan, cosine
        self.margin = margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        anchor, instance = embeddings
        distance = self.distance(anchor, instance)
        contrastive_loss = 0.5 * (labels.float() * torch.pow(distance, 2) +
                                  (1 - labels.float()) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return contrastive_loss.mean()


# Metric Learning: Multiple Negative Ranking Loss for CLIP Model (Image to Text)
class CLIPMultipleNegativeRankingLoss(nn.Module):
    """
    Multiple Negative Ranking Loss for CLIP Model
    main concept is same as original one, but append suitable for other type of model (Not Sentence-Transformers)
    if you set more batch size, you can get more negative pairs for each anchor & positive pair
    Args:
        scale: output of similarity function is multiplied by this value => I don't know why this is needed
        similarity_fct: standard of distance metrics, default cosine similarity

    Example:
        model = SentenceTransformer('distil-bert-base-uncased')
        model = SentenceTransformer('distil-bert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
        InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

    Reference:
        https://arxiv.org/pdf/1705.00652.pdf
        https://www.sbert.net/docs/package_reference/losses.html
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr/notebook
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://www.youtube.com/watch?v=b_2v9Hpfnbw&ab_channel=NicholasBroad
    """
    def __init__(self, reduction: str, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.reduction = reduction
        self.cross_entropy_loss = CrossEntropyLoss(self.reduction)

    def forward(self, embeddings_a, embeddings_b):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row, same index means that they are ground truth
        This indicates that `a_i` and `b_j` have high similarity
        when `i==j` and low similarity when `i!=j`.
        Example a[i] should match with b[i]
        """
        similarity_scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale + self.eps

        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )
        return self.cross_entropy_loss(similarity_scores, labels)


# Multiple Negative Ranking Loss, source code from UKPLab
class MultipleNegativeRankingLoss(nn.Module):
    """
    Multiple Negative Ranking Loss (MNRL)
    main concept is same as contrastive loss, but it can be useful when label data have only positive value
    if you set more batch size, you can get more negative pairs for each anchor & positive pair
    Args:
        scale: output of similarity function is multiplied by this value
        similarity_fct: standard of distance metrics, default cosine similarity

    Example:
        model = SentenceTransformer('distil-bert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
        InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

    Reference:
        https://arxiv.org/pdf/1705.00652.pdf
        https://www.sbert.net/docs/package_reference/losses.html
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
        https://www.kaggle.com/code/nbroad/multiple-negatives-ranking-loss-lecr/notebook
        https://github.com/KevinMusgrave/pytorch-metric-learning
        https://www.youtube.com/watch?v=b_2v9Hpfnbw&ab_channel=NicholasBroad
    """
    def __init__(self, model: sentence_transformers, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        super().__init__()
        self.model = model
        self.scale = scale
        self.distance_metric = similarity_fct
        self.cross_entropy_loss = CrossEntropyLoss('mean')  # default setting: mean

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]]) -> Tensor:
        """
        need to append reps for CLIP Pipeline, not sentence-transformers
        In original code, embedding_b for concatenating reps start at index 1, but in this code, it start at index 0
        Because in starting from 1, it doesn't calculate cosine similarity between anchor and positive pair
        """
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embedding_a = reps[0]  # Embedding which is stacking N sentence(text, paragraph, document, etc.)
        embedding_b = torch.cat(reps[0:])  # Embedding which is stacking N sentence(text, paragraph, document, etc.)

        scores = self.similarity_fct(embedding_a, embedding_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long)
        return self.cross_entropy_loss(scores, labels)


# Arcface + CrossEntropy
class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace Loss, which is part of Metric Learning
    main concept of algorithm is same as other metric learning, but this method add margin to angular directly

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

    Reference:
        https://arxiv.org/abs/1801.07698
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    """

    def __init__(self, in_features: int, out_features: int, s: float, m: float, easy_margin: bool, ls_eps: float):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # size of concat embedding size
        self.out_features = out_features # num classes:
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
