from .model_utils import *
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


# Multiple Negative Ranking Loss, source code from UKPLab
class MultipleNegativeRankingLoss(nn.Module):
    """
    Multiple Negative Ranking Loss (MNRL) for Dictionary Wise Pipeline, This class has one change point
    main concept is same as contrastive loss, but it can be useful when label data have only positive value
    if you set more batch size, you can get more negative pairs for each anchor & positive pair
    Change Point:
        In original code & paper, they set label from range(len()), This mean that not needed to use label feature
        But in our case, we need to use label feature, so we change label from range(len()) to give label feature
    Args:
        scale: output of similarity function is multiplied by this value
        similarity_fct: standard of distance metrics, default cosine similarity

    Example:
        model = SentenceTransformer('distil-bert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
        InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    Warnings:
        class param 'similarity_fct' is init with function 'cos_sim' in sentence_transformers.util
        util.cos_sim has not data checker like as torch.nn.functional.CosineSimilarity
        we add filter for this problem, but you must check your data before use this class
        if you use torch.cuda.amp, you must use eps == 1e-4

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
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.reduction = reduction
        self.cross_entropy_loss = CrossEntropyLoss(self.reduction)

    def forward(self, embeddings_a: Tensor, embeddings_b: Tensor, labels: Tensor = None) -> Tensor:
        """
        This Multiple Negative Ranking Loss (MNRL) is used for same embedding list,
        Args:
            embeddings_a: embeddings of shape for mini-batch
            embeddings_b: labels of mini-batch instance from competition dataset (rank), must be on same device with embedding
            labels:
        """
        similarity_scores = zero_filtering(self.similarity_fct(embeddings_a, embeddings_b)) * self.scale
        if check_nan(similarity_scores):
            """ Check NaN Value in similarity_scores """
            similarity_scores = nan_filtering(similarity_scores)
        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )
        # labels = embeddings_a.T.type(torch.long).to(similarity_scores.device)
        return self.cross_entropy_loss(similarity_scores, labels)
