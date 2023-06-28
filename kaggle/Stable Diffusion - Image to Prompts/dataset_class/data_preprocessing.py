import re, gc
import pandas as pd
import numpy as np
import torch
import albumentations
import configuration as configuration
from torch import Tensor
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from albumentations.pytorch import ToTensorV2


def load_style_embedding():
    """ load style-embedding table (384x384x3 by convnext_base-384)"""
    return torch.as_tensor(torch.load('./dataset_class/embedding_table.pth'))


def clip_img_process(cfg: configuration.CFG, image: np.array) -> Tensor:
    """
    Preprocess Image For CLIP
    Arges:
        cfg: configuration.CFG, needed to load img_processor from Huggingface CLIPProcessor
        img: image convert to np.array type
    """
    if not type(image) == np.ndarray:
        raise TypeError('image must be np.ndarray type')

    return torch.as_tensor(cfg.img_processor(images=image)['pixel_values'])


def img_transform(image: np.ndarray) -> Tensor:
    """
    Preprocess image by albumentations
    Args:
        image: image convert to np.array type
    """
    transform = albumentations.Compose([
        # albumentations.Resize(384, 384),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()]
    )
    return transform(image=image)['image']


def tokenizing(cfg: configuration.CFG, text: str):
    """
    Preprocess text function for CLIP
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    """
    inputs = cfg.tokenizer(
        text,
        max_length=cfg.max_len,
        padding='max_length',
        truncation=True,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.as_tensor(v)
    return inputs


def kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ KFold """
    fold = KFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df)):
        df.loc[vx, "fold"] = int(num)
    return df


def mls_kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ Multilabel Stratified KFold """
    tmp_df = df.copy()
    y = pd.get_dummies(data=tmp_df.iloc[:, 2:8], columns=tmp_df.columns[2:8])
    fold = MultilabelStratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    for num, (tx, vx) in enumerate(fold.split(X=df, y=y)):
        df.loc[vx, "fold"] = int(num)
    del tmp_df
    gc.collect()
    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def create_word_normalizer():
    """
    Create a function that normalizes a word.
    """
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def normalize(word):
        w = word.lower()
        w = lemmatizer.lemmatize(w)
        w = ps.stem(w)
        return w
    return normalize


def __normalize_words(titles: list) -> list:
    """
    Normalize a list of words
    1) Remove stop words
    2) Apply Porter Stemmer, Lemmatizer
    """
    stop_words = set(stopwords.words('english'))
    normalizer = create_word_normalizer()
    titles = [normalizer(t) for t in titles if t not in stop_words]
    return titles


def normalize_words(words: np.ndarray, unique=True) -> list:
    """
    Normalize a list of words
    1) Apply __normalize_word function
    2) Apply Regular Expression to remove special characters
    """
    if type(words) is str:
        words = [words]
    sep_re = r'[\s\(\){}\[\];,\.]+'
    num_re = r'\d'
    words = re.split(sep_re, ' '.join(words).lower())
    words = [w for w in words if len(w) >= 3 and not re.match(num_re, w)]
    if unique:
        words = list(set(words))
        words = set(__normalize_words(words))
    else:
        words = __normalize_words(words)
    return words


def check_zero(x: np.ndarray) -> bool:
    """ Returns True if the NumPy array `x` contains any 0 values."""
    return np.any(x == 0)


