import re, gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm.auto import tqdm


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


def text_preprocess(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    For FB3 Text Data
    FB3 Text data_folder has '\n\n', meaning that separate paragraphs are separated by '\n\n'
    DeBERTa does not handle '\n\n' well, so we need to change them into token '[PARAGRAPH]'
    """
    text_list = df['full_text'].values.tolist()
    text_list = [text.replace('\n\n', '[PARAGRAPH] ') for text in text_list]
    df['full_text'] = text_list
    df.reset_index(drop=True, inplace=True)
    df = mls_kfold(df, cfg)
    return df


def fb1_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    For FB1 Text Data
    Make FB1 Text Data for Meta Pseudo Labeling
    """
    unique_id, essay = 0, ''
    df = df[['id', 'discourse_text']]
    df.rename(columns={'id': 'text_id', 'discourse_text': 'full_text'}, inplace=True)

    tmp_df = pd.DataFrame(columns=['text_id', 'full_text'])
    unique_list = df['text_id'].unique()
    tmp_df['text_id'] = unique_list

    for idx in tqdm(range(len(df))):
        """
        Except Example for last idx
        """
        if df.iloc[idx, 0] == unique_list[unique_id]:
            if idx == len(df) - 1:
                essay += df.iloc[idx, 1]
                tmp_df.iloc[unique_id, 1] = essay
                break

            if df.iloc[idx + 1, 0] != unique_list[unique_id]:
                essay += df.iloc[idx, 1]
                tmp_df.iloc[unique_id, 1] = essay
                essay = ''
                unique_id += 1

            else:
                essay += df.iloc[idx, 1] + '[PARAGRAPH]'
    return tmp_df


def fb2_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    For FB2 Text Data
    Make FB2 Text Data for Meta Pseudo Labeling
    """
    unique_id, essay = 0, ''
    df = df[['discourse_id', 'discourse_text']]
    df.rename(columns={'discourse_id': 'text_id', 'discourse_text': 'full_text'}, inplace=True)

    tmp_df = pd.DataFrame(columns=['text_id', 'full_text'])
    unique_list = df['text_id'].unique()
    tmp_df['text_id'] = unique_list

    for idx in tqdm(range(len(df))):
        """
        Except Example for last idx
        """
        if df.iloc[idx, 0] == unique_list[unique_id]:
            if idx == len(df) - 1:
                essay += df.iloc[idx, 1]
                tmp_df.iloc[unique_id, 1] = essay
                break

            if df.iloc[idx + 1, 0] != unique_list[unique_id]:
                essay += df.iloc[idx, 1]
                tmp_df.iloc[unique_id, 1] = essay
                essay = ''
                unique_id += 1

            else:
                essay += df.iloc[idx, 1] + '[PARAGRAPH]'
    return tmp_df


def pseudo_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Make Pseudo DataFrame for Meta Pseudo Labeling
    Data from FB1 and FB2 are combined
    This DataSet is Un-Labled Data
    """
    pseudo_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    pseudo_df.reset_index(drop=True, inplace=True)
    pseudo_df = kfold(pseudo_df, cfg)
    return pseudo_df


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


