import re, gc
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.feature_extraction.text import CountVectorizer
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


def stratified_groupkfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ sklearn Stratified Group KFold """
    fold = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df["score_class"] = df["score"].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df, df["score_class"], df["anchor"])):
        df.loc[vx, "fold"] = num
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


def filter_title(title: str) -> str:
    include_words= 0
    titles = normalize_words(title, unique=False)
    return ','.join([t for t in titles if t in include_words])


def add_special_token(cfg) -> None:
    """ Add [TAR] Token to pretrained tokenizer """
    tar_token = '[TAR]'
    special_tokens_dict = {'additional_special_tokens': [f'{tar_token}']}
    cfg.tokenizer.add_special_tokens(special_tokens_dict)
    tar_token_id = cfg.tokenizer(f'{tar_token}', add_special_tokens=False)['input_ids'][0]
    setattr(cfg.tokenizer, 'tar_token', f'{tar_token}')
    setattr(cfg.tokenizer, 'tar_token_id', tar_token_id)
    cfg.tokenizer.save_pretrained(f'{cfg.checkpoint_dir}/tokenizer/')


def cpc_preprocess(data_path, cfg) -> dict:
    """ Make Outside of Competition Data from CPC BigQuery in Kaggle Dataset """
    train_df = stratified_groupkfold(load_data(data_path), cfg)
    cpc_codes = pd.read_csv("./dataset/titles.csv", engine='python')

    norm_titles = normalize_words(cpc_codes['title'].to_numpy(), unique=False)  # 여기는 big query dataset을 정규화
    anchor_targets = train_df['target'].unique().tolist() + train_df['anchor'].unique().tolist()  # original train dataset
    norm_anchor_targets = normalize_words(anchor_targets)  # Original Train Dataset 정규화
    include_words = set(norm_titles) & norm_anchor_targets  # Anchor & Target 공통되는 단어
    tmp_cpc_codes = cpc_codes.copy()
    tmp_cpc_codes = tmp_cpc_codes[cpc_codes['code'].str.len() >= 4]

    tmp_cpc_codes['section_class'] = tmp_cpc_codes['code'].apply(lambda x: x[:3])
    title_group_df = tmp_cpc_codes.groupby('section_class', as_index=False)[['title']].agg(list)
    title_group_df = title_group_df[title_group_df['section_class'].str.len() == 3]
    title_group_df['title'] = title_group_df['title'].apply(lambda lst: ' '.join(lst))
    title_group_df['norm_title'] = title_group_df['title'].agg(filter_title)
    vectorizer = CountVectorizer()
    c_vect = vectorizer.fit_transform(title_group_df['norm_title'])
    r = np.argsort(c_vect.toarray(), axis=1)[:, ::-1][::, :400]
    vect_words = vectorizer.get_feature_names_out()
    t_words = np.vectorize(lambda v: vect_words[v])(r)
    norm_title = title_group_df['norm_title'].str.split(',').to_numpy().tolist()
    res = []
    for (n, t) in zip(norm_title, t_words):
        res.append(','.join(set(n) & set(t)))
    title_group_df['norm_title'] = res
    title_group_df['section'] = title_group_df.section_class.str[0:1]
    title_group_df['section_title'] = title_group_df['section'].map(
        cpc_codes.set_index('code')['title']).str.lower() + ';' + title_group_df['section_class'].map(
        cpc_codes.set_index('code')['title']).str.lower()
    title_group_df['context_text'] = title_group_df['section_title'] + ' [SEP] ' + title_group_df['norm_title']
    cpc_texts = dict(title_group_df[['section_class', 'context_text']].to_numpy().tolist())
    return cpc_texts


def token_preprocess(test_path: str, cpc_path: str):
    """ Preprocess for Token Classification """
    test = pd.read_csv(test_path)
    cpc_texts = torch.load(cpc_path)
    test['context_text'] = test['context'].map(cpc_texts)
    anchor_context_grouped_target = test.groupby(['anchor', 'context'])['target'].apply(list)
    anchor_context_grouped_id = test.groupby(['anchor', 'context'])['id'].apply(list)
    i = pd.DataFrame(anchor_context_grouped_id).reset_index()
    t = pd.DataFrame(anchor_context_grouped_target).reset_index()
    test = t.merge(i, on=['anchor', 'context'])
    test['context_text'] = test['context'].map(cpc_texts)
    test = test.rename(columns={'target': 'targets', 'id': 'ids'})
    test['n_ids'] = test['ids'].map(len)
    return test


def sentence_preprocess(test_path: str, sentence_cpc_path: str):
    """ Preprocess for Sentence Classification """
    sentence_test = pd.read_csv(test_path)
    cpc_texts = torch.load(sentence_cpc_path)
    sentence_test['context_text'] = sentence_test['context'].map(cpc_texts)
    return sentence_test
