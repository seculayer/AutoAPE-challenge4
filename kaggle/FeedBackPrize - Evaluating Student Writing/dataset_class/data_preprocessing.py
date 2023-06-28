import gc, os
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from torch import Tensor
from collections import Counter
from bisect import bisect_left
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm


def ner_tokenizing(cfg: configuration.CFG, text: str):
    """
    Preprocess text for NER Pipeline
    if you want to set param 'return_offsets_mapping' == True, you must use FastTokenizer
    you must use PretrainedTokenizer which is supported FastTokenizer
    Converting text to torch.Tensor will be done in Custom Dataset Class
    Params:
        return_offsets_mapping:
            - bool, defaults to False
            - Whether or not to return (char_start, char_end) for each token.
            => useful for NER Task
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    """
    inputs = cfg.tokenizer(
        text,
        return_offsets_mapping=True,  # only available for FastTokenizer by Rust, not erase /n, /n/n
        max_length=cfg.max_len,
        padding='max_length',
        truncation=True,
        return_tensors=None,
        add_special_tokens=True,
    )
    return inputs


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Tokenize input sentence to longer sequence than common tokenizing
    Append padding strategy NOT Apply same max length, similar concept to dynamic padding
    Truncate longer sequence to match LLM max sequence
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/343714
        https://github.com/louis-she/ai4code/blob/master/tests/test_utils.py#L6

    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        max_length=128,
        padding=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,  # No need to special token to subsequent text sequence
    )
    return inputs['input_ids']


def adjust_sequences(sequences: list, max_len: int):
    """
    Similar to dynamic padding concept
    Append slicing index from original, because original source code is implemented weired
    So it generates some problem for applying very longer sequence
    Add -1 value to slicing index, so we can get result what we want
    Args:
        sequences: list of each cell's token sequence in one unique notebook id, must pass tokenized sequence input_ids
        => sequences = [[1,2,3,4,5,6], [1,2,3,4,5,6], ... , [1,2,3,4,5]]
        max_len: max length of sequence into LLM Embedding Layer, default is 2048 for DeBERTa-V3-Large
    Reference:
         https://github.com/louis-she/ai4code/blob/master/ai4code/utils.py#L70
    """
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l-1] for i, l in enumerate(length_of_seqs)]

    return sequences, length_of_seqs


def subsequent_decode(cfg: configuration.CFG, token_list: list) -> any:
    """
    Return decoded text from subsequent_tokenizing & adjust_sequences
    For making prompt text
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token_list: token list from subsequent_tokenizing & adjust_sequences
    """
    output = cfg.tokenizer.decode(token_list)
    return output


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc """
    return df.isnull().sum()


def check_inf(df: pd.DataFrame) -> bool:
    """ check if input dataframe has null type object...etc """
    checker = False
    if True in np.isinf(df.to_numpy()):
        checker = True
    return checker


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


def group_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = GroupKFold(
        n_splits=cfg.n_folds,
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, y=df['pct_rank'], groups=df['ancestor_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def stratified_groupkfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ Stratified Group KFold from sklearn.model_selection """
    fold = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df, df['target'], df['topics_ids'])):
        df.loc[vx, 'fold'] = int(num)  # Assign fold group number

    df['fold'] = df['fold'].astype(int)  # type casting for fold value
    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def get_n_grams(train: pd.DataFrame, n_grams: float, top_n: float = 10):
    """
    Return Top-10 n-grams from the each discourse type
    Source code from Reference URL, but I modified some part
    you can compare each discourse type's result, we can find really unique words for each discourse type
    Args:
        train: original train dataset from competition
        n_grams: set number of n-grams (window size)
        top_n: value of how many result do you want to see, sorted by descending counts value, default is 10

    [Reference]
    https://www.kaggle.com/code/erikbruin/nlp-on-student-writing-eda/notebook
    """
    df_words = pd.DataFrame()
    for dt in tqdm(train['discourse_type'].unique()):
        df = train.query('discourse_type == @dt')
        texts = df['discourse_text'].tolist()
        vec = CountVectorizer(
            lowercase = True,
            stop_words = 'english',
            ngram_range=(n_grams, n_grams)
        ).fit(texts)
        bag_of_words = vec.transform(texts)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        cvec_df = pd.DataFrame.from_records(words_freq, columns= ['words', 'counts']).sort_values(by="counts", ascending=False)
        cvec_df.insert(0, "Discourse_type", dt)
        cvec_df = cvec_df.iloc[:top_n,:]
        df_words = df_words.append(cvec_df)
    return df_words


def get_ner_labels(df: pd.DataFrame, text_df: pd.DataFrame) -> None:
    """
    Make NER labels feature for each token in sequence
    Args:
        df: original train dataset from train.csv
        text_df: text dataset from train.txt
    Reference:
        https://www.kaggle.com/code/cdeotte/pytorch-bigbird-ner-cv-0-615/notebook
    """
    all_entities = []
    for idx, i in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(idx, ', ', end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"] * total
        for j in df[df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]:
                entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    text_df['entities'] = all_entities
    text_df.to_csv('train_NER.csv',index=False)


def labels2ids():
    """
    Encoding labels to ids for neural network with BIO Styles
    labels2dict = {
    'O': 0, 'B-Lead': 1, 'I-Lead': 2, 'B-Position': 3, 'I-Position': 4, 'B-Claim': 5,
    'I-Claim': 6, 'B-Counterclaim': 7, 'I-Counterclaim': 8, 'B-Rebuttal': 9, 'I-Rebuttal': 10,
    'B-Evidence': 11, 'I-Evidence': 12, 'B-Concluding Statement': 13, 'I-Concluding Statement': 14
     }
    """
    output_labels = [
        'O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
        'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
        'I-Concluding Statement'
    ]
    labels_to_ids = {v: k for k, v in enumerate(output_labels)}
    return labels_to_ids


def ids2labels():
    """
    Decoding labels to ids for neural network with BIO Styles
    labels2dict = {
    'O': 0, 'B-Lead': 1, 'I-Lead': 2, 'B-Position': 3, 'I-Position': 4, 'B-Claim': 5,
    'I-Claim': 6, 'B-Counterclaim': 7, 'I-Counterclaim': 8, 'B-Rebuttal': 9, 'I-Rebuttal': 10,
    'B-Evidence': 11, 'I-Evidence': 12, 'B-Concluding Statement': 13, 'I-Concluding Statement': 14
     }

    """
    output_labels = [
        'O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
        'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
        'I-Concluding Statement'
    ]
    ids_to_labels = {k: v for k, v in enumerate(output_labels)}
    return ids_to_labels


def txt2df(data_path: str) -> pd.DataFrame:
    """
    Convert txt to dataframe for inference & submission
    Args:
        data_path: txt file path from competition host
    Reference:
        https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb/notebook
    """
    text_id, text = [], []
    for f in tqdm(list(os.listdir(data_path))):
        text_id.append(f.replace('.txt', ''))
        text.append(open(data_path + f, 'r').read())

    df = pd.DataFrame({'id': text_id, 'text': text})
    return df


def sequence_length(cfg: configuration.CFG, text_list: list) -> list:
    """ Get sequence length of all text data for checking statistics value """
    length_list = []
    for text in tqdm(text_list):
        tmp_text = ner_tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def split_mapping(unsplit):
    """ Return array which is mapping character index to index of word in list of split() words """
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit), -1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx


def sorted_quantile(array: list, q: float):
    """
    This is used to prevent re-sorting to compute quantile for every sequence.
    Args:
        array: list of element
        q: accumulate probability which you want to calculate spot
    Reference:
        https://stackoverflow.com/questions/60467081/linear-interpolation-in-numpy-quantile
        https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb/notebook
    """
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


def sequence_dataset(
    disc_type: str,
    valid_word_preds: np.ndarray,
    test_word_preds: np.ndarray = None,
    pred_indices: bool = None,
    submit: bool = False
        ):
    """
    Function for making sequence dataset for changing NER Task to Multi-Class Classification Task
    Args:
        disc_type: discourse type, for example 'Claim', 'Evidence' later turned into target classes
        valid_word_preds: valid word predictions from neural network which is trained NER Task
        test_word_preds: test word predictions from neural network which is trained NER Task
        pred_indices: indices of valid word predictions
        submit: if True, use test_word_preds instead of valid_word_preds
    Reference:
        https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb/notebook
    """
    word_preds = valid_word_preds if not submit else test_word_preds
    window = pred_indices if pred_indices else range(len(word_preds))
    X = np.empty((int(1e6), 13), dtype=np.float32)
    X_ind = 0
    y = []
    truePos = []
    wordRanges = []
    groups = []
    for text_i in tqdm(window):
        text_preds = np.array(word_preds[text_i])
        num_words = len(text_preds)
        disc_begin, disc_inside = disc_type_to_ids[disc_type]

        # The probability that a word corresponds to either a 'B'-egin or 'I'-nside token for a class
        prob_or = lambda word_preds: (1 - (1 - word_preds[:, disc_begin]) * (1 - word_preds[:, disc_inside]))

        if not submit:
            gt_idx = set()
            gt_arr = np.zeros(num_words, dtype=int)
            text_gt = valid.loc[valid.id == test_dataset.id.values[text_i]]
            disc_gt = text_gt.loc[text_gt.discourse_type == disc_type]

            # Represent the discourse instance locations in a hash set and an integer array for speed
            for row_i, row in enumerate(disc_gt.iterrows()):
                splt = row[1]['predictionstring'].split()
                start, end = int(splt[0]), int(splt[-1]) + 1
                gt_idx.add((start, end))
                gt_arr[start:end] = row_i + 1
            gt_lens = np.bincount(gt_arr)

        # Iterate over every sub-sequence in the text
        quants = np.linspace(0, 1, 7)  # number of target classes are 7
        prob_begins = np.copy(text_preds[:, disc_begin])
        min_begin = MIN_BEGIN_PROB[disc_type]
        for pred_start in range(num_words):
            prob_begin = prob_begins[pred_start]
            if prob_begin > min_begin:
                begin_or_inside = []
                for pred_end in range(pred_start + 1, min(num_words + 1, pred_start + MAX_SEQ_LEN[disc_type] + 1)):

                    new_prob = prob_or(text_preds[pred_end - 1:pred_end])
                    insert_i = bisect_left(begin_or_inside, new_prob)
                    begin_or_inside.insert(insert_i, new_prob[0])
                    features = [pred_end - pred_start, pred_start / float(num_words), pred_end / float(num_words)]
                    features.extend(list(sorted_quantile(begin_or_inside, quants)))
                    features.append(prob_or(text_preds[pred_start - 1:pred_start])[0] if pred_start > 0 else 0)
                    features.append(prob_or(text_preds[pred_end:pred_end + 1])[0] if pred_end < num_words else 0)
                    features.append(text_preds[pred_start, disc_begin])
                    exact_match = (pred_start, pred_end) in gt_idx if not submit else None
                    if not submit:
                        true_pos = False
                        for match_cand, count in Counter(gt_arr[pred_start:pred_end]).most_common(2):
                            if match_cand != 0 and count / float(pred_end - pred_start) >= .5 and float(count) / \
                                gt_lens[match_cand] >= .5: true_pos = True
                    else:
                        true_pos = None

                    if X_ind >= X.shape[0]:
                        new_X = np.empty((X.shape[0] * 2, 13), dtype=np.float32)
                        new_X[:X.shape[0]] = X
                        X = new_X
                    X[X_ind] = features
                    X_ind += 1

                    y.append(exact_match)
                    truePos.append(true_pos)
                    wordRanges.append((np.int16(pred_start), np.int16(pred_end)))
                    groups.append(np.int16(text_i))

    return SeqDataset(X[:X_ind], y, groups, wordRanges, truePos)