import ast
from scipy import stats
from torch.utils.data import Dataset
from torch import Tensor

import configuration
from dataset_class.data_preprocessing import *


class NERDataset(Dataset):
    """
    Custom Dataset Class for NER Task
    Args:
        cfg: configuration.CFG
        df: dataframe from .txt file
        is_train: if this param set False, return word_ids from self.df.entities
    """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame, is_train: bool = True) -> None:
        self.cfg = cfg
        self.df = df
        self.tokenizer = ner_tokenizing
        self.labels2ids = labels2ids()  # Function for Encoding Labels to ids
        self.ids2labels = ids2labels()  # Function for Decoding ids to Labels
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> tuple[list, [dict[Tensor, Tensor, Tensor], Tensor]]:
        """
        1) Tokenizing input text:
            - if you param 'return_offsets_mapping' == True, tokenizer doen't erase \n or \n\n
              but, I don't know this param also applying for DeBERTa Pretrained Tokenizer
        2) Create targets and mapping of tokens to split() words by tokenizer
            - Mapping Labels to split tokens
            - Iterate in reverse to label whitespace tokens until a Begin token is encountered
            - Tokenizer will split word into subsequent of character such as copied => copy, ##ed
            - So, we need to find having same parent token and then label BIO NER Tags
        3) Return dict:
            - Train: dict.keys = [inputs_id, attention_mask, token_type_ids, labels]
            - Validation/Test: dict.keys = [inputs_id, attention_mask, token_type_ids, word_ids]
        """
        ids = self.df.id[item]
        text = self.df.text[item]
        if self.is_train:
            word_labels = ast.literal_eval(self.df.entities[item])

        # 1) Tokenizing input text
        encoding = self.tokenizer(
            self.cfg,
            text,
        )
        word_ids = encoding.word_ids()
        split_word_ids = np.full(len(word_ids), -1)
        offset_to_wordidx = split_mapping(text)  # [1, sequence_length]
        offsets = encoding['offset_mapping']  # [(src, end), (src, end), ...]

        # 2) Find having same parent token and then label BIO NER Tags
        label_ids = []
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):
            if word_idx is None:
                """ for padding token """
                if self.is_train:
                    label_ids.append(-100)
            else:
                if offsets[token_idx] != (0, 0):
                    # Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx][0]:offsets[token_idx][1]]
                    split_index = stats.mode(split_idxs[split_idxs != -1]).mode[0] if len(
                        np.unique(split_idxs)) > 1 else split_idxs[0]
                    if split_index != -1:
                        if self.is_train:
                            label_ids.append(self.labels2ids[word_labels[split_index]])
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and self.ids2labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if self.is_train:
                                label_ids.append(label_ids[-1])
                        else:
                            if self.is_train:
                                label_ids.append(-100)
                else:
                    if self.is_train:
                        label_ids.append(-100)
        if not self.is_train:
            encoding['word_ids'] = torch.as_tensor(split_word_ids)
        else:
            encoding['labels'] = list(reversed(label_ids))
        for k, v in encoding.items():
            encoding[k] = torch.as_tensor(v)
        return ids, encoding

