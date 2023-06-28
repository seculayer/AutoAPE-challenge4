import random, ast
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from torch.utils.data import Dataset
from torch import Tensor

import configuration
from dataset_class.data_preprocessing import *


class DictionaryWiseDataset(Dataset):
    """ Dataset class For Dictionary-wise(Multiple Neaative Ranking Loss) Pipeline """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame, is_valid=False) -> None:
        self.cfg = cfg
        self.id_list = df.id.to_numpy()
        self.cell_id_list = df.cell_id.to_numpy()
        self.cell_type_list = df.cell_type.to_numpy()
        self.rank_list = df['rank'].to_numpy()
        self.source_list = df.source.to_numpy()
        self.tokenizer = tokenizing
        self.subsequent_tokenizing = subsequent_tokenizing
        self.subsequent_decode = subsequent_decode
        self.is_valid = is_valid

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        1) Apply data augment
            - shuffle both of them, markdown text & code text
        2) Apply dynamic padding
        3) Make Embedding Shape:
            - Data: [CLS]+[MD]+[markdown text]+[MD]+[markdown text]+[MD]+[SEP]+[CD]+[code text]+[CD]+[SEP]
            - Label: just forward pass rank value?
        4) Make each unique cell's position list (all_position):
            for calculating subsequence of prompt's embedding, ex) markdown source text1 => 0.xxx
        """
        cell_ids = np.array(ast.literal_eval(self.cell_id_list[item]))
        cell_types = np.array(ast.literal_eval(self.cell_type_list[item]))
        ranks = np.array(ast.literal_eval(self.rank_list[item]))
        sources = np.array(ast.literal_eval(self.source_list[item]))

        # 1) Augment Data for train stage: shuffle target value's position index
        if not self.is_valid:
            indices = list(range(len(ranks)))
            random.shuffle(indices)
            cell_ids = cell_ids[indices]
            cell_types = cell_types[indices]
            ranks = ranks[indices]
            sources = sources[indices]  # [source6, source1, .... , source88]

        # 2) Apply Dynamic Padding
        tmp_token_list = []
        for idx in range(len(ranks)):
            tmp_token_list.append(subsequent_tokenizing(self.cfg, cleaning_words(sources[idx])))

        adjust_inputs, _ = adjust_sequences(tmp_token_list, (self.cfg.max_len - len(ranks) + 5))
        for idx in range(len(adjust_inputs)):
            sources[idx] = self.subsequent_decode(self.cfg, adjust_inputs[idx])  # decode to prompt text & convert

        # 3) Make prompt for model
        md_prompt = self.cfg.tokenizer.cls_token + self.cfg.tokenizer.markdown_token
        cd_prompt = self.cfg.tokenizer.sep_token + self.cfg.tokenizer.code_token
        md_rank, cd_rank = [], []
        for idx in range(len(ranks)):
            if cell_types[idx] == 'markdown':
                md_prompt += sources[idx] + self.cfg.tokenizer.markdown_token
                md_rank.append(ranks[idx])
            elif cell_types[idx] == 'code':
                cd_prompt += sources[idx] + self.cfg.tokenizer.code_token
                cd_rank.append(ranks[idx])

        prompt = md_prompt + cd_prompt + self.cfg.tokenizer.sep_token
        prompt = self.tokenizer(self.cfg, prompt)  # need to update with dynamic padding, and then make target mask
        ranks = torch.tensor(md_rank + cd_rank, dtype=torch.float32)

        # 4) Make each unique cell's position list (all_position)
        md_position, cd_position = [], []
        md_count, cd_count, sep_count, src, end = 0, 0, 0, 0, 0
        for idx, input_id in enumerate(prompt['input_ids']):
            # make markdown token position list
            if idx == 1:
                md_count += 1
                src = idx + 1
                continue
            if input_id == self.cfg.tokenizer.markdown_token_id and prompt['input_ids'][idx+1] != self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                md_position.append([src, end])
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.markdown_token_id and prompt['input_ids'][idx+1] == self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                md_position.append([src, end])
                sep_count += 1
                continue
            # make code token position list
            if input_id == self.cfg.tokenizer.code_token_id and cd_count == 0:
                cd_count += 1
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.code_token_id and prompt['input_ids'][idx+1] != self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                cd_position.append([src, end])
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.code_token_id and prompt['input_ids'][idx+1] == self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                cd_position.append([src, end])
                break

        md_position = torch.as_tensor(md_position)  # for validation stage
        cd_position = torch.as_tensor(cd_position)
        all_position = torch.cat([md_position, cd_position], dim=0)
        return prompt, ranks, all_position


class PairwiseDataset(Dataset):
    """ Dataset class For Pairwise(Margin Ranking Loss) Pipeline """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame, is_valid=False) -> None:
        self.cfg = cfg
        self.id_list = df.id.to_numpy()
        self.cell_id_list = df.cell_id.to_numpy()
        self.cell_type_list = df.cell_type.to_numpy()
        self.rank_list = df['rank'].to_numpy()
        self.source_list = df.source.to_numpy()
        self.tokenizer = tokenizing
        self.subsequent_tokenizing = subsequent_tokenizing
        self.subsequent_decode = subsequent_decode
        self.is_valid = is_valid

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        1) Apply data augment
            - shuffle both of them, markdown text & code text
        2) Apply dynamic padding
        3) Make Embedding Shape:
            - Data: [CLS]+[MD]+[markdown text]+[MD]+[markdown text]+[MD]+[SEP]+[CD]+[code text]+[CD]+[SEP]
            - Label: just forward pass rank value?
        4) Make each unique cell's position list (all_position):
            for calculating subsequence of prompt's embedding, ex) markdown source text1 => 0.xxx
        5) Make Ranking list for Margin Ranking Loss by itertools.combinations
            - sample two different cell and then compare each ranking
            - if left element greater than right element
                - label value goes 1
            - else
                - label value goes -1
        """
        cell_ids = np.array(ast.literal_eval(self.cell_id_list[item]))
        cell_types = np.array(ast.literal_eval(self.cell_type_list[item]))
        ranks = np.array(ast.literal_eval(self.rank_list[item]))
        sources = np.array(ast.literal_eval(self.source_list[item]))

        # 1) Augment Data for train stage: shuffle target value's position index
        if not self.is_valid:
            indices = list(range(len(ranks)))
            random.shuffle(indices)
            cell_ids = cell_ids[indices]
            cell_types = cell_types[indices]
            ranks = ranks[indices]
            sources = sources[indices]  # [source6, source1, .... , source88]

        # 2) Apply Dynamic Padding
        tmp_token_list = []
        for idx in range(len(ranks)):
            tmp_token_list.append(subsequent_tokenizing(self.cfg, cleaning_words(sources[idx])))
        adjust_inputs, _ = adjust_sequences(tmp_token_list, (self.cfg.max_len - len(ranks) + 5))
        for idx in range(len(adjust_inputs)):
            sources[idx] = self.subsequent_decode(self.cfg, adjust_inputs[idx])  # decode to prompt text & convert

        # 3) Make prompt for model
        md_prompt = self.cfg.tokenizer.cls_token + self.cfg.tokenizer.markdown_token
        cd_prompt = self.cfg.tokenizer.sep_token + self.cfg.tokenizer.code_token
        md_rank, cd_rank = [], []
        for idx in range(len(ranks)):
            if cell_types[idx] == 'markdown':
                md_prompt += sources[idx] + self.cfg.tokenizer.markdown_token
                md_rank.append(ranks[idx])
            elif cell_types[idx] == 'code':
                cd_prompt += sources[idx] + self.cfg.tokenizer.code_token
                cd_rank.append(ranks[idx])

        prompt = md_prompt + cd_prompt + self.cfg.tokenizer.sep_token
        prompt = self.tokenizer(self.cfg, prompt)  # need to update with dynamic padding, and then make target mask
        ranks = torch.tensor(md_rank + cd_rank)

        # 4) Make each unique cell's position list (all_position)
        md_position, cd_position = [], []
        md_count, cd_count, sep_count, src, end = 0, 0, 0, 0, 0
        for idx, input_id in enumerate(prompt['input_ids']):
            # make markdown token position list, need to check this algorthm is correct
            if idx == 1:
                md_count += 1
                src = idx + 1
                continue
            if input_id == self.cfg.tokenizer.markdown_token_id and prompt['input_ids'][idx+1] != self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                md_position.append([src, end])
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.markdown_token_id and prompt['input_ids'][idx+1] == self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                md_position.append([src, end])
                sep_count += 1
                continue
            # make code token position list
            if input_id == self.cfg.tokenizer.code_token_id and cd_count == 0:
                cd_count += 1
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.code_token_id and prompt['input_ids'][idx+1] != self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                cd_position.append([src, end])
                src = idx + 1
                continue
            elif input_id == self.cfg.tokenizer.code_token_id and prompt['input_ids'][idx+1] == self.cfg.tokenizer.sep_token_id:
                end = idx - 1
                cd_position.append([src, end])
                break

        md_position = torch.tensor(md_position)  # for validation stage
        cd_position = torch.tensor(cd_position)
        all_position = torch.cat([md_position, cd_position], dim=0)

        # 5) Make Ranking list for Margin Ranking Loss by itertools.combinations
        pair_rank_list = list(combinations(range(len(cell_ids)), 2))  # pairwise == 2
        pair_target_list = []
        for idx in range(len(pair_rank_list)):
            if ranks[pair_rank_list[idx][0]] < ranks[pair_rank_list[idx][1]]:
                pair_target_list.append(-1)
            else:
                pair_target_list.append(1)
        pair_rank_list = torch.tensor(pair_rank_list)
        pair_target_list = torch.tensor(pair_target_list)
        return prompt, ranks, all_position, pair_rank_list, pair_target_list

