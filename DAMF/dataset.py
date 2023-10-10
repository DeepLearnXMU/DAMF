# coding=utf-8
import random

import jsonlines
from torch.utils.data import Dataset
import numpy as np
from scipy.special import softmax

class DuSincDataset(Dataset):
    def __init__(self, file_path='../saved_data/data_dusinc/dev.json', skip_empty=True):
        with jsonlines.open(file_path, 'r') as reader:
            self.data = [line for line in reader if not skip_empty or line['query']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        return example['dialogue'], example['query']

class RAQGDataset(Dataset):
    def __init__(self, file_path, topk=10, bm25=True, threshold=20, gap = 5.):
        self.topk = topk
        self.bm25 = bm25
        self.gap = gap
        self.threshold = threshold
        print(f'Prepare data from {file_path}')
        with jsonlines.open(file_path, 'r') as reader:
            self.raw_data = [line for line in reader]
        self.data = []
        for example in self.raw_data:
            new_example = self.process_candidate(example)
            if new_example:
                self.data.append(new_example)
        print(f'Filter dataset {len(self.raw_data)} --> {len(self.data)}')

    def process_candidate(self, example):
        candidate_queries = example['candidate_queries']
        if candidate_queries is None:
            return None
        if self.bm25:
            if len(candidate_queries) < 2:
                return None
            sorted_queries = sorted(candidate_queries, key=lambda x: candidate_queries[x], reverse=True)[:self.topk]
            scores = np.array([candidate_queries[query] for query in sorted_queries])
            _max, _min = np.max(scores), np.min(scores)
            if _max - _min < self.gap or _max < self.threshold:
                return None
            rewards = (scores - _min) / (_max - _min) - min((np.mean(scores) - _min) / (_max - _min), 0.5)
        else:
            if '<EMPTY>' in candidate_queries:
                candidate_queries.pop('<EMPTY>') # seems useless
            if len(candidate_queries) < 2:
                return None
            sorted_queries = sorted(candidate_queries, key=lambda x: candidate_queries[x], reverse=True)[:self.topk]
            scores = np.array([candidate_queries[query] for query in sorted_queries])

            _max, _min = np.max(scores), np.min(scores)
            if _max - _min < self.gap or _max < self.threshold:
                return None
            rewards = (scores - _min) / (_max - _min) - min((np.mean(scores) - _min) / (_max - _min), 0.5)
        example['queries'] = sorted_queries
        example['rewards'] = rewards
        return example

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        query = example['queries']
        dialog = [example['input']] * len(query)
        score = example['rewards']
        return [dialog, query, score]