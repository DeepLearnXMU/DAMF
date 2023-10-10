# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from numpy.random import choice


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, n_context=0, split='train'):
        self.data = data
        self.n_context = n_context
        self.split = split

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'response' in example:
            if isinstance(example['response'], list):
                return random.choice(example['response'])
            elif isinstance(example['response'], str):
                return example['response']
        return None

    def __getitem__(self, index):
        example = self.data[index]
        dialog = example['context']
        target = self.get_target(example)

        if 'knowledge' in example and example['knowledge'] and self.n_context > 0:
            if self.n_context < len(example['knowledge']):
                if self.split == 'train':
                    indices, scores = zip(*[(i, x['bm25_score']) for i, x in enumerate(example['knowledge'])])
                    scores = np.array(scores)
                    score_sum = np.sum(scores)
                    if score_sum > 0.01:
                        scores /= score_sum
                        sampled_indices = choice(indices, size=self.n_context, replace=False, p=scores)
                    else:
                        sampled_indices = random.sample(indices, k=self.n_context)
                    contexts = [example['knowledge'][i] for i in sampled_indices]
                else:
                    contexts = example['knowledge'][:self.n_context]
            else:
                contexts = example['knowledge']
            passages = [""] + [c["text"] for c in contexts]
        else:
            passages = None

        return {
            'index': index,
            'dialog': dialog,
            'target': target,
            'passages': passages,
        }

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, tokenizer, special_tokens, text_maxlength=128, doc_maxlength=128, answer_maxlength=64,
                 n_docs=5, lang='zh'):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.doc_maxlength = doc_maxlength
        self.answer_maxlength = answer_maxlength
        self.n_docs = n_docs
        self.special_tokens = special_tokens
        self.lang = lang

    def __call__(self, batch):
        assert (batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                padding=True,
                return_tensors='pt',
                truncation=True if self.answer_maxlength > 0 else False, )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            def truncate(str, max_length, side='right',):
                if self.lang == 'zh':
                    if side == 'left':
                        return str[-max_length:]
                    elif side == 'right':
                        return str[:max_length]
                elif self.lang == 'en':
                    _str = str.split()
                    if side == 'left':
                        return ' '.join(_str[-max_length:])
                    elif side == 'right':
                        return ' '.join(_str[:max_length])
            dialog = truncate(example['dialog'], self.text_maxlength, 'left')
            if example['passages'] is None:
                return [self.special_tokens['cls'] + dialog]
            return [self.special_tokens['cls'] + dialog + self.special_tokens['sep'] +
                    truncate(doc, self.doc_maxlength, 'right') for doc in example['passages']]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages, self.tokenizer, 450)
        passage_num = min(passage_ids.shape[1], self.n_docs)
        return (index, target_ids, target_mask, passage_ids, passage_masks, passage_num)


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    data = open(data_path, 'r')  # end with .json but use jsonl
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
    data.close()
    return examples
