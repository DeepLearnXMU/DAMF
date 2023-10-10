'''
This script aims to calculate BM25 scores for retrieved passages
'''
# coding=utf-8
from sys import argv
import jsonlines
from rank_bm25 import BM25Plus
from string import punctuation
from tqdm import tqdm
import numpy as np

_, data_path, passage_path, output_path, max_query_num = argv
max_query_num = int(max_query_num)

with jsonlines.open(passage_path, 'r') as reader:
    database = {line['query']: {'passages': line['passages'], 'tokens': [[item[0] for item in seq['words']] for seq in line['parse']]} for line in reader}

print('read in queries', len(database))

def sort_query_bm25(response, passages):
    sort_queries = []
    sort_passages = []
    for passage in passages:
        for query in passage['queries']:
            sort_queries.append(query)
            sort_passages.append(passage['tokens'])
    if len(sort_passages) == 0:
        return None
    bm25 = BM25Plus(sort_passages)
    scores = bm25.get_scores(response)
    query_scores = {}
    for q, s in zip(sort_queries, scores):
        if q in query_scores:
            query_scores[q] = max(query_scores[q], s)
        else:
            query_scores[q] = s
    return query_scores

with jsonlines.open(data_path, 'r') as reader:
    raw_data = [line for line in reader]

new_data = []
pbar = tqdm(total=len(raw_data))
for session in raw_data:
    dialogue, queries = [], []
    for utterance in session:
        text = utterance['text']
        parse = utterance['parse']
        if dialogue:
            queries += utterance['queries']
            tokens = [item[0] for item in parse['words']]
            retrieved_passages = {}
            for query in set(queries[-max_query_num:]):
                if query in database:
                    for passage, pparse in zip(database[query]['passages'], database[query]['tokens']):
                        if passage not in retrieved_passages:
                            retrieved_passages[passage] = {'text': passage, 'queries': [query], 'tokens': pparse}
                        else:
                            retrieved_passages[passage]['queries'].append(query)
            retrieved_passages = list(retrieved_passages.values())
            query_scores = sort_query_bm25(tokens, retrieved_passages)
            new_data.append({'input': '\n'.join(dialogue), 'candidate_queries': query_scores})
        for entity in parse['entities']:
            if entity[3][1] not in ['数量', '时间']:
                queries.append(entity[0])
        dialogue.append(text)
    pbar.update()
pbar.close()
with jsonlines.open(output_path, 'w') as writer:
    writer.write_all(new_data)
