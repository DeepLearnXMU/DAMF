'''
This script aims to calculate BM25 scores for retrieved passages
'''
# coding=utf-8
from sys import argv
import jsonlines
from tqdm import tqdm

_, data_path, passage_path, output_path, max_query_num = argv
max_query_num = int(max_query_num)

with jsonlines.open(passage_path, 'r') as reader:
    database = {line['query']: {'passages': line['passages'], 'tokens': [[item[0] for item in seq['words']] for seq in line['parse']]} for line in reader}
print('read in queries', len(database))

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
                            retrieved_passages[passage] = {'text': passage, 'queries': [query]}
                        else:
                            retrieved_passages[passage]['queries'].append(query)
            retrieved_passages = list(retrieved_passages.values())
            new_data.append({'context': '\n'.join(dialogue), 'response': text,
                             'knowledge': retrieved_passages})
        for entity in parse['entities']:
            if entity[3][1] not in ['数量', '时间']:
                queries.append(entity[0])
        dialogue.append(text)
    pbar.update()
pbar.close()
with jsonlines.open(output_path, 'w') as writer:
    writer.write_all(new_data)
