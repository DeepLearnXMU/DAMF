'''
This script is used to tokenize webpages using texsmart (M)
'''

#encoding=utf-8
import sys
from sys import argv
_, texsmart_lib, texsmart_kb, data_path, output_path = argv
sys.path.append(texsmart_lib)
import string
import jsonlines
from tqdm import tqdm
from tencent_ai_texsmart import *

print('Creating and initializing the NLU engine...')
engine = NluEngine(texsmart_kb, 1)

def parse(text):
    output = engine.parse_text(text)
    results = {'words': [(item.str, item.offset, item.len, item.tag, item.freq) for item in output.words()],
               'phrases': [(item.str, item.offset, item.len, item.tag, item.freq) for item in output.phrases()],
               'entities': [(entity.str, entity.offset, entity.len,
                             (entity.type.name, entity.type.i18n, entity.type.flag, entity.type.path), entity.meaning) for
                            entity in output.entities()]}
    return results

def is_all_english(strs):
    for i in strs:
        if i not in string.ascii_lowercase + string.ascii_uppercase:
            return False
    return True

def process_text(text):
    seq = ''
    last_state = False
    for token in text.strip().split():
        curr_state = is_all_english(token)
        if last_state and curr_state:
            seq += f" {token}"
        else:
            seq += token
        last_state = curr_state
    return seq


with jsonlines.open(data_path, 'r') as reader:
    data = [line for line in reader]

pbar = tqdm(total=len(data))
for id, session in enumerate(data):
    for utterance in session:
        utterance['parse'] = parse(utterance['text'])
    if (id + 1) % 10 == 0:
        pbar.update(10)
pbar.close()

with jsonlines.open(output_path, 'w') as writer:
    writer.write_all(data)
