'''
This script is used to tokenize webpages using texsmart (M)
'''

#encoding=utf-8
import sys
from sys import argv
_, texsmart_lib, texsmart_kb, data_path, output_path = argv
sys.path.append(texsmart_lib)
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

with jsonlines.open(data_path, 'r') as reader:
    data = [line for line in reader]

pbar = tqdm(total=len(data))
for id, instance in enumerate(data):
    instance['parse'] = [parse(text) for text in instance['passages']]
    if (id + 1) % 10 == 0:
        pbar.update(10)

with jsonlines.open(output_path, 'w') as writer:
    writer.write_all(data)
