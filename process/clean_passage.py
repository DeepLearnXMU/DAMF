'''
This script aims to clean webpages
--> 10% noisy char cleaned
'''

from sys import argv
import unicodedata
import jionlp
import jsonlines
from tqdm import tqdm
import re
from jionlp.rule.rule_pattern import EXCEPTION_PATTERN

_, data_path, output_path = argv

with jsonlines.open(data_path, 'r') as reader:
    data = [line for line in reader]

exception_pattern = re.compile(EXCEPTION_PATTERN)
def remove_exception_char(text):
    return exception_pattern.sub('', text)

pbar = tqdm(total=len(data))
cleaned_passages = []
avg_len_b, avg_len_a = 0, 0
for id, example in enumerate(data):
    query = example['query']
    cleaned_abstracts = []
    for page in example['passages']:
        avg_len_b += len(page['abstract'])
        abstract = unicodedata.normalize("NFKD", page['abstract'])
        abstract = jionlp.clean_text(abstract, remove_phone_number=False, remove_exception_char=False)
        abstract = remove_exception_char(abstract).strip()
        if abstract:
            cleaned_abstracts.append(abstract)
        avg_len_a += len(abstract)
    cleaned_passages.append({'query': query, 'passages': cleaned_abstracts})
    if (id + 1) % 50 == 0:
        pbar.update(50)
print(avg_len_a / avg_len_b)

with jsonlines.open(output_path, 'w') as writer:
    writer.write_all(cleaned_passages)
