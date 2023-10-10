import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
from sys import argv
import string
import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_, data_path, model_path, output_path, query_num = argv
query_num = int(query_num)

tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left',)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
pos_tok, neg_tok = tokenizer.convert_tokens_to_ids('<extra_id_1>'), tokenizer.convert_tokens_to_ids('<extra_id_0>')

def is_all_english(strs):
    for i in strs:
        if i not in string.ascii_lowercase + string.ascii_uppercase:
            return False
    return True

# this should be done in previous steps
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

def generate(session):
    inputs = ['\n'.join(session[:i]) for i in range(1, len(session))]

    model_inputs = tokenizer(inputs, return_tensors='pt', max_length=200, truncation=True, padding=True)
    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

    model_inputs['decoder_input_ids'] = torch.LongTensor([[0, pos_tok]]).cuda().expand(len(inputs), 2)
    predictions = model.generate(**model_inputs, num_beams=query_num, num_return_sequences=query_num, max_length=50)
    predictions = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    assert len(predictions) == len(inputs) * query_num

    output = [{'text': session[0], 'queries': []}]
    for idx in range(len(session) - 1):
        prediction = predictions[query_num * idx: query_num * idx + query_num]
        updated_queries = []
        # filter low quality queries, the top prediction can repeat at most 1 char
        query = prediction[0]
        if len(query) - len(set(query)) <= 1:
            updated_queries.append(query)
        for query in prediction[1:]:
            if len(query) == len(set(query)) and query not in updated_queries:
                updated_queries.append(query)
        output.append({'text': session[idx + 1], 'queries': updated_queries})
    return output

reader = jsonlines.open(data_path, 'r')
raw_data = [line for line in reader]
outputs = []
for line in tqdm(raw_data):
    outputs.append(generate(line))
# outputs = [generate(line) for line in reader]

writer = jsonlines.open(output_path, 'w')
writer.write_all(outputs)

