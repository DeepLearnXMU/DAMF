# coding=utf-8
import os

import torch

os.environ['CUDA_VISIBLE_DEVICES']='1'
import re
import string
import six
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import jsonlines
from sacrebleu.metrics import BLEU
import numpy as np
from tqdm import tqdm

from rouge_score.rouge_scorer import _score_lcs, _summary_level_lcs, _create_ngrams, _score_ngrams, RougeScorer


class CNRougeScorer(RougeScorer):
    def __init__(self, rouge_types):
        super().__init__(rouge_types)

    def tokenize(self, str):
        def is_english(char):
            return char in string.ascii_letters + string.digits
        tokens = []
        last_state = None
        curr_token = ''
        for char in str.strip():
            curr_state = is_english(char)
            if curr_state:
                curr_token += char
            else:
                if last_state:
                    tokens.append(curr_token)
                    curr_token = ''
                tokens.append(char)
            last_state = curr_state
        if curr_token:
            tokens.append(curr_token)
        tokenized_str = re.sub('\s+', ' ', ' '.join(tokens))
        return tokenized_str

    def score(self, target, prediction):
        """Calculates rouge scores between the target and prediction.

        Args:
          target: Text containing the target (ground truth) text.
          prediction: Text containing the predicted text.
        Returns:
          A dict mapping each rouge type to a Score object.
        Raises:
          ValueError: If an invalid rouge type is encountered.
        """

        target_tokens = self.tokenize(target)
        prediction_tokens = self.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = _score_lcs(target_tokens, prediction_tokens)
            elif rouge_type == "rougeLsum":
                # Note: Does not support multi-line text.
                def get_sents(text):
                    # Assume sentences are separated by newline.
                    sents = six.ensure_str(text).split("\n")
                    sents = [x for x in sents if len(x)]
                    return sents

                target_tokens_list = [
                    self.tokenize(s) for s in get_sents(target)]
                prediction_tokens_list = [
                    self.tokenize(s) for s in get_sents(prediction)]
                scores = _summary_level_lcs(target_tokens_list,
                                            prediction_tokens_list)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = _create_ngrams(target_tokens, n)
                prediction_ngrams = _create_ngrams(prediction_tokens, n)
                scores = _score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result

scorer = CNRougeScorer(['rouge1', 'rouge2', 'rougeL'])

def cal_rouge_score(preds, labels):
    rouge1, rouge2, rougeL = [], [], []
    for pred, label in zip(preds, labels):
        if isinstance(label, list):
            max_rouge1, max_rouge2, max_rougeL = 0, 0, 0
            for ref in label:
                score = scorer.score(ref, pred)
                max_rouge1 = max(max_rouge1, score['rouge1'][2])
                max_rouge2 = max(max_rouge2, score['rouge2'][2])
                max_rougeL = max(max_rougeL, score['rougeL'][2])
                rouge1.append(max_rouge1)
                rouge2.append(max_rouge2)
                rougeL.append(max_rougeL)
        else:
            score = scorer.score(label, pred)
            rouge1.append(score['rouge1'][2])
            rouge2.append(score['rouge2'][2])
            rougeL.append(score['rougeL'][2])
    return np.mean(rouge1), np.mean(rouge2), np.mean(rougeL)


def unigram_f1_score(prediction, ground_truth):
    pred_set = set(prediction)
    ref_set = set(ground_truth)
    common_len = len(pred_set & ref_set)
    pred_len = len(prediction)
    ref_len = len(ground_truth)
    p = common_len / pred_len
    r = common_len / ref_len
    if p > 0 and r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0

if __name__ == '__main__':
    model_path = '../saved_data/mengzi-t5-base-dusinc'
    data_path = '../saved_data/data_dusinc/dev.json'

    bleu1 = BLEU(tokenize='zh', max_ngram_order=1)
    bleu2 = BLEU(tokenize='zh', max_ngram_order=2)

    inputs, targets = [], []
    with jsonlines.open(data_path, 'r') as reader:
        for line in reader:
            inputs.append(line['dialogue'])
            targets.append(line['query'])

    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left',)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    pos_tok, neg_tok = tokenizer.convert_tokens_to_ids('<extra_id_1>'), tokenizer.convert_tokens_to_ids('<extra_id_0>')

    # accuracy
    predictions, references = [], []
    hit = []
    pbar = tqdm(total=len(inputs))
    for input, target in zip(inputs, targets):
        model_inputs = tokenizer(input, return_tensors='pt', max_length=200, truncation=True)
        model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

        if target is not None:
            model_inputs['decoder_input_ids'] = torch.LongTensor([[0, pos_tok]]).cuda()
            prediction = model.generate(**model_inputs, num_beams=4, max_length=50)[0]
            prediction = tokenizer.decode(prediction, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            predictions.append(prediction)
            references.append(target)

        model_inputs['decoder_input_ids'] = torch.LongTensor([[0]]).cuda()
        logits = model(**model_inputs).logits
        pos_prob = logits[0, 0, pos_tok]
        neg_prob = logits[0, 0, neg_tok]
        if (pos_prob > neg_prob and target) or (pos_prob <= neg_prob and target is None):
            hit.append(1)
        else:
            hit.append(0)

        pbar.update()

    print(f'Accuracy {np.mean(hit)}')
    b1_score = bleu1.corpus_score(predictions, [references]).score
    b2_score = bleu2.corpus_score(predictions, [references]).score
    print(f'BLEU-1/2 {round(b1_score, 2)} / {round(b2_score, 2)}')
    uni_f1 = np.mean(
        [unigram_f1_score(prediction, reference) for prediction, reference in zip(predictions, references)])
    print(f'Uni. F1 {round(uni_f1 * 100, 2)}')
    r1, r2, rL = cal_rouge_score(predictions, references)
    print(f'Rouge-1/2/L {round(r1 * 100, 2)} / {round(r2 * 100, 2)} / {round(rL * 100, 2)}')
