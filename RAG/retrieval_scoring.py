# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random

import jsonlines
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, dataset, dataloader, tokenizer, opt):

    model.eval()
    if hasattr(model, "module"):
        model = model.module

    pbar = tqdm(total=len(dataloader))
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, n_docs) = batch

            encoder = model.get_encoder()
            encoder_outputs = encoder(input_ids=context_ids.cuda(),
                                      attention_mask=context_mask.cuda(),
                                      n_docs=n_docs, )
            doc_scores = encoder_outputs["all_doc_scores"][0]
            # doc_scores = torch.nn.functional.softmax(doc_scores)
            doc_scores = doc_scores.tolist()

            example = dataset.data[idx[0]]
            scored_queries = {}
            for passage, score in zip(example['knowledge'], doc_scores[1:]):
                passage['score'] = score
                for query in passage['queries']:
                    if query in scored_queries:
                        scored_queries[query] = max(scored_queries[query], score)
                    else:
                        scored_queries[query] = score
            example['knowledge'].append({'text': '<EMPTY>', 'score': doc_scores[0]})
            scored_queries['<EMPTY>'] = doc_scores[0] # the first is empty sequence
            # queries, query_scores = zip(*[[k, v] for k, v in scored_queries.items()])
            # query_scores = torch.nn.functional.softmax(torch.tensor(query_scores))
            # query_scores = query_scores.tolist()
            # example['scored_queries'] = {k: v for k, v in zip(queries, query_scores)}
            example["scored_queries"] = scored_queries

            pbar.update()
    if opt.is_distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    model_name = opt.model_name
    model_class = src.model.T5ForRAG

    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)
    # add special token
    special_tokens = {'sep': '<sep>', 'cls': '<cls>'}
    tokenizer.add_tokens(list(special_tokens.values()))

    collator = src.data.Collator(tokenizer, special_tokens, text_maxlength=opt.text_maxlength,
                                 doc_maxlength=opt.doc_maxlength, answer_maxlength=opt.answer_maxlength, lang=opt.lang)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context,
        split='eval'
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=20, 
        collate_fn=collator
    )

    model = model_class.from_pretrained(opt.model_path)
    assert model.config.n_docs == opt.n_docs
    model = model.to(opt.device)

    logger.info("Start eval")
    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    with jsonlines.open(opt.output_file, 'w') as writer:
        writer.write_all([{'input': example['context'], 'candidate_queries': example['scored_queries']}
                          for example in eval_dataset.data])


