# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random

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
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    uf1_scores = []
    pbar = tqdm(total=len(dataloader), desc='Evaluating Uni. F1')
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, n_docs) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                n_docs=n_docs,
                max_length=opt.answer_maxlength,
                num_beams=4,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'response' in example:
                    score = src.evaluation.ufs(ans, [example['response']])
                    # score = src.evaluation.ems(ans, example['answers'])
                    uf1_scores.append(score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')

                total += 1
            # if (i + 1) % opt.eval_print_freq == 0:
            #     log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
            #     if len(uf1_scores) == 0:
            #         log += '| no answer to compute scores'
            #     else:
            #         log += f' | average = {np.mean(uf1_scores):.3f}'
            #     logger.warning(log)
            if total % 5 == 0:
                pbar.set_description(f'Evaluating Uni. F1 {round(np.mean(uf1_scores), 4)}')
            pbar.update()

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(uf1_scores):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(uf1_scores), total, opt)
    
    return score, total


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
                                 doc_maxlength=opt.doc_maxlength, answer_maxlength=opt.answer_maxlength)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context,
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
    uf1_scores, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'Uni.F1 {100*uf1_scores:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path)

