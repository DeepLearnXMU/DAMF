# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from tqdm import tqdm

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    _step = 0
    model.train()
    pbar = tqdm(total=opt.total_steps, desc='EPOCH 1')
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            _step += 1
            (idx, labels, _, context_ids, context_mask, n_docs) = batch
            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                n_docs=n_docs,
                return_dict=True
            ).loss/opt.accumulation_steps

            train_loss.backward()
            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            if _step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                pbar.update(1)
                pbar.set_description(f'EPOCH {epoch}')

                _step = 0
                step += 1
                if step % opt.eval_freq == 0:
                    dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                    model.train()
                    if opt.is_main:
                        if dev_em > best_dev_em:
                            best_dev_em = dev_em
                            src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                      opt, checkpoint_path, 'best_dev')
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                        # log += f"evaluation: {100*dev_em:.2f}EM |"
                        log += f"evaluation: {100*dev_em:.2f}Uni. F1 |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        logger.info(log)
                        if tb_logger is not None:
                            tb_logger.add_scalar("Evaluation", dev_em, step)
                            tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                        curr_loss = 0.

                if opt.is_main and step % opt.save_freq == 0:
                    src.util.save(model, optimizer, scheduler, step, best_dev_em,
                              opt, checkpoint_path, f"step-{step}")
                if step > opt.total_steps:
                    break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, n_docs) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                n_docs=n_docs,
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['response']
                # score = src.evaluation.ems(ans, gold)
                score = src.evaluation.ufs(ans, gold if isinstance(gold, list) else [gold], lang=opt.lang)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()

    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = opt.model_name
    model_class = src.model.T5ForRAG

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    # add special token
    special_tokens = {'sep': '<sep>', 'cls': '<cls>'}
    tokenizer.add_tokens(list(special_tokens.values()))

    collator = src.data.Collator(tokenizer, special_tokens, text_maxlength=opt.text_maxlength,
                                 doc_maxlength=opt.doc_maxlength, answer_maxlength=opt.answer_maxlength, lang=opt.lang)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context, split='eval')

    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    t5.resize_token_embeddings(len(tokenizer))
    setattr(t5.config, "n_docs", opt.n_docs)
    model = src.model.T5ForRAG(t5.config)
    model.load_t5(t5.state_dict())
    # add special token
    model = model.to(opt.local_rank)
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
