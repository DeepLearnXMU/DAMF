#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python train_reader.py \
        --train_data ../saved_data/data_wow/train.json \
        --eval_data ../saved_data/data_wow/dev.json \
        --model_name google/t5-v1_1-base \
        --per_gpu_batch_size 1 \
        --n_context 0 \
        --name t5-base \
        --checkpoint_dir ../saved_data/seq2seq_wow \
        --text_maxlength 100 \
        --doc_maxlength 150 \
        --answer_maxlength 64 \
        --accumulation_steps 64 \
        --total_steps 4000 \
        --save_freq 500 \
        --eval_freq 500 \
        --eval_print_freq 500 \
        --lr 5e-5 \
        --warmup_step 500 \
        --n_docs 4 \
        --scheduler linear \
        --lang en