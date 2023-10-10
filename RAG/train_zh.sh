#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python train_reader.py \
        --train_data ../saved_data/DuConv/train.json \
        --eval_data ../saved_data/DuConv/dev.json \
        --model_name Langboat/mengzi-t5-base \
        --per_gpu_batch_size 1 \
        --n_context 50 \
        --name mengzi-t5-base \
        --checkpoint_dir ../saved_data/RAG_duconv \
        --text_maxlength 150 \
        --doc_maxlength 150 \
        --answer_maxlength 64 \
        --accumulation_steps 64 \
        --total_steps 5000 \
        --save_freq 500 \
        --eval_freq 500 \
        --eval_print_freq 500 \
        --lr 5e-5 \
        --warmup_step 500 \
        --n_docs 5 \
        --scheduler linear