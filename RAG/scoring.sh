#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3 python retrieval_scoring.py \
        --model_path ../saved_data/RAG_kdconv/mengzi-t5-base/checkpoint/best_dev \
        --eval_data ../saved_data/data_kdconv/train_rag.json \
        --model_name Langboat/mengzi-t5-base \
        --per_gpu_batch_size 1 \
        --n_context 1000 \
        --name my_test \
        --text_maxlength 100 \
        --doc_maxlength 150 \
        --answer_maxlength 64 \
        --n_docs 5 \
        --lang zh \
        --output_file ../saved_data/data_kdconv/train.json \
        --checkpoint_dir ../saved_data/RAG_kdconv