#!/usr/bin/env bash

DATA_DIR='../saved_data/data_dusinc'
OUTPUT_DIR='../saved_data'

CUDA_VISIBLE_DEVICES=1 python run_model.py \
    --model_name_or_path Langboat/mengzi-t5-base \
    --do_train \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/dev.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/mengzi-t5-base-dusinc" \
    --per_device_train_batch_size=50 \
    --per_device_eval_batch_size=50 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --num_train_epochs 4 \
    --save_total_limit 1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --weight_decay 0.01 \
    --text_column="dialogue" \
    --summary_column="query" \
    --remove_unused_columns false \
    --max_source_length 200 \
    --max_target_length 50 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir