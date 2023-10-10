#!/usr/bin/env bash

#### Script to prepare data for weakly-supervised training ####

# data dir
DATA_DIR=../saved_data/data_dusinc_low_resource
# target split: train/dev/test
SPLIT=dev
# data file path, this file should be a jsonl with each line as ``session[List(turn[String])]''
DATA_FILE_PATH=${DATA_DIR}/${SPLIT}.json
# model file path
MODEL_FILE_PATH=../saved_data/mengzi-t5-base-dusinc-0.1

echo 'step 1: generate candidate queries'
CANDIDATE_NUM=5 # generated query number for each utterance
DEVICE=7
STEP1_OUTPUT=${DATA_DIR}/${SPLIT}_with_queries.json
CUDA_VISIBLE_DEVICES=${DEVICE} python generate_queries.py ${DATA_FILE_PATH} ${MODEL_FILE_PATH} ${STEP1_OUTPUT} ${CANDIDATE_NUM}

echo 'step 2: using texsmart to parse data and extract entities'
STEP2_OUTPUT=${DATA_DIR}/${SPLIT}_parsed_with_queries.json
TEXSMART_LIB_PATH=/harddisk/user/antewang/texsmart-sdk-0.3.6-m/lib/
TEXSMART_KB_PATH=/harddisk/user/antewang/texsmart-sdk-0.3.6-m/data/nlu/kb/
python tokenize_dialogue.py ${TEXSMART_LIB_PATH} ${TEXSMART_KB_PATH} ${STEP1_OUTPUT} ${STEP2_OUTPUT}

echo 'step 3: retrieve passage from search engine, rewrite this part if using different search engine'
STEP3_OUTPUT=${DATA_DIR}/${SPLIT}_passages.json
python retrieve_passages.py ${STEP2_OUTPUT} ${STEP3_OUTPUT}

echo 'step 4: clean the noisy retrieved text'
STEP4_OUTPUT=${DATA_DIR}/${SPLIT}_cleaned_passages.json
python clean_passage.py ${STEP3_OUTPUT} ${STEP4_OUTPUT}

echo 'step 5: parse the retrieved text'
STEP5_OUTPUT=${DATA_DIR}/${SPLIT}_parsed_passages.json
python tokenize_passage.py ${TEXSMART_LIB_PATH} ${TEXSMART_KB_PATH} ${STEP4_OUTPUT} ${STEP5_OUTPUT}

echo 'step 6: prepare data for rag training'
STEP6_OUTPUT=${DATA_DIR}/${SPLIT}_rag_4_rg.json
MAX_PASSAGE_NUM=25
MAX_QUERY_NUM=10
python prepare_data_for_rag.py ${STEP2_OUTPUT} ${STEP5_OUTPUT} ${STEP6_OUTPUT} ${MAX_PASSAGE_NUM} ${MAX_QUERY_NUM}

echo 'step 7: prepare data for query production using bm25'
MAX_QUERY_NUM=10
STEP7_OUTPUT=${DATA_DIR}/${SPLIT}_bm25_4_qp.json
python prepare_data_for_qp.py ${STEP2_OUTPUT} ${STEP5_OUTPUT} ${STEP7_OUTPUT} ${MAX_QUERY_NUM}

echo 'step 8: prepare data for query evaluation using rag model'
STEP8_OUTPUT=${DATA_DIR}/${SPLIT}_rag_4_q_eval.json
python prepare_data_for_q_eval.py ${STEP2_OUTPUT} ${STEP5_OUTPUT} ${STEP8_OUTPUT} ${MAX_QUERY_NUM}

echo 'step 9: prepare data for query production using rag'
echo 'please do this step manually using scoring.py under RAG after the rg model is trained done'