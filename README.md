Code for paper "Domain Adaptation for Conversational Query Production with the RAG Model Feedback" (EMNLP23 Findings)

# Initialization: Supervised Training
## Input Format
Prepare the train, valid, test file that end with '.json'. For each file, each line is a instance dict like:
> {"dialogue": xxx, "query": yyy, "response": zzz}
> {"dialogue": xxx, "query": yyy, "response": zzz}

## Training and Testing
Run the training script 'Baseline/train_woi.sh' to train for WOI dataset and 'Baseline/train_dusinc.sh' to train for DUSINC dataset. Then, use evaluation script 'Baseline/eval.py' to get the automatic evaluation results.

## Stage 1: Data Preparation
To get the data for the following procedure, you need a search engine API to retrieve knowledge. We provide our processing script for this step. However, you have to rewrite the search engine calling part and adjust some other code according to your used dataset.
Make sure the file format are consisted as they will be fed to models:
> step 6: prepare data for RAG training
> {"content": "a dialogue context", "response": "the target response", "knowledge": [{"text": "document content", "queries": ["query to retrieve this document", "..."]}]

> step 7: prepare data for QP training using BM25 scores
> {"input": "dialogue context", "candidate_queries": {"query_A": "score_A", ...}}

> step 8: prepare data for RAG evaluation, which will be fed to a trained RAG model for the QP training file with RAG retrieval scores. Same as the format of step 6 output file.

## Stage 2: RAG training
Fed the output file of step 6 above to train the RAG model. The training script is "RAG/train_zh.sh" and "RAG/train_en.sh". Then, fed the output file of step 8 to run script "RAG/scoring.sh" to get the training file for QP guided with RAG retrieval scores.

## Stage 3: Reinforcement learning
Run the script "DAMF/train.sh" to train our model. When training is done, use the same script "Baseline/eval.py" for evaluation. For Chinese annotated test set, we have provided them under "eval_data". Note that it is only used for reproducing query production in our paper. We reported results on full test datasets for response generation.