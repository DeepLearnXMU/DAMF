import sys
import random
import importlib
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from dataset import RAQGDataset, DuSincDataset
from utils import *
# from config import *
from nltk import word_tokenize
from model import T5ForConditionalGenerationRAQG

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, truncation_side = 'left')
    model = T5ForConditionalGenerationRAQG.from_pretrained(args.model_name_or_path)
    model = model.cuda()

    def collate_fn(batch):
        inputs, targets, scores = zip(*batch)
        inputs, targets, scores = inputs[0], targets[0], scores[0] # for batch size is 1
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True,
                                 truncation=True, return_tensors='pt')
        if args.add_prefix:
            targets = [f'<extra_id_1>{target}' for target in targets]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding=True,
                               truncation=True, return_tensors='pt')
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]

        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["rewards"] = torch.tensor(scores)

        if args.add_prefix:
            model_inputs["labels"][:, 0] = -100
        return model_inputs

    def support_collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs, targets = list(inputs), list(targets)
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True,
                                 truncation=True, return_tensors='pt')
        if args.add_prefix:
            targets = [f'<extra_id_1>{target}' if target else '<extra_id_0>' for target in targets]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding=True,
                               truncation=True, return_tensors='pt')
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]

        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
        model_inputs["decoder_input_ids"] = decoder_input_ids
        return model_inputs

    def eval_collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs, targets = list(inputs), list(targets)
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True,
                                 truncation=True, return_tensors='pt')
        if args.add_prefix:
            model_inputs["decoder_input_ids"] = torch.LongTensor([[0, tokenizer.convert_tokens_to_ids('<extra_id_1>')]]*len(inputs))
        return model_inputs, targets

    train_dataset = RAQGDataset(args.train_file, args.topk, bm25=args.bm25, threshold=args.threshold, gap=args.gap)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_dataset = DuSincDataset(args.valid_file)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=eval_collate_fn)

    support_dataset = DuSincDataset(args.support_file)
    support_dataloader = DataLoader(support_dataset, batch_size=args.support_batch_size, shuffle=True, collate_fn=support_collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    def evaluate(model):
        predictions, references = [], []
        model.eval()
        for inputs, refs in valid_dataloader:
            inputs = preprocess_batch(inputs)
            with torch.no_grad():
                pred_tokens = model.generate(**inputs, num_beams=4, max_length=args.max_target_length)
            preds = tokenizer.batch_decode(pred_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            predictions += preds
            references += refs
        if args.lang == 'en':
            def preprocess(strs):
                return [word_tokenize(str) for str in strs]
            predictions, references = preprocess(predictions), preprocess(references)
        return uni_F1_score(predictions, references)

    eval_score = evaluate(model)
    print(f"Step {0}, Evaluation score {round(eval_score, 4)}")

    if args.do_train:
        backward_steps, global_steps = 0, 0
        train_loss, support_loss = 0., 0.
        support_iterator = iter(support_dataloader)
        model.train()
        pbar = tqdm(total=len(train_dataloader) * args.epoches // args.gradient_accumulation_steps)
        for epoch in range(args.epoches):
            model.train()
            for batch in train_dataloader:
                batch = preprocess_batch(batch)
                loss = model.forward_ra(**batch)
                loss /= args.gradient_accumulation_steps
                train_loss += loss.item()
                loss.backward()
                backward_steps += 1
                if backward_steps % args.gradient_accumulation_steps == 0:
                    if args.do_support:
                        support_batch = next(support_iterator, None)
                        if support_batch is None:
                            support_iterator = iter(support_dataloader)
                            support_batch = next(support_iterator, None)
                        support_batch = preprocess_batch(support_batch)
                        support_loss = model(**support_batch).loss * args.alpha
                        support_loss.backward()
                        support_loss = support_loss.item()
                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch+1} Loss {round(train_loss, 3)} "
                                         f"Support-Loss {round(support_loss, 3)} ")
                    global_steps += 1
                    train_loss = 0.
                    optimizer.step()
                    optimizer.zero_grad()
            eval_score = evaluate(model)
            print(f"Step {global_steps}, Evaluation score {round(eval_score, 4)}")
            if args.save_model:
                save_model(args, model, tokenizer, step=global_steps)
                print(f'saving model at checkpoint step {global_steps}')
        # eval_score = evaluate(model)
        # print(f"Step {global_steps}, Evaluation score {round(eval_score, 4)}")
        # save_model(args, model, tokenizer, step=global_steps)
        # print(f'saving model at checkpoint step {global_steps}')

if __name__ == '__main__':
    _, arg_name = sys.argv
    args = importlib.import_module(f"config.{arg_name}").Arguments()
    if args.random_seed:
        seed = random.randint(0, 9999)
        random.seed(seed)
        torch.manual_seed(seed)
        args.output_dir = f"{args.output_dir}-{seed}"
        print(f"Trained models will be saved in {args.output_dir}..")
    main(args)
