# File: t5_docid_trainer.py

import os
import time
import torch
import random
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)

from utils import set_seed, load_model
from trie import Trie
from evaluate_per_query import evaluator
from pretrain.T5ForPretrain import T5ForPretrain
from pretrain_dataset import PretrainDataForT5
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--per_gpu_batch_size", default=25, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--output_every_n_step", default=25, type=int)
    parser.add_argument("--save_every_n_epoch", default=1, type=int)
    parser.add_argument("--operation", default="training", type=str)
    parser.add_argument("--use_docid_rank", default="False", type=str)
    parser.add_argument("--load_ckpt", default="False", type=str)
    parser.add_argument("--save_path", default="./model/", type=str)
    parser.add_argument("--log_path", default="./log/", type=str)
    parser.add_argument("--doc_file_path", required=True, type=str)
    parser.add_argument("--docid_path", required=True, type=str)
    parser.add_argument("--train_file_path", type=str)
    parser.add_argument("--test_file_path", type=str)
    parser.add_argument("--pretrain_model_path", required=True, type=str)
    parser.add_argument("--load_ckpt_path", default="./model/", type=str)
    parser.add_argument("--dataset_script_dir", type=str)
    parser.add_argument("--dataset_cache_dir", type=str)
    parser.add_argument("--add_doc_num", type=int)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_docid_length", type=int, default=1)
    parser.add_argument("--use_origin_head", default="False", type=str)
    parser.add_argument("--num_beams", default=10, type=int)
    return parser.parse_args()


def load_data(file_path):
    if os.path.isfile(file_path):
        return [file_path]
    return [os.path.join(file_path, fn) for fn in os.listdir(file_path)]


def load_encoded_docid(docid_path):
    encode_2_docid, encoded_docids = {}, []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            encode_list = [int(x) for x in encode.split(",") if x not in ["0", "1"]]
            encoded_docids.append(encode_list)
            encode_key = ','.join(map(str, encode_list))
            encode_2_docid.setdefault(encode_key, []).append(docid.lower())
    return encoded_docids, encode_2_docid


def prefix_allowed_tokens_fn_builder(trie, tokenizer):
    def fn(batch_id, sent):
        allowed = trie.get(sent.tolist())
        if not allowed:
            partial = sent.tolist()[:-1]
            fallback = trie.get(partial) if partial else []
            return fallback or [tokenizer.pad_token_id, tokenizer.eos_token_id]
        return allowed
    return fn


def train(args, tokenizer):
    train_data = load_data(args.train_file_path)
    model = T5ForPretrain(
        T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path).resize_token_embeddings(
            T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path).config.vocab_size + args.add_doc_num
        ),
        args
    )
    if args.load_ckpt == "True":
        model.load_state_dict(load_model(args.load_ckpt_path))
    model.to("cuda:0")
    model = torch.nn.DataParallel(model)

    dataset = PretrainDataForT5(train_data, args.max_seq_length, args.max_docid_length, tokenizer,
                                args.dataset_script_dir, args.dataset_cache_dir, args)
    dataloader = DataLoader(dataset, batch_size=args.per_gpu_batch_size * torch.cuda.device_count(), shuffle=True, num_workers=8)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)

    os.makedirs(args.save_path, exist_ok=True)
    logger = open(args.log_path, "a")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.cuda() for k, v in batch.items() if k not in ["query_id", "doc_id"]}
            loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["docid_labels"])
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            total_loss += loss.item()
            if step % args.output_every_n_step == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader):.4f}")
        if (epoch+1) % args.save_every_n_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch{epoch+1}.pt"))
    logger.close()


def evaluate(args, tokenizer):
    model = T5ForPretrain(
        T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path).resize_token_embeddings(
            T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path).config.vocab_size + args.add_doc_num
        ),
        args
    )
    model.load_state_dict(load_model(args.save_path))
    model.to("cuda:0")
    model.eval()

    encoded_docids, encode_2_docid = load_encoded_docid(args.docid_path)
    trie = Trie([[0]+docid for docid in encoded_docids])
    prefix_fn = prefix_allowed_tokens_fn_builder(trie, tokenizer)
    dataset = PretrainDataForT5(load_data(args.test_file_path), args.max_seq_length, args.max_docid_length, tokenizer,
                                args.dataset_script_dir, args.dataset_cache_dir, args)
    dataloader = DataLoader(dataset, batch_size=args.per_gpu_batch_size * torch.cuda.device_count(), shuffle=False, num_workers=8)

    truth, prediction = [], []
    for batch in tqdm(dataloader):
        batch = {k: v.cuda() for k, v in batch.items() if k not in ["query_id", "doc_id"]}
        input_ids = batch["input_ids"]
        labels = batch["docid_labels"]
        truth.extend([[','.join(map(str, doc[:doc.index(1)]))] for doc in labels.tolist()])
        outputs = model.module.generate(input_ids, max_length=args.max_docid_length+1, num_beams=args.num_beams,
                                        num_return_sequences=args.num_beams, prefix_allowed_tokens_fn=prefix_fn)
        for i in range(0, len(outputs), args.num_beams):
            doc_rank = []
            batch_output = outputs[i:i+args.num_beams].cpu().numpy().tolist()
            for docid in batch_output:
                try:
                    string = ','.join(str(x) for x in docid if x != 0 and x != 1)
                    doc_rank.append(string)
                except Exception:
                    doc_rank.append('')
            prediction.append(doc_rank)

    eval_df = evaluator().evaluate_ranking(truth, prediction)
    print(eval_df.mean())
    eval_df.to_csv(args.log_path.replace(".log", ".csv"), index=False)


if __name__ == '__main__':
    args = parse_arguments()
    set_seed()
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    if args.operation == "training":
        train(args, tokenizer)
    elif args.operation == "testing":
        evaluate(args, tokenizer)
