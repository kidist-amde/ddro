import argparse
import csv
import gzip
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

from T5ForPretrain import T5ForPretrainDPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_doc_offsets(path):
    offsets = {}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for docid, _, offset in reader:
            offsets[docid] = int(offset)
    return offsets


def load_checkpoint(path):
    raw_state = torch.load(path)
    state = {}
    for k, v in raw_state.items():
        key = ".".join(k.split(".")[1:]) if k.startswith("module.") else k
        state[key] = v
    return state


def load_query_texts(path):
    mapping = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            topic_id, query = line.strip().split("\t")
            mapping[topic_id] = query
    return mapping


def load_encoded_docids(path):
    doc_map = {}
    with open(path, "r") as f:
        for line in f:
            docid, pq_ids = line.strip().split("\t")
            docid = docid.strip("[]").upper()
            doc_map[docid] = [int(x) for x in pq_ids.split(",")]
    return doc_map


def load_training_data(path, doc_map, query_map):
    pairs = []
    with open(path, "r") as f:
        for line in f:
            qid, pos_id, neg_id = line.strip().split("\t")
            if pos_id in doc_map and neg_id in doc_map:
                pairs.append({
                    "query": query_map[qid],
                    "positive_doc_id": doc_map[pos_id],
                    "negative_doc_id": doc_map[neg_id]
                })
    return pairs


def return_prompt_and_responses(samples):
    return {
        "prompt": samples["query"],
        "chosen": samples["positive_doc_id"],
        "rejected": samples["negative_doc_id"],
    }


def create_datasets(docid_path, train_file, dev_file, train_queries_file, dev_queries_file, num_proc=24):
    docids = load_encoded_docids(docid_path)
    train_queries = load_query_texts(train_queries_file)
    dev_queries = load_query_texts(dev_queries_file)

    train = Dataset.from_list(load_training_data(train_file, docids, train_queries))
    dev = Dataset.from_list(load_training_data(dev_file, docids, dev_queries))
    train = train.map(return_prompt_and_responses, batched=True, num_proc=num_proc, remove_columns=train.column_names)
    dev = dev.map(return_prompt_and_responses, batched=True, num_proc=num_proc, remove_columns=dev.column_names)
    return train, dev


def tokenize_encoder_decoder(batch, tokenizer, prompt, chosen, rejected, args):
    chosen_tokens = {"input_ids": chosen, "attention_mask": [1] * len(chosen)}
    rejected_tokens = {"input_ids": rejected, "attention_mask": [1] * len(rejected)}
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True)

    batch["chosen_labels"] = chosen_tokens["input_ids"]
    batch["rejected_labels"] = rejected_tokens["input_ids"]
    batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]


def patch_trl_tokenizer():
    def _tokenize(features, tokenizer, args, processor=None, model=None):
        batch = defaultdict(list)
        if model is None:
            return super(type(model), model)._tokenize(features, tokenizer, args, processor, model)
        tokenize_encoder_decoder(batch, tokenizer, features["prompt"], features["chosen"], features["rejected"], args)
        return dict(batch)

    import trl.trainer.dpo_trainer
    trl.trainer.dpo_trainer._tokenize = _tokenize


class T5DPOTrainer(DPOTrainer):
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        if not self.is_encoder_decoder:
            return super().tokenize_row(feature, model)

        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)

        batch = {
            "chosen_labels": chosen["input_ids"].to(device),
            "rejected_labels": rejected["input_ids"].to(device),
            "prompt_input_ids": prompt_tokens["input_ids"].to(device),
            "prompt_attention_mask": prompt_tokens["attention_mask"].to(device)
        }

        if model and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["rejected_labels"]).to(device))
            batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["chosen_labels"]).to(device))

        return batch


def load_model_from_checkpoint(cli_args):
    base_model = T5ForConditionalGeneration.from_pretrained(cli_args.pretrain_model_path, device_map="auto")
    base_model.resize_token_embeddings(base_model.config.vocab_size + 6144)
    model = T5ForPretrainDPO(base_model, cli_args)
    model.load_state_dict(load_checkpoint(cli_args.checkpoint_path))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train T5 model using DPO")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--docid_path", type=str, required=True)
    parser.add_argument("--train_queries_file", type=str, required=True)
    parser.add_argument("--dev_queries_file", type=str, required=True)
    parser.add_argument("--pretrain_model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--use_origin_head", type=str, default="False")
    args = parser.parse_args()

    patch_trl_tokenizer()

    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_data, dev_data = create_datasets(
        args.docid_path, args.train_file, args.dev_file,
        args.train_queries_file, args.dev_queries_file
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=64,
        save_total_limit=2,
        save_steps=500,
        eval_steps=500,
        num_train_epochs=2,
        learning_rate=1e-6,
        warmup_steps=1000,
        max_grad_norm=0.5,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
    )

    model = load_model_from_checkpoint(args).to(device)
    model_ref = load_model_from_checkpoint(args).to(device)

    trainer = T5DPOTrainer(
        model=model,
        model_ref=model_ref,
        args=training_args,
        beta=0.4,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        is_encoder_decoder=True,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()
    trainer.save_model()
    torch.save(model.state_dict(), os.path.join(args.output_dir, "dpo_model_final.pkl"))


if __name__ == "__main__":
    main()
