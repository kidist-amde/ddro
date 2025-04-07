import argparse
import gzip
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments
)

def extract_query_doc_pairs(input_path: str, output_path: str):
    with gzip.open(input_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=[
            'query', 'id', 'long_answer', 'short_answer', 'title',
            'abstract', 'content', 'document_url', 'doc_tac', 'language'
        ])
    df_extracted = df[['id', 'query', 'doc_tac']].fillna('')
    df_extracted.to_csv(output_path, sep='\t', index=False, header=False, compression='gzip')

def prepare_dataset(data_path: str, tokenizer, max_length: int = 512):
    df = pd.read_csv(data_path, sep='\t', names=['id', 'query', 'doc_tac'], compression='gzip')
    df['doc_tac'] = df['doc_tac'].fillna('').astype(str)

    def tokenize(example):
        inputs = tokenizer(
            example['doc_tac'], max_length=max_length,
            truncation=True, padding="max_length"
        )
        targets = tokenizer(
            example['query'], max_length=max_length,
            truncation=True, padding="max_length"
        )
        inputs['labels'] = targets['input_ids']
        return inputs

    dataset = Dataset.from_pandas(df[['query', 'doc_tac']])
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

def split_dataset(dataset, test_size: float = 0.2):
    split = dataset.train_test_split(test_size=test_size)
    return split['train'], split['test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache")
    args = parser.parse_args()

    extract_query_doc_pairs(args.dataset_path, args.output_file)

    tokenizer = T5TokenizerFast.from_pretrained(
        "castorini/doc2query-t5-large-msmarco",
        cache_dir=args.cache_dir,
        legacy=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        "castorini/doc2query-t5-large-msmarco",
        cache_dir=args.cache_dir
    )

    dataset = prepare_dataset(args.output_file, tokenizer)
    train_dataset, eval_dataset = split_dataset(dataset)

    training_args = TrainingArguments(
        output_dir="resources/transformer_models",
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=2000,
        logging_dir="logs",
        logging_steps=500,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model("resources/transformer_models/finetuned_doc2query_t5_large_msmarco")

if __name__ == "__main__":
    main()
