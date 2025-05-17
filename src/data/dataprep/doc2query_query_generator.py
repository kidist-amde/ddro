
import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
    Trainer
)

set_seed(313)

class SubSampleData(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield SubSampleData([data[j] for j in range(i, min(i + batch_size, len(data)))])

class DocumentDataset(Dataset):
    def __init__(self, data_path, max_length, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append({
                    'id': data.get('id', ''),
                    'contents': data.get('contents', '')
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokenized = self.tokenizer(
            sample['contents'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized['doc_id'] = sample['id']
        return tokenized

class QueryCollator(DataCollatorWithPadding):
    def __call__(self, features):
        if all('input_ids' in feature for feature in features):
            return super().__call__(features)
        raise KeyError("Missing 'input_ids' in one or more features.")

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-queries using T5-based doc2query model.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL document file")
    parser.add_argument("--cache_dir", type=str, default='cache', help="Cache directory for transformers")
    parser.add_argument("--output_path", type=str, default='pseudo_queries_output.json', help="Output file path")
    parser.add_argument("--max_docs", type=int, default=10, help="Maximum number of documents to process")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=256, help="Max input sequence length")
    parser.add_argument("--q_max_length", type=int, default=32, help="Max generated query length")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling for generation")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="Number of queries per document")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained checkpoint")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    training_args = TrainingArguments(
        output_dir="pseudo_query_gen",
        per_device_eval_batch_size=4,
        dataloader_num_workers=4,
        report_to="none",
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )

    if not os.path.exists(os.path.join(args.checkpoint_path, "tokenizer.json")):
        tokenizer = T5Tokenizer.from_pretrained("castorini/doc2query-t5-large-msmarco")
        tokenizer.save_pretrained(args.checkpoint_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_path, cache_dir=args.cache_dir)

    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path, cache_dir=args.cache_dir)
    fast_tokenizer = T5TokenizerFast.from_pretrained(args.checkpoint_path, cache_dir=args.cache_dir)

    dataset = DocumentDataset(
        data_path=args.input_file,
        max_length=args.max_length,
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=QueryCollator(tokenizer, padding='longest')
    )

    existing_docids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            existing_docids = {json.loads(line)['doc_id'] for line in f}

    processed = 0
    with open(args.output_path, 'a', encoding='utf-8') as fout:
        for batch in batchify(dataset, args.batch_size):
            input_ids = torch.cat([b['input_ids'] for b in batch], dim=0).to(model.device)
            attention_mask = torch.cat([b['attention_mask'] for b in batch], dim=0).to(model.device)
            doc_ids = [b['doc_id'] for b in batch]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.q_max_length,
                    top_k=args.top_k,
                    num_return_sequences=args.num_return_sequences,
                    do_sample=True
                )

            for i, out in enumerate(outputs):
                doc_id = doc_ids[i // args.num_return_sequences]
                query = fast_tokenizer.decode(out, skip_special_tokens=True)
                fout.write(json.dumps({"doc_id": doc_id, "query": query}) + '\n')

            processed += len(batch)
            print(f"Processed {processed} documents...", flush=True)
            if processed >= args.max_docs:
                print(f"Reached max document limit ({args.max_docs}). Stopping.", flush=True)
                break

    print("Pseudo-query generation completed.", flush=True)

if __name__ == "__main__":
    main()