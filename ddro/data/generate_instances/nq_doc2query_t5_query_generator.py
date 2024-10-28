import json
import os
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
import sys

# Set random seed for reproducibility
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
        end = min(i + batch_size, len(data))
        samples = [data[j] for j in range(i, end)]
        yield SubSampleData(samples)

# Custom dataset class to handle nq-docs-sents.json structure
class GenerateDataset(Dataset):
    def __init__(self, path_to_data, max_length, cache_dir, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if "nq-docs-sents.json" in path_to_data:
            print("Loading nq-docs-sents.json dataset for docTquery generation...")
            with open(path_to_data, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.samples.append({
                        'id': data.get('id', ''),
                        'contents': data.get('contents', '')  # Using 'contents' as the document text
                    })
            print(f"Loaded {len(self.samples)} samples from dataset", flush=True)
        else:
            raise NotImplementedError(f"Dataset {path_to_data} for docTquery generation is not defined.")

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
        tokenized['doc_id'] = sample['id']  # Adding document ID for reference
        return tokenized

# Custom collator to handle padding and format requirements
class QueryEvalCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding=True):
        super().__init__(tokenizer, padding=padding)
    
    def __call__(self, features):
        # Ensure each feature has an 'input_ids' key before accessing it
        if all('input_ids' in feature for feature in features):
            return super().__call__(features)
        else:
            raise KeyError("Each feature must contain 'input_ids'. Please check your dataset processing.")

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-queries for documents.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--cache_dir", type=str, default='cache', help="Directory to store cached models and tokenizer")
    parser.add_argument("--output_path", type=str, default='pseudo_queries_output.json', help="Path to the output file.")
    parser.add_argument("--max_docs", type=int, default=10, help="Maximum number of documents to process.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of the documents")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling value for generating queries")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="Number of query sequences to return per document")
    parser.add_argument("--q_max_length", type=int, default=32, help="Maximum length of the generated queries")
    parser.add_argument("--checkpoint_path", type=str, default='resources/transformer_models/docTTTTTquery_finetuned/finetuned_doc2query_t5_large_msmarco', help="Path to the model checkpoint")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Initializing training arguments...", flush=True)
    training_args = TrainingArguments(
        "psudo_query_genration",
        per_device_eval_batch_size=4,
        dataloader_num_workers=10,
        report_to="none",
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )

    # Define checkpoint path
    checkpoint_path =args.checkpoint_path

    # Check and save tokenizer files if missing
    if not all(os.path.exists(os.path.join(checkpoint_path, file)) for file in ["tokenizer.json", "tokenizer_config.json"]):
        print("Tokenizer files missing in checkpoint. Initializing from base model and saving...", flush=True)
        tokenizer = T5Tokenizer.from_pretrained("castorini/doc2query-t5-large-msmarco")
        tokenizer.save_pretrained(checkpoint_path)
    else:
        print("Loading tokenizer from checkpoint...", flush=True)
        tokenizer = T5Tokenizer.from_pretrained(checkpoint_path, cache_dir=args.cache_dir)

    print("Loading model from checkpoint...", flush=True)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path, cache_dir=args.cache_dir)
    fast_tokenizer = T5TokenizerFast.from_pretrained(checkpoint_path, cache_dir=args.cache_dir)

    print("Initializing dataset...", flush=True)
    generate_dataset = GenerateDataset(
        path_to_data=args.input_file,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=QueryEvalCollator(tokenizer, padding='longest'),
    )

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf8') as f:
            existing_docids = set([json.loads(line)['doc_id'] for line in f.readlines()])
    else:
        existing_docids = set()

    processed_docs = 0
    print("Starting batch processing...", flush=True)
    
    with open(args.output_path, 'a', encoding='utf-8') as outfile:
        for batch in batchify(generate_dataset, args.batch_size):
            input_ids = torch.cat([item['input_ids'] for item in batch], dim=0).to(model.device)
            attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0).to(model.device)
            doc_ids = [item['doc_id'] for item in batch]

            with torch.no_grad():
                generated_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.q_max_length,
                    top_k=args.top_k,
                    num_return_sequences=args.num_return_sequences,
                    do_sample=True  # Enable sampling to allow multiple sequences
                )

            for i, output in enumerate(generated_outputs):
                doc_id = doc_ids[i // args.num_return_sequences]  # Adjust doc_id indexing for multiple sequences
                query = fast_tokenizer.decode(output, skip_special_tokens=True)
                jitem = json.dumps({'doc_id': doc_id, 'query': query})
                outfile.write(jitem + '\n')

            processed_docs += len(batch)
            print(f"Processed {processed_docs} documents so far...", flush=True)
            if processed_docs >= args.max_docs:
                print(f"Reached max document limit of {args.max_docs}. Stopping.", flush=True)
                break

    print("Pseudo-query generation completed.", flush=True)

if __name__ == "__main__":
    main()
