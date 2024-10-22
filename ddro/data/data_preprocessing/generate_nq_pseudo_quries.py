import json
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    set_seed
)

# Import custom classes and functions
from utils.custom_datasets import GenerateDataset, QueryEvalCollator  # Make sure these are correctly implemented
from utils.custom_trainers import DSITrainer, DocTqueryTrainer  # Ensure these trainers are appropriately defined

# Set random seed for reproducibility
set_seed(313)

# Custom dataset class to handle memory-efficient data batching
class SubSampleData(Dataset):
    def __init__(self, samples):
        self.samples = samples 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

# Function to batchify data
def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        end = min(i + batch_size, len(data))  # Prevent out-of-index error
        samples = [data[j] for j in range(i, end)]
        yield SubSampleData(samples)

def main():
    # Argument parsing
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

    # Parse arguments
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="pseudo_query_generation",  # Change to relevant directory name
        per_device_eval_batch_size=4,  # Reduce batch size if memory is an issue
        dataloader_num_workers=10,
        report_to="none",  # Disabling WandB logging
        logging_steps=100,
        fp16=True,  # Enable mixed precision (ensure hardware supports this)
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    )

    # Load tokenizer and model from the cache directory
    tokenizer = T5Tokenizer.from_pretrained(args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.cache_dir)
    fast_tokenizer = T5TokenizerFast.from_pretrained(args.cache_dir)

    # Create dataset
    generate_dataset = GenerateDataset(
        path_to_data=args.input_file,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        tokenizer=tokenizer
    )

    # Initialize the trainer with the custom collator
    trainer = DocTqueryTrainer(
        do_generation=True,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=QueryEvalCollator(
            tokenizer,
            padding='longest',
        ),
    )

    # Check if the output file already exists to avoid duplicating work
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf8') as f:
            existing_docids = set([json.loads(line)['doc_id'] for line in f.readlines()])
    else:
        existing_docids = set()

    processed_docs = 0
    
    with open(args.output_path, 'a', encoding='utf-8') as outfile:
        # Batchify input data and generate queries
        for batch in batchify(generate_dataset, args.batch_size):
            with torch.no_grad():  # Avoid storing gradients
                predict_results = trainer.predict(
                    batch,
                    top_k=args.top_k,
                    num_return_sequences=args.num_return_sequences,
                    max_length=args.q_max_length
                )

            # Writing queries to the output file
            for batch_tokens, batch_ids in zip(predict_results.predictions, predict_results.label_ids):
                for tokens, docid in tqdm(zip(batch_tokens, batch_ids), desc="Writing pseudo-queries to file"):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    if docid.item() not in existing_docids:
                        jitem = json.dumps({'doc_id': docid.item(), 'query': query})
                        outfile.write(jitem + '\n')
            
            # Stop if we reach the max_docs limit
            processed_docs += len(batch)
            if processed_docs >= args.max_docs:
                print(f"Processed {processed_docs} documents. Stopping as per the limit.")
                break

if __name__ == "__main__":
    main()
