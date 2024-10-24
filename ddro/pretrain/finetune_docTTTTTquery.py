from transformers import T5ForConditionalGeneration, T5TokenizerFast, Trainer, TrainingArguments
import pandas as pd
import gzip
import argparse
import torch
from datasets import Dataset

''' This script fine-tunes doc2query-t5-large-msmarco on the Natural Questions dataset using query-document pairs.
It saves the extracted document IDs for easier tracking while passing only query and doc_tac for model training.
'''

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="nq_merged.tsv.gz", type=str, help="Path to the NQ data file")
parser.add_argument("--output_file", default="qcontent_train_512.tsv.gz", type=str, help="Path to save the extracted content")
parser.add_argument("--cache_dir", default="cache/", type=str, help="Cache directory for models")
args = parser.parse_args()

def extract_query_document_pairs(dataset_path, output_file):
    print(f"Loading data from {dataset_path}...")
    with gzip.open(dataset_path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', header=None, 
                         names=['query', 'id', 'long_answer', 'short_answer', 
                                'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])
    
    print(f"Loaded {len(df)} documents!")

    # Extract id, query, and doc_tac
    df_extracted = df[['id', 'query', 'doc_tac']].copy()

    # Fill missing values in 'doc_tac' with empty strings
    df_extracted['doc_tac'] = df_extracted['doc_tac'].fillna('')

    # Save the id, query, and doc_tac as a compressed TSV
    df_extracted.to_csv(output_file, sep='\t', index=False, header=False, compression='gzip')

    print(f"Query-document pairs (with IDs) saved to {output_file}.")

def prepare_dataset(data_file, tokenizer, max_length=512):
    # Load the compressed TSV file
    df = pd.read_csv(data_file, sep='\t', names=['id', 'query', 'doc_tac'], compression='gzip')

    # Ensure doc_tac content is a string and handle missing values
    df['doc_tac'] = df['doc_tac'].fillna('').astype(str)

    # Define the tokenization function
    def tokenize_function(examples):
        # Tokenize the document (input) and query (target)
        inputs = tokenizer(examples['doc_tac'], max_length=max_length, truncation=True, padding="max_length")
        targets = tokenizer(examples['query'], max_length=max_length, truncation=True, padding="max_length")
        inputs['labels'] = targets['input_ids']
        return inputs

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['query', 'doc_tac']])  # We pass only query and doc_tac for training

    # Apply the tokenization
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set format to return PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset

def split_dataset(dataset, split_ratio=0.2):
    # Use the Hugging Face `train_test_split` method for splitting
    split_data = dataset.train_test_split(test_size=split_ratio)
    train_dataset = split_data['train']
    eval_dataset = split_data['test']
    
    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Step 1: Extract query-document pairs from nq_merged.tsv.gz (including document IDs for later tracking)
    # extract_query_document_pairs(args.dataset_path, args.output_file)

    # Step 2: Load pre-trained doc2query-t5-large-msmarco model and tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("castorini/doc2query-t5-large-msmarco", cache_dir=args.cache_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("castorini/doc2query-t5-large-msmarco", cache_dir=args.cache_dir)

    # Step 3: Prepare the dataset for training (passing only query and doc_tac)
    full_dataset = prepare_dataset(args.output_file, tokenizer)

    # Step 4: Split the dataset into training and evaluation sets (80/20 split)
    train_dataset, eval_dataset = split_dataset(full_dataset, split_ratio=0.2)

    # Step 5: Define training arguments
    training_args = TrainingArguments(
        output_dir="resources/transformer_models",              # Directory to save the model
        evaluation_strategy="steps",          # Evaluate during training
        per_device_train_batch_size=2,        # Adjust based on your hardware
        per_device_eval_batch_size=2,
        learning_rate=3e-4,
        num_train_epochs=3,                   # Adjust for your needs
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,                       # Adjust based on the dataset size
        logging_dir="logs",                   # Directory to save logs
        logging_steps=100,
        load_best_model_at_end=True,          # Load the best model at the end of training
        report_to="none"                      # Avoid logging to WandB or Hugging Face
    )

    # Step 6: Create Trainer instance with evaluation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=eval_dataset     # Evaluation dataset
    )

    # Step 7: Fine-tune the model
    trainer.train()

    # Step 8: Save the fine-tuned model
    trainer.save_model("resources/transformer_models/finetuned_doc2query_t5_large_msmarco")
    print(f"Model fine-tuned and saved to resources/transformer_models/finetuned_doc2query_t5_large_msmarco")
