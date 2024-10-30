import json
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from trl import DPOTrainer
from datasets import Dataset
import argparse
from typing import Dict, Optional, Union, List
from transformers import PreTrainedModel
import torch.nn as nn
from T5ForPretrain import T5ForPretrainDPO
from transformers import PreTrainedTokenizerBase
from trl.trainer.dpo_config import DPOConfig
import os
from collections import defaultdict
from typing import Callable
import trl
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging configuration
def setup_logging(log_path: str):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%a %b %d %H:%M:%S %Y",
    )
    logging.info("Logging setup complete.")

def log_args(args):
    logging.info(f"Starting a new run with args: {args}")

def load_model(save_path):
    logging.info(f"Loading model from {save_path}")
    save_model = torch.load(save_path)
    new_save_model = {}
    for key in save_model:
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
        else:
            new_key = key
        new_save_model[new_key] = save_model[key]
    
    logging.info(f"Model loaded successfully from {save_path}")
    return new_save_model

class T5DpoTrainer(DPOTrainer):
    
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        
        if not self.is_encoder_decoder:
            return super().tokenize_row(feature,model)
         
        else:
            batch = {}
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            chosen_tokens = chosen
            rejected_tokens =   rejected
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"].to(device)
            batch["rejected_labels"] = rejected_tokens["input_ids"].to(device)
            batch["prompt_input_ids"] = prompt_tokens["input_ids"].to(device)
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"].to(device)

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"]).to(device)
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"]).to(device)
                )

        return batch

def _tokenize_encoder_decoder(
    batch: Dict[str, List[int]],
    tokenizer: PreTrainedTokenizerBase,
    prompt: List[str],
    chosen: List[str],
    rejected: List[str],
    args: DPOConfig,
    ) -> None:

    chosen_tokens_input_ids =   chosen
    rejected_tokens_input_ids = rejected
    chosen_tokens = {"input_ids": chosen_tokens_input_ids,"attention_mask": [1] * len(chosen_tokens_input_ids)}
    rejected_tokens = {"input_ids": rejected_tokens_input_ids,"attention_mask": [1] * len(rejected_tokens_input_ids)}
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True)

    batch["chosen_labels"] = chosen_tokens["input_ids"]
    batch["rejected_labels"] = rejected_tokens["input_ids"]
    batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
def _tokenize(
    features: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    args: DPOConfig,
    processor: Optional[Callable] = None,
    model: Optional[PreTrainedModel] = None,
) -> Dict[str, List]:
    """
    Tokenizes and processes a batch of input features using the provided tokenizer and processor.
    """
    batch = defaultdict(list)
    if model is None:
        return super()._tokenize(features, tokenizer, args, processor, model)
    else:
        _tokenize_encoder_decoder(batch, tokenizer, features["prompt"], features["chosen"], features["rejected"], args)
        return dict(batch)
        
trl.trainer.dpo_trainer._tokenize = _tokenize

def log_training_step(step, learning_rate, loss):
    logging.info(f"Step: {step}, lr: {learning_rate:.10f}, loss: {loss:.10f}")

def create_datasets(docid_path, train_file_path, dev_file_path, num_proc=24):
    def load_line_dataset(path, doc_ids):
        with open(path, "r") as f:
            lines = f.readlines()
            output = []
            for line in lines:
                line = line.strip()
                topic_id, query, p_url, pid, n_url, nid = line.split("\t")
                if pid in doc_ids and nid in doc_ids:
                    output.append({
                        "query": query,
                        "positive_doc_id": doc_ids[pid],
                        "negative_doc_id": doc_ids[nid],
                    })
        return output

    def load_pq_docids(path):
        with open(path, "r") as f:
            lines = f.readlines()
            output = {}
            for line in lines:
                line = line.strip()
                docid, pq_ids = line.split("\t")
                docid = docid.strip("[").strip("]").upper()
                pq_ids = [int(x) for x in pq_ids.split(",")]
                output[docid] = pq_ids
        return output

    def return_prompt_and_responses(samples):
        return {
            "prompt": samples["query"],
            "chosen": samples["positive_doc_id"],   
            "rejected": samples["negative_doc_id"], 
        }

    logging.info("Loading doc IDs and datasets")
    doc_ids = load_pq_docids(docid_path)
    train_data = load_line_dataset(train_file_path, doc_ids)
    dev_data = load_line_dataset(dev_file_path, doc_ids)
    logging.info(f"Size of the train set: {len(train_data)}, Size of the validation set: {len(dev_data)}")

    train_data = Dataset.from_list(train_data)
    dev_data = Dataset.from_list(dev_data)
    original_columns = train_data.column_names
    train_data = train_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    dev_data = dev_data.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    logging.info("Datasets created successfully")
    return train_data, dev_data

def load_model_from_checkpoint(cli_args):
    logging.info("Loading model from checkpoint")
    pretrain_model = T5ForConditionalGeneration.from_pretrained(cli_args.pretrain_model_path, device_map="auto")
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + 6144)
    model = T5ForPretrainDPO(pretrain_model, cli_args)
    state_dict = load_model(cli_args.checkpoint_path)
    model.load_state_dict(state_dict)
    logging.info("Model loaded successfully from checkpoint")
    return model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a DPO model.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--pretrain_model_path", type=str, help="Path of the pretrained model checkpoint")
    parser.add_argument("--use_origin_head", default="False", type=str, help="Whether to load the lm_head from the pretrained model.")
    parser.add_argument("--checkpoint_path", default="WebUltron/outputs/t5_128_1_top_300k_pq_pretrain_search_finetune/model_9.pkl", type=str, help="Path of the checkpoint to load")
    parser.add_argument("--log_path", type=str, default="ddro/logs-nq-dpo/dpo.log", help="Path to the log file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    cli_args = parser.parse_args()

    # Setup logging
    setup_logging(cli_args.log_path)
    log_args(cli_args)

    # Load tokenizer and datasets
    tokenizer = T5Tokenizer.from_pretrained(cli_args.pretrain_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, dev_dataset = create_datasets(cli_args.docid_path, cli_args.train_file, cli_args.dev_file)

    # Configure training arguments
    training_args = DPOConfig(
        output_dir=cli_args.output_dir,
        per_device_train_batch_size=64,
        save_total_limit=2,
        save_steps=1000,
        eval_steps=500,
        num_train_epochs=7,
        learning_rate=5e-6,
        warmup_steps=500,
        max_grad_norm=1.0,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
    )

    # Load models
    model = load_model_from_checkpoint(cli_args).to(device)
    model_ref = load_model_from_checkpoint(cli_args).to(device)
    logging.info("Models loaded successfully")

    # Set up T5DpoTrainer
    dpo_trainer = T5DpoTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.49,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        is_encoder_decoder=True,
    )

    # Train the model
    logging.info("Starting model training")
    for epoch in range(1, cli_args.epochs + 1):
        logging.info(f"Epoch {epoch}/{cli_args.epochs}")
        for step, outputs in enumerate(dpo_trainer.train()):
            loss = outputs.loss.item()
            learning_rate = dpo_trainer.args.learning_rate
            log_training_step(step, learning_rate, loss)
    logging.info("Model training complete")

    # Save the final model
    dpo_trainer.save_model()
    torch.save(model.state_dict(), os.path.join(cli_args.output_dir, f"dpo_model_final.pkl"))
    logging.info("Model saved successfully")

if __name__ == "__main__":
    main()
 