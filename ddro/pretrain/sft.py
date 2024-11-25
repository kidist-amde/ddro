import json
from torch.utils.data import Dataset 
import numpy as np
from transformers import T5Tokenizer,T5ForConditionalGeneration
import IPython
import gzip , csv
import pandas as pd 
import torch
import argparse
from transformers import TrainingArguments
import warnings
from transformers import Trainer, TrainingArguments


def get_url(docid, docoffset, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """
    
    f.seek(docoffset[docid])
    line = f.readline()
    _docid, url, title, body= line.split("\t")
    _docid = _docid.replace("D","")
    docid = docid.replace("D","")
    assert _docid == docid, f"Looking for {docid}, found {line}"
    return url

def load_pseudoquery_line_dataset(path, msmarco_file_path, docoffset):
     with open(path,"r") as f ,gzip.open(msmarco_file_path, "rt", encoding="utf8") as msmarco_file:
        lines = f.readlines()
        output = []
        for line in lines:
            linedata = json.loads(line)
            doc_id = "D"+str(linedata["doc_id"])
            try:
                doc_url = get_url(doc_id, docoffset, msmarco_file)
            except:
                doc_url = get_url(doc_id[1:], docoffset, msmarco_file)
            
            output.append( { 
                "query_text":linedata["query"],
                "positive_doc_url":doc_url,
                })
        return output
     
class SFTDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, pseudoquery_dataset = None):
        super().__init__()
        self.dataset = []
        self.tokenizer = tokenizer
        with gzip.open(dataset_path, "rt") as f:
            for line in f.readlines():
                topicid, query_text, posurl, positive_docid, negurl, negative_docid = line.split("\t")
                self.dataset.append({"query_text":query_text,"positive_doc_url":posurl})
        if pseudoquery_dataset is not None:
            self.dataset.extend(pseudoquery_dataset)
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        record = self.dataset[index]
        model_inputs = self.tokenizer(record["query_text"],
                                      return_tensors="pt",padding = "max_length",
                                      truncation = True, max_length = 128)
        labels = self.tokenizer(
            record["positive_doc_url"], padding = "max_length",
                                      truncation = True, max_length = 128, return_tensors="pt"
        ).input_ids
        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs['labels'] = labels
        for key in model_inputs:
            model_inputs[key] = torch.squeeze(model_inputs[key])
        return model_inputs 

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a T5 model using SFT.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to the development dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints and logs")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to the checkpoint to resume training from")
    parser.add_argument("--msmarco_file_path", type=str, default="../data/raw/msmarco-docs.tsv", help="Path to the MSMARCO file")
    parser.add_argument("--pseudoquery_file_path", type=str, default="../data/processed/pseudoqueries.jsonl", help="Path to the pseudoquery file")
    parser.add_argument("--docoffset_file_path", type=str, default="../data/raw/msmarco-docs-lookup.tsv.gz", help="Path to the docoffset file")


    args = parser.parse_args()

    docoffset = {}
    with gzip.open(args.docoffset_file_path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            if docid.startswith("D"):
                docid = docid[1:]
            docoffset[docid] = int(offset)

    pseudoquery_dataset = load_pseudoquery_line_dataset(args.pseudoquery_file_path, args.msmarco_file_path, docoffset)
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    
    # Dataset paths
    train_dataset = SFTDataset(args.train_file, tokenizer, pseudoquery_dataset)
    dev_dataset = SFTDataset(args.dev_file, tokenizer)

  # Output name for training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        per_device_train_batch_size=64, # number of training examples per forward/backward pass
        evaluation_strategy="steps", # evaluate every eval_steps
        eval_steps=3000, # evaluate every 3000 steps
        save_total_limit=2, # only keep the last 2 checkpoints
        save_steps=3000, # save a checkpoint every 3000 steps
        num_train_epochs=10,
        learning_rate=1e-6,
        load_best_model_at_end=True
    )
  
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    # Resume from checkpoint if provided
    trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    print(trainer.evaluate())
    # Save the model
    trainer.save_model()

   
    
if __name__ == "__main__":
    main()