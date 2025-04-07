#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import os
import torch
from collections import Counter
from transformers import BertTokenizer, BertModel


def set_seed(seed=0):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, tokenizer, output_dir):
    """
    Save model and tokenizer to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.bert_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model(save_path):
    """
    Load model weights from a .bin file, adjusting keys if trained with DataParallel.
    """
    save_model = torch.load(save_path)
    new_save_model = {}

    for key in save_model:
        new_key = ".".join(key.split(".")[1:]) if "module" in key else key
        new_save_model[new_key] = save_model[key]

    print("Loaded model from", save_path)
    return new_save_model


def initialize_docid_embed(bert_model_path, doc_file_path, bert_embedding, docs_embed, store_path):
    """
    Initialize document ID embeddings by averaging token embeddings for each document.

    Args:
        bert_model_path (str): Path to pre-trained BERT model.
        doc_file_path (str): Path to the input JSONL document file.
        bert_embedding (np.ndarray): BERT embedding matrix [vocab_size, embed_size].
        docs_embed (np.ndarray): Placeholder for document embeddings.
        store_path (str): Path to save the final document embeddings.

    Returns:
        np.ndarray: Updated document embeddings.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='Initializing docid embeddings'):
            data = json.loads(line)
            token_ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(data["title"] + ' ' + data["url"])
            )
            token_count = Counter(token_ids)
            embed = docs_embed[i]

            for token_id, count in token_count.items():
                embed += bert_embedding[token_id]

            embed /= (len(token_count) + 1)
            docs_embed[i] = embed

    with open(store_path, "wb") as fw:
        pickle.dump(docs_embed, fw)

    return docs_embed
