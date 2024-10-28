import json
import pickle
import numpy as np
from tqdm import tqdm
import random, os, torch
from collections import Counter
from transformers import BertTokenizer, BertModel

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    # torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    model_to_save.bert_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model(save_path):
    save_model = torch.load(save_path)
    new_save_model = {}
    for key in save_model:
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
        else:
            new_key = key
        new_save_model[new_key] = save_model[key]
    
    print("load model from ", save_path)
    return new_save_model

# 生成每个doc中token的情况，以便用于初始化docid的表示向量。表示向量直接取均值，也可以计算加权平均或者tf-idf。
def initialize_docid_embed(bert_model_path, doc_file_path, bert_embedding, docs_embed, store_path):
    """
        args:
            bert_embedding: [vocab_size, embed_size], bert最底层的embedding矩阵
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    start_doc_id = tokenizer.vocab_size
    vocab_size, embed_dim = bert_embedding.shape
    doc_tokens = {}

    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin), desc='constructing all_documents list'):
            data = json.loads(line)
            title = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data["title"] + ' ' + data["url"]))
            token_count = Counter(title)
            embed = docs_embed[i]
            for token_id, count in token_count.items():
                embed += bert_embedding[token_id]
            embed /= (len(token_count)+1)
            docs_embed[i] = embed
    with open(store_path, "wb") as fw:
        pickle.dump(docs_embed, fw)
    return docs_embed