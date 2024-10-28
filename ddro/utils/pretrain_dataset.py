import datasets
import torch
from torch.utils.data import Dataset
import numpy as np

class PretrainDataForT5(Dataset):
    def __init__(self, filename, max_seq_length, max_docid_length, tokenizer, dataset_script_dir, dataset_cache_dir, args=None):
        self.args = args
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._max_docid_length = max_docid_length
        self._tokenizer = tokenizer
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json_builder.py',
            data_files = self._filename,
            # ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            trust_remote_code=True,  # Add this argument to trust the remote code
            features=datasets.Features({
                'query_id': datasets.Value("string"),
                'doc_id': datasets.Value("string"),
                'input_ids': [datasets.Value("int32")],
            })
        )['train']
        self.total_len = len(self.nlp_dataset)  
      
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        data = self.add_padding(data)
        return {
            "query_id": data['query_id'],
            "input_ids": np.array(data['input_ids']),
            "docid_labels": np.array(data['docid_labels']),
            "attention_mask": np.array(data['attention_mask']),
        }

    def add_padding(self, training_instance):
        padded_input_ids = [0 for i in range(self._max_seq_length)]
        padded_attention_mask = [0 for i in range(self._max_seq_length)]
        padded_docid_labels = [0 for i in range(self._max_docid_length)]
        
        input_ids = training_instance["input_ids"][:self._max_seq_length]

        for i, iid in enumerate(input_ids):
            padded_input_ids[i] = iid
            padded_attention_mask[i] = 1

        encoded_docid = [int(x) for x in training_instance["doc_id"].split(",")][:self._max_docid_length]
        padded_docid_labels[:len(encoded_docid)] = encoded_docid

        new_instance = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "query_id": training_instance["query_id"],
            "docid_labels": padded_docid_labels,
        }
        return new_instance