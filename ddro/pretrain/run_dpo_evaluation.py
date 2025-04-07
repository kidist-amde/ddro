import os
import time
import torch
import random
import argparse
from utils import load_model, set_seed
from tqdm.auto import tqdm
from trie import Trie
from evaluate_per_query import evaluator
from collections import defaultdict
from torch.utils.data import DataLoader
from pretrain.T5ForPretrain import T5ForPretrain
from utils.pretrain_dataset import PretrainDataForT5
from transformers import T5Tokenizer, T5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--per_gpu_batch_size", default=25, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--output_every_n_step", default=25, type=int)
    parser.add_argument("--save_every_n_epoch", default=25, type=int)
    parser.add_argument("--operation", default="training", type=str)
    parser.add_argument("--use_docid_rank", default="False", type=str)
    parser.add_argument("--load_ckpt", default="False", type=str)
    parser.add_argument("--save_path", default="./model/", type=str)
    parser.add_argument("--log_path", default="./log/", type=str)
    parser.add_argument("--doc_file_path", type=str, required=True)
    parser.add_argument("--docid_path", type=str, required=True)
    parser.add_argument("--train_file_path", type=str)
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--pretrain_model_path", type=str, required=True)
    parser.add_argument("--load_ckpt_path", default="./model/", type=str)
    parser.add_argument("--dataset_script_dir", type=str, required=True)
    parser.add_argument("--dataset_cache_dir", type=str, required=True)
    parser.add_argument("--add_doc_num", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_docid_length", type=int, default=1)
    parser.add_argument("--use_origin_head", default="False", type=str)
    parser.add_argument("--num_beams", default=10, type=int)
    return parser.parse_args()


def load_encoded_docid(docid_path):
    encode_2_docid, encoded_docids = {}, []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            encode_list = [int(x) for x in encode.split(",") if x not in ["0", "1"]]
            encoded_docids.append(encode_list)
            encode_str = ','.join(map(str, encode_list))
            encode_2_docid.setdefault(encode_str, []).append(docid.lower())
    return encoded_docids, encode_2_docid


def docid2string(docid):
    return ",".join(str(x) for x in docid if x not in [0, 1])


def evaluate_beamsearch(args):
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)
    model.load_state_dict(load_model(args.save_path))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    logger = open(args.log_path, "a")
    logger.write(f"\nStart evaluation with args: {args}\n")

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + d for d in encoded_docid])

    def prefix_allowed_tokens_fn(batch_id, sent):
        out = docid_trie.get(sent.tolist())
        return out if out else [tokenizer.pad_token_id]

    test_data = [args.test_file_path]
    dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    dataloader = DataLoader(dataset, batch_size=args.per_gpu_batch_size, shuffle=False, num_workers=4)

    evaluator_ = evaluator()
    truth, prediction = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.cuda() for k, v in batch.items() if k not in ["query_id", "doc_id"]}
        input_ids = batch["input_ids"]

        if args.use_docid_rank == "False":
            labels = batch["docid_labels"].cpu().numpy().tolist()
            truth.extend([[docid2string(x)] for x in labels])
        else:
            truth.extend([[x] for x in batch["query_id"]])

        outputs = model.generate(
            input_ids,
            max_length=args.max_docid_length + 1,
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )

        for i in range(input_ids.size(0)):
            batch_output = outputs[i * args.num_beams:(i + 1) * args.num_beams].cpu().numpy().tolist()
            doc_rank = []
            for docid in batch_output:
                doc_str = docid2string(docid)
                if args.use_docid_rank == "False":
                    doc_rank.append(doc_str)
                else:
                    docids = encode_2_docid.get(doc_str, [])
                    random.shuffle(docids)
                    doc_rank.extend(docids)
            prediction.append(doc_rank)

    results = evaluator_.evaluate_ranking(truth, prediction)
    results.to_csv(args.log_path.replace(".log", ".csv"), index=False)
    logger.write(str(results.mean()) + "\n")
    print("Evaluation complete. Metrics written.")


if __name__ == '__main__':
    args = parse_args()
    if args.operation == "testing":
        evaluate_beamsearch(args)
