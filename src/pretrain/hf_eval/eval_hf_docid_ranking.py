from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Ensure `src/` is importable when called from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.evaluate import evaluator  # type: ignore
from utils.pretrain_dataset import PretrainDataForT5  # type: ignore
from utils.trie import Trie  # type: ignore


# ----------------------------
# Argparse
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--per_gpu_batch_size", type=int, default=4)
    p.add_argument("--log_path", type=str, required=True)
    p.add_argument("--docid_path", type=str, required=True)
    p.add_argument("--test_file_path", type=str, required=True)
    p.add_argument("--pretrain_model_path", type=str, default="kiyam/ddro-msmarco-tu")
    p.add_argument("--dataset_script_dir", type=str, required=True)
    p.add_argument("--dataset_cache_dir", type=str, default="./cache")
    p.add_argument("--add_doc_num", type=int, required=True)
    p.add_argument("--max_seq_length", type=int, default=64)
    p.add_argument("--max_docid_length", type=int, default=100)
    p.add_argument("--num_beams", type=int, default=15)
    p.add_argument("--use_docid_rank", type=str, default="True")  # keep parity
    p.add_argument(
        "--docid_format",
        type=str,
        default="msmarco",
        choices=["msmarco", "nq"],
        help="Controls EOS/pad handling for docid encodings.",
    )
    p.add_argument(
        "--lookup_fallback",
        type=str,
        default="True",
        help="If True, attempt alternate key shapes on miss (with/without EOS).",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    return p


# ----------------------------
# Data loading helpers
# ----------------------------

def load_data_paths(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    return [os.path.join(path, fn) for fn in os.listdir(path)]


def load_encoded_docid_msmarco(docid_path: str) -> Tuple[List[List[int]], Dict[str, List[str]]]:
    """MSMARCO: keep full sequence including EOS=1 for keys."""
    encode_2_docid: Dict[str, List[str]] = {}
    encoded_docids: List[List[int]] = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("	")
            key_list = [int(x) for x in encode.split(",")]
            encoded_docids.append(key_list)
            key = ",".join(map(str, key_list))
            encode_2_docid.setdefault(key, []).append(docid.lower())
    return encoded_docids, encode_2_docid


def load_encoded_docid_nq(docid_path: str) -> Tuple[List[List[int]], Dict[str, List[str]]]:
    """NQ: strip 0/1 when building keys (historical convention)."""
    encode_2_docid: Dict[str, List[str]] = {}
    encoded_docids: List[List[int]] = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("	")
            encode_list = [int(x) for x in encode.split(",")]
            filtered = [x for x in encode_list if x not in (0, 1)]
            encoded_docids.append(filtered)
            key = ",".join(map(str, filtered))
            encode_2_docid.setdefault(key, []).append(docid.lower())
    return encoded_docids, encode_2_docid


def make_docid_trie(encoded_docids: Iterable[List[int]]) -> Trie:
    # Prefix with 0 as BOS for constrained decoding; mirrors training.
    return Trie([[0] + ids for ids in encoded_docids])


# ----------------------------
# Decoding utilities
# ----------------------------

def docid2string_msmarco(ids: List[int]) -> str:
    """Drop 0s, keep a single EOS=1, trim after first 1."""
    seq: List[int] = []
    for x in ids:
        if x == 0:
            continue
        if x == 1:
            seq.append(1)
            break
        seq.append(x)
    return ",".join(map(str, seq))


def docid2string_nq(ids: List[int]) -> str:
    """Strip 0 and 1 entirely for NQ to match keys built without EOS/pad."""
    return ",".join(str(x) for x in ids if x not in (0, 1))


def safe_lookup(
    key: str,
    table: Dict[str, List[str]],
    *,
    fallback: bool,
    alt_key: str | None = None,
) -> List[str]:
    if key in table:
        return table[key]
    if fallback and alt_key is not None and alt_key in table:
        return alt_key and table[alt_key] or table[key]
    raise KeyError(key)


# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Resolve flags and device
    device = torch.device(args.device)
    lookup_fallback = (args.lookup_fallback.lower() == "true")
    use_docid_rank = (args.use_docid_rank.lower() == "true")

    # Log file (append). Create parent dir if needed.
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logger = open(args.log_path, "a")
    logger.write("")
    logger.write(f"start a new running with args: {args}")

    # Tokenizer/model
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    # Keep embeddings aligned with tokenizer; avoid blind resizing.
    current = model.get_input_embeddings().num_embeddings
    target = len(tokenizer)
    if target != current:
        model.resize_token_embeddings(target)
    model.to(device).eval()

    # Encoded docids + Trie
    if args.docid_format == "msmarco":
        encoded_docids, encode_2_docid = load_encoded_docid_msmarco(args.docid_path)
        to_str = docid2string_msmarco
    else:
        encoded_docids, encode_2_docid = load_encoded_docid_nq(args.docid_path)
        to_str = docid2string_nq
    docid_trie = make_docid_trie(encoded_docids)

    def prefix_allowed_tokens_fn(batch_id, sent_ids):
        # Using Trie limits to valid docid prefixes; keeps decoding on-manifold.
        nxt = docid_trie.get(sent_ids.tolist())
        return nxt if nxt else [tokenizer.pad_token_id]

    # Data
    if not os.path.exists(args.test_file_path):
        raise FileNotFoundError(args.test_file_path)
    test_paths = load_data_paths(args.test_file_path)
    test_dataset = PretrainDataForT5(
        test_paths,
        args.max_seq_length,
        args.max_docid_length,
        tokenizer,
        args.dataset_script_dir,
        args.dataset_cache_dir,
        args,
    )
    batch_size = args.per_gpu_batch_size * max(1, torch.cuda.device_count())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Eval
    truth: List[List[str]] = []
    prediction: List[List[str]] = []

    print("Start evaluating...")
    for _, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        with torch.no_grad():
            for k in list(batch.keys()):
                if k not in ("query_id", "doc_id"):
                    batch[k] = batch[k].to(device)

        input_ids = batch["input_ids"]
        if not use_docid_rank:
            labels = batch["docid_labels"]
            truth.extend([[to_str(ids)] for ids in labels.cpu().numpy().tolist()])
        else:
            labels = batch["query_id"]
            truth.extend([[qid] for qid in labels])

        outputs = model.generate(
            input_ids,
            max_length=args.max_docid_length + 1,
            num_return_sequences=args.num_beams,
            num_beams=args.num_beams,
            do_sample=False,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        # Collect N-best per input
        for j in range(input_ids.shape[0]):
            doc_rank: List[str] = []
            batch_out = outputs[j * args.num_beams : (j + 1) * args.num_beams].cpu().numpy().tolist()
            for out_ids in batch_out:
                if not use_docid_rank:
                    doc_rank.append(to_str(out_ids))
                else:
                    key_primary = to_str(out_ids)
                    # Build alternate shape once (with vs without EOS) for fallback lookups.
                    if args.docid_format == "msmarco":
                        alt = key_primary[:-2] if key_primary.endswith(",1") else key_primary + ",1"
                    else:
                        alt = key_primary + ",1" if key_primary and not key_primary.endswith(",1") else key_primary[:-2]

                    docids = safe_lookup(key_primary, encode_2_docid, fallback=lookup_fallback, alt_key=alt)
                    # Shuffle only to break ties when many docids share encoding.
                    random.shuffle(docids)
                    doc_rank.extend(docids)
            prediction.append(doc_rank)

    # Metrics
    ev = evaluator()
    result_df = ev.evaluate_ranking(truth, prediction)
    metrics = {
        "MRR@10": result_df["MRR@10"].values.mean(),
        "MRR": result_df["MRR"].values.mean(),
        "P@1": result_df["P@1"].values.mean(),
        "P@10": result_df["P@10"].values.mean(),
        "P@20": result_df["P@20"].values.mean(),
        "P@100": result_df["P@100"].values.mean(),
        "R@1": result_df["R@1"].values.mean(),
        "R@10": result_df["R@10"].values.mean(),
        "R@100": result_df["R@100"].values.mean(),
        "R@1000": result_df["R@1000"].values.mean(),
        "Hit@1": result_df["Hit@1"].values.mean(),
        "Hit@5": result_df["Hit@5"].values.mean(),
        "Hit@10": result_df["Hit@10"].values.mean(),
        "Hit@100": result_df["Hit@100"].values.mean(),
    }

    msg = ", ".join(f"{k}:{v}" for k, v in metrics.items())
    print(msg)
    logger.write(msg + "")
    csv_path = args.log_path.replace(".log", ".csv")
    result_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
