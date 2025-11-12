#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import glob
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, __version__ as HF_VER

# Why: allow pulling inputs from HF without refactoring dataset code.
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception as e:  # pragma: no cover
    hf_hub_download = None
    snapshot_download = None

# Ensure `src/` is importable when called from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.evaluate import evaluator  # type: ignore
from utils.pretrain_dataset import PretrainDataForT5  # type: ignore
from utils.trie import Trie  # type: ignore


# ----------------------------
# Argparse
# ----------------------------

def str2bool(x: str) -> bool:
    return x.lower() in {"1", "true", "t", "yes", "y"}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--per_gpu_batch_size", type=int, default=4)
    p.add_argument("--log_path", type=str, required=True)
    p.add_argument("--docid_path", type=str, required=True, help="Local path or hf:dataset:<repo_id>:<file>")
    p.add_argument("--test_file_path", type=str, required=True, help="File/dir or hf:dataset:<repo_id>:<file|dir|glob>")
    p.add_argument("--pretrain_model_path", type=str, default="kiyam/ddro-msmarco-pq")
    p.add_argument("--dataset_script_dir", type=str, required=True)
    p.add_argument("--dataset_cache_dir", type=str, default="./cache")
    p.add_argument("--add_doc_num", type=int, required=True)
    p.add_argument("--max_seq_length", type=int, default=64)
    p.add_argument("--max_docid_length", type=int, default=100)
    p.add_argument("--num_beams", type=int, default=15)
    p.add_argument("--use_docid_rank", type=str2bool, default=True)  # True=ranking by docids, False=exact matching
    p.add_argument("--docid_format", type=str, default="msmarco", choices=["msmarco", "nq"])
    p.add_argument("--lookup_fallback", type=str2bool, default=True, help="Try alt key shape on miss (with/without EOS).")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--assert_strict", type=str2bool, default=True, help="Assert vocab/head/allowed id alignment.")
    return p


# ----------------------------
# HF path resolver
# ----------------------------

def _is_hf_uri(x: str) -> bool:
    return isinstance(x, str) and x.startswith("hf:")

def _parse_hf_uri(uri: str) -> tuple[str, str, str]:
    """
    Syntax:
      hf:dataset:<repo_id>:<path_or_glob>
      hf:model:<repo_id>:<path_or_glob>
    Defaults to dataset if segment[1] != 'model'.
    """
    parts = uri.split(":", 3)
    if len(parts) < 3:
        raise ValueError(f"Bad HF URI: {uri}")
    if len(parts) == 3:
        repo_type, repo_id, path = "dataset", parts[1], ""
    else:
        _, maybe_type, repo_id, path = parts
        repo_type = "model" if maybe_type == "model" else "dataset"
    if not repo_id:
        raise ValueError(f"HF URI missing repo id: {uri}")
    return repo_type, repo_id, path

def hf_resolve(uri: str) -> List[str]:
    """Return local file path(s) for an HF-backed uri."""
    if hf_hub_download is None or snapshot_download is None:
        raise RuntimeError("huggingface_hub is required but not installed.")
    repo_type, repo_id, subpath = _parse_hf_uri(uri)

    # Glob/dir support via snapshot
    if not subpath or subpath.endswith("/") or ("*" in subpath) or ("?" in subpath) or ("[" in subpath):
        allow = None
        if subpath and not subpath.endswith("/"):
            allow = subpath  # allow_patterns supports glob
        repo_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=allow)
        base = os.path.join(repo_dir, os.path.dirname(subpath)) if subpath else repo_dir
        pattern = os.path.join(repo_dir, subpath) if subpath else os.path.join(repo_dir, "*")
        files = sorted([p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)])
        if not files:
            raise FileNotFoundError(f"No files matched HF pattern: {uri} (resolved {pattern})")
        return files

    # Single file
    fp = hf_hub_download(repo_id=repo_id, filename=subpath, repo_type=repo_type)
    return [fp]


# ----------------------------
# Logging
# ----------------------------

def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("eval_pq")
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ----------------------------
# Data loading helpers
# ----------------------------

def load_data_paths(path: str) -> List[str]:
    # Why: keep backward compatibility; add HF support.
    if _is_hf_uri(path):
        return hf_resolve(path)
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = [os.path.join(path, fn) for fn in os.listdir(path) if not fn.startswith(".")]
        files.sort()
        return files
    raise FileNotFoundError(path)

def load_encoded_docid_msmarco(docid_path: str) -> Tuple[List[List[int]], Dict[str, List[str]]]:
    """MS MARCO: keep full sequence including EOS=1 for keys."""
    encode_2_docid: Dict[str, List[str]] = {}
    encoded_docids: List[List[int]] = []
    with open(docid_path, "r", encoding="utf-8") as fr:
        for line in fr:
            docid, encode = line.rstrip("\n").split("\t")
            key_list = [int(x) for x in encode.split(",")]
            encoded_docids.append(key_list)
            key = ",".join(map(str, key_list))
            encode_2_docid.setdefault(key, []).append(docid.lower())
    return encoded_docids, encode_2_docid

def load_encoded_docid_nq(docid_path: str) -> Tuple[List[List[int]], Dict[str, List[str]]]:
    """NQ: strip 0/1 when building keys (historical convention)."""
    encode_2_docid: Dict[str, List[str]] = {}
    encoded_docids: List[List[int]] = []
    with open(docid_path, "r", encoding="utf-8") as fr:
        for line in fr:
            docid, encode = line.rstrip("\n").split("\t")
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
    if fallback and alt_key and alt_key in table:
        return table[alt_key]
    return []


# ----------------------------
# Tokenizer loading with fallback
# ----------------------------

def load_tokenizer_with_fallback(model_path: str, logger: logging.Logger):
    """
    Load T5 tokenizer with multiple fallback strategies to handle
    non-consecutive added token issues.
    """
    # Strategy 1: Try T5TokenizerFast (more lenient with added tokens)
    try:
        logger.info("Attempting to load T5TokenizerFast from %s", model_path)
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        logger.info("Successfully loaded T5TokenizerFast")
        return tokenizer
    except Exception as e:
        logger.warning("T5TokenizerFast failed: %s", str(e))
    
    # Strategy 2: Try T5Tokenizer with legacy=False
    try:
        logger.info("Attempting to load T5Tokenizer with legacy=False from %s", model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        logger.info("Successfully loaded T5Tokenizer with legacy=False")
        return tokenizer
    except ValueError as e:
        logger.warning("T5Tokenizer with legacy=False failed: %s", str(e))
    
    # Strategy 3: Try T5Tokenizer with legacy=True
    try:
        logger.info("Attempting to load T5Tokenizer with legacy=True from %s", model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=True)
        logger.info("Successfully loaded T5Tokenizer with legacy=True")
        return tokenizer
    except Exception as e:
        logger.error("T5Tokenizer with legacy=True failed: %s", str(e))
    
    # Strategy 4: Load base tokenizer and manually add tokens
    try:
        logger.info("All standard methods failed. Attempting base T5 tokenizer approach...")
        from transformers import AutoTokenizer
        
        # Try to load as AutoTokenizer which may handle edge cases better
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logger.info("Successfully loaded tokenizer using AutoTokenizer fallback")
        return tokenizer
    except Exception as e:
        logger.error("AutoTokenizer fallback failed: %s", str(e))
        raise RuntimeError(f"All tokenizer loading strategies failed for {model_path}")


# ----------------------------
# Main eval
# ----------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Logging & env info
    logger = setup_logging(args.log_path)
    logger.info("Args: %s", json.dumps(vars(args), indent=2, sort_keys=True))
    logger.info("Torch: %s | CUDA avail: %s | GPUs: %s", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    logger.info("Transformers: %s", HF_VER)

    # Determinism
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Requested device %s but CUDA not available. Falling back to CPU.", args.device)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ---- HF resolution for inputs ----
    if _is_hf_uri(args.docid_path):
        resolved = hf_resolve(args.docid_path)
        if len(resolved) != 1:
            raise ValueError(f"--docid_path must resolve to a single file, got {len(resolved)}")
        args.docid_path = resolved[0]
    if _is_hf_uri(args.test_file_path):
        resolved_files = hf_resolve(args.test_file_path)
        test_paths = resolved_files
    else:
        test_paths = load_data_paths(args.test_file_path)

    # Tokenizer loading with robust fallback mechanism
    tokenizer = load_tokenizer_with_fallback(args.pretrain_model_path, logger)
    
    # Load model
    logger.info("Loading model from %s", args.pretrain_model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    model.to(device).eval()
    logger.info("Model loaded successfully")

    # Keep embeddings aligned with tokenizer
    current = model.get_input_embeddings().num_embeddings
    target = len(tokenizer)
    if target != current:
        logger.info("Resizing token embeddings: %d -> %d", current, target)
        model.resize_token_embeddings(target)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # Encoded docids + Trie
    if args.docid_format == "msmarco":
        encoded_docids, encode_2_docid = load_encoded_docid_msmarco(args.docid_path)
        to_str = docid2string_msmarco
    else:
        encoded_docids, encode_2_docid = load_encoded_docid_nq(args.docid_path)
        to_str = docid2string_nq
    docid_trie = make_docid_trie(encoded_docids)

    # ----- SANITY PRINTS / ASSERTS -----
    EXPECTED_BASE = 32128  # base T5 vocab
    expected_target = EXPECTED_BASE + int(args.add_doc_num)

    allowed_vocab = {0}
    for ids in encoded_docids:
        allowed_vocab.update(ids)
    max_allowed_id = max(allowed_vocab) if allowed_vocab else -1

    sanity = {
        "model.config.vocab_size": int(model.config.vocab_size),
        "tokenizer_size": int(len(tokenizer)),
        "input_embeddings.num_embeddings": int(model.get_input_embeddings().num_embeddings),
        "lm_head.shape": tuple(model.lm_head.weight.shape),
        "expected_target_vocab": int(expected_target),
        "max_allowed_id_from_docids": int(max_allowed_id),
    }
    logger.info("SANITY: %s", json.dumps(sanity, indent=2, sort_keys=True))
    print("=== SANITY CHECKS ===")
    for k, v in sanity.items():
        print(f"{k}: {v}")

    if args.assert_strict:
        assert model.get_input_embeddings().num_embeddings == len(tokenizer), "embeddings != tokenizer size"
        assert model.lm_head.weight.shape[0] == len(tokenizer), "lm_head rows != tokenizer size"
        if len(tokenizer) != expected_target:
            logger.warning("Tokenizer size (%d) != expected_target_vocab (%d). If this is PQ, check add_doc_num.",
                           len(tokenizer), expected_target)
        assert max_allowed_id < len(tokenizer), "allowed docid token id >= vocab size"
    # -----------------------------------

    # prefix constraint
    def prefix_allowed_tokens_fn(batch_id, sent_ids):
        nxt = docid_trie.get(sent_ids.tolist())
        return nxt if nxt else [tokenizer.pad_token_id]

    # Data
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
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    logger.info("Start evaluating...")
    print("Start evaluating...")

    # Eval
    truth: List[List[str]] = []
    prediction: List[List[str]] = []

    for _, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        with torch.no_grad():
            for k in list(batch.keys()):
                if k not in ("query_id", "doc_id"):
                    batch[k] = batch[k].to(device)

        input_ids = batch["input_ids"]
        if not args.use_docid_rank:
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
            batch_out = outputs[j * args.num_beams: (j + 1) * args.num_beams].cpu().numpy().tolist()
            for out_ids in batch_out:
                if not args.use_docid_rank:
                    doc_rank.append(to_str(out_ids))
                else:
                    key_primary = to_str(out_ids)
                    if args.docid_format == "msmarco":
                        alt = key_primary[:-2] if key_primary.endswith(",1") else (key_primary + ",1" if key_primary else "1")
                    else:
                        alt = key_primary + ",1" if key_primary and not key_primary.endswith(",1") else key_primary[:-2]

                    docids = safe_lookup(key_primary, encode_2_docid, fallback=args.lookup_fallback, alt_key=alt)
                    if docids:
                        random.shuffle(docids)
                        doc_rank.extend(docids)
            prediction.append(doc_rank)

    # Metrics
    ev = evaluator()
    result_df = ev.evaluate_ranking(truth, prediction)
    metrics = {
        "MRR@10": float(result_df["MRR@10"].values.mean()),
        "MRR": float(result_df["MRR"].values.mean()),
        "P@1": float(result_df["P@1"].values.mean()),
        "P@10": float(result_df["P@10"].values.mean()),
        "P@20": float(result_df["P@20"].values.mean()),
        "P@100": float(result_df["P@100"].values.mean()),
        "R@1": float(result_df["R@1"].values.mean()),
        "R@5": float(result_df["R@5"].values.mean()),  
        "R@10": float(result_df["R@10"].values.mean()),
        "R@100": float(result_df["R@100"].values.mean()),
        "R@1000": float(result_df["R@1000"].values.mean()),
        "Hit@1": float(result_df["Hit@1"].values.mean()),
        "Hit@5": float(result_df["Hit@5"].values.mean()),
        "Hit@10": float(result_df["Hit@10"].values.mean()),
        "Hit@100": float(result_df["Hit@100"].values.mean()),
        "Hit@1000": float(result_df["Hit@1000"].values.mean()),
    }

    # Pretty block print of final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS".center(80))
    print("=" * 80)

    print(f"MRR@10: {metrics['MRR@10']:.4f} | MRR: {metrics['MRR']:.4f}")
    print(f"P@1: {metrics['P@1']:.4f} | P@10: {metrics['P@10']:.4f} | P@20: {metrics['P@20']:.4f}")
    print(f"R@10: {metrics['R@10']:.4f} | R@100: {metrics['R@100']:.4f} | R@1000: {metrics['R@1000']:.4f}")
    print(f"R@1: {metrics['R@1']:.4f} | R@5: {metrics['R@5']:.4f} | R@10: {metrics['R@10']:.4f} | R@100: {metrics['R@100']:.4f} | R@1000: {metrics['R@1000']:.4f}")
    print(f"Hit@1: {metrics['Hit@1']:.4f} | Hit@10: {metrics['Hit@10']:.4f} | Hit@100: {metrics['Hit@100']:.4f} | Hit@1000: {metrics['Hit@1000']:.4f}")

    print("=" * 80 + "\n")

    # Save detailed CSV next to log
    csv_path = args.log_path.replace(".log", ".csv")
    result_df.to_csv(csv_path, index=False)
    logger.info("Saved metrics table to %s", csv_path)


if __name__ == "__main__":
    main()