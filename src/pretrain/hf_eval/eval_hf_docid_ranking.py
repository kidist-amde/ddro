import os
import torch
import random
import argparse
from tqdm.auto import tqdm
from trie import Trie
from evaluate_per_query import evaluator
from torch.utils.data import DataLoader
from utils.pretrain_dataset import PretrainDataForT5
from transformers import T5Tokenizer, T5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face T5 models for document ranking")
    
    # Hugging Face model
    parser.add_argument("--hf_model_name", type=str, default="kiyam/ddro-msmarco-tu",
                       help="Hugging Face model name")
    
    # Data paths
    parser.add_argument("--doc_file_path", type=str, required=True, help="Path to document file")
    parser.add_argument("--docid_path", type=str, required=True, help="Path to encoded docid mapping")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--dataset_script_dir", type=str, required=True, help="Dataset script directory")
    parser.add_argument("--dataset_cache_dir", type=str, required=True, help="Dataset cache directory")
    
    # Model configuration
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum input sequence length")
    parser.add_argument("--max_docid_length", type=int, default=1, help="Maximum docid length")
    
    # Evaluation parameters
    parser.add_argument("--per_gpu_batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search")
    parser.add_argument("--use_docid_rank", type=str, default="True", help="Whether to use docid ranking")
    
    # Performance options
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 for faster inference")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping strategy")
    
    # Logging
    parser.add_argument("--log_path", type=str, required=True, help="Path for logging results")
    
    return parser.parse_args()


def load_encoded_docid(docid_path):
    """Load encoded docid mappings from file."""
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
    """Convert docid list to string representation."""
    return ",".join(str(x) for x in docid if x not in [0, 1])


def load_hf_model(args):
    """Load model and tokenizer from Hugging Face Hub."""
    print(f"Loading model from Hugging Face: {args.hf_model_name}")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.hf_model_name)
    
    # Load model with appropriate settings
    model_kwargs = {
        "device_map": args.device_map,
    }
    
    if args.use_fp16:
        model_kwargs["torch_dtype"] = torch.float16
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.hf_model_name,
        **model_kwargs
    )
    
    print(f"Model loaded successfully!")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def evaluate_hf_model(args):
    """Main evaluation function for HF models."""
    
    # Load HF model and tokenizer
    model, tokenizer = load_hf_model(args)
    model.eval()
    
    # Setup logging
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logger = open(args.log_path, "w")
    logger.write(f"Evaluating HF model: {args.hf_model_name}\n")
    logger.write(f"Args: {args}\n\n")
    
    # Load docid mappings and create trie for constrained generation
    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + d for d in encoded_docid])
    
    def prefix_allowed_tokens_fn(batch_id, sent):
        """Constrain generation to valid docids only."""
        out = docid_trie.get(sent.tolist())
        return out if out else [tokenizer.pad_token_id]
    
    # Create a simple args object for dataset compatibility
    class DatasetArgs:
        def __init__(self, args):
            self.max_seq_length = args.max_seq_length
            self.max_docid_length = args.max_docid_length
            # Add any other attributes that PretrainDataForT5 might need
            for key, value in vars(args).items():
                setattr(self, key, value)
    
    dataset_args = DatasetArgs(args)
    
    # Load test data
    test_data = [args.test_file_path]
    dataset = PretrainDataForT5(
        test_data,
        args.max_seq_length,
        args.max_docid_length,
        tokenizer,
        args.dataset_script_dir,
        args.dataset_cache_dir,
        dataset_args
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_gpu_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize evaluator
    evaluator_ = evaluator()
    truth, prediction = [], []
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    # Run evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            
            # Move tensors to appropriate device (handle multi-GPU setups)
            input_ids = batch["input_ids"]
            if not args.device_map or args.device_map == "auto":
                # Let the model handle device placement
                pass
            else:
                input_ids = input_ids.to(model.device)
            
            # Prepare ground truth
            if args.use_docid_rank == "False":
                labels = batch["docid_labels"].cpu().numpy().tolist()
                truth.extend([[docid2string(x)] for x in labels])
            else:
                truth.extend([[x] for x in batch.get("query_id", [f"query_{batch_idx}_{i}" for i in range(len(input_ids))])])
            
            # Generate predictions using beam search
            try:
                outputs = model.generate(
                    input_ids,
                    max_length=args.max_docid_length + 1,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    do_sample=False,
                    early_stopping=True
                )
            except Exception as e:
                print(f"Error during generation for batch {batch_idx}: {e}")
                # Skip this batch or provide default predictions
                for i in range(input_ids.size(0)):
                    prediction.append([])
                continue
            
            # Process predictions
            for i in range(input_ids.size(0)):
                try:
                    batch_output = outputs[i * args.num_beams:(i + 1) * args.num_beams].cpu().numpy().tolist()
                    doc_rank = []
                    
                    for docid in batch_output:
                        doc_str = docid2string(docid)
                        if args.use_docid_rank == "False":
                            doc_rank.append(doc_str)
                        else:
                            docids = encode_2_docid.get(doc_str, [])
                            if docids:
                                random.shuffle(docids)
                                doc_rank.extend(docids)
                    
                    prediction.append(doc_rank)
                    
                except Exception as e:
                    print(f"Error processing predictions for sample {i} in batch {batch_idx}: {e}")
                    prediction.append([])
    
    print(f"Evaluation completed. Processing {len(prediction)} predictions...")
    
    # Evaluate and save results
    try:
        results = evaluator_.evaluate_ranking(truth, prediction)
        
        # Save detailed results
        csv_path = args.log_path.replace(".log", ".csv")
        results.to_csv(csv_path, index=False)
        
        # Log summary
        logger.write("Evaluation Results:\n")
        logger.write(str(results.mean()) + "\n")
        
        print("Evaluation complete!")
        print(f"Results saved to: {csv_path}")
        print("Mean metrics:")
        print(results.mean())
        
    except Exception as e:
        error_msg = f"Error during evaluation: {e}"
        print(error_msg)
        logger.write(error_msg + "\n")
    
    finally:
        logger.close()


if __name__ == '__main__':
    args = parse_args()
    evaluate_hf_model(args)