import os
import sys
import torch
import random
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
# Add src to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import load_model
from utils.trie import Trie
from utils.evaluate import evaluator
from pretrain.T5ForPretrain import T5ForPretrain
from utils.pretrain_dataset import PretrainDataForT5


device = torch.device("cuda:0")
parser = argparse.ArgumentParser()
### training settings
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="The ratio of warmup steps.")
parser.add_argument("--output_every_n_step", default=25, type=int, help="The steps to output training information.")
parser.add_argument("--save_every_n_epoch", default=25, type=int, help="The epochs to save the trained models.")
parser.add_argument("--operation", default="training", type=str, help="which operation to take, training/testing")
parser.add_argument("--use_docid_rank", default="False", type=str, help="whether to use docid for ranking, or only doc code.")
parser.add_argument("--load_ckpt", default="False", type=str, help="whether to load a trained model checkpoint.")
parser.add_argument("--debug", default="False", type=str, help="Enable detailed debugging output")

### path to load data and save models
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save trained models.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--doc_file_path", default="/ivi/ilps/personal/kmekonn/projects/DPO-Enhanced-DSI/data/processed/msmarco-docs-sents.top.300k.json", type=str, help='path of origin sent data.')
parser.add_argument("--docid_path", default="None", type=str, help='path of the encoded docid.')
parser.add_argument("--train_file_path", type=str, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the pretrained model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt of a trained model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path of dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path of dataset cache.")

### hyper-parameters to control the model
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--max_docid_length", type=int, default=1, help="the max length of docid sequences.")
parser.add_argument("--use_origin_head", default="False", type=str, help="whether to load the lm_head from the pretrained model.")
parser.add_argument("--num_beams", default=10, type=int, help="the number of beams.")

args = parser.parse_args()

DEBUG = args.debug == "True"

print("="*80)
print("CONFIGURATION SUMMARY")
print("="*80)
print(f"Operation: {args.operation}")
print(f"Debug mode: {DEBUG}")
print(f"Device: {device}")
print(f"Batch size: {args.per_gpu_batch_size * torch.cuda.device_count()}")
print(f"Num beams: {args.num_beams}")
print(f"Max docid length: {args.max_docid_length}")
print(f"Test file: {args.test_file_path}")
print(f"Docid path: {args.docid_path}")
print(f"Checkpoint: {args.save_path}")
print("="*80)

args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()

logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    """
        function: load data from the file_path
        args: file_path  -- a directory or a specific file
    """
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        data_dir = file_path
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    print("file path: ", fns)
    return fns


def load_encoded_docid(docid_path, dataset="msmarco"):
    """
        function: load encoded docid data from the docid_path
        return:
            encoded_docids: list of all encoded document identifiers.
            encode_2_docid: dict from encoded document identifiers to original unique id.
    """
    print("\n" + "="*80)
    print("LOADING ENCODED DOCIDS")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Docid path: {docid_path}")
    
    encode_2_docid = {}
    encoded_docids = []
    
    with open(docid_path, "r") as fr:
        for idx, line in enumerate(fr):
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            
            original_encode = encode
            
            # For NQ dataset, remove padding tokens (0 and 1)
            if dataset == "nq":
                encode_list = encode.split(",")
                encode = [int(x) for x in encode_list if x not in ["0", "1"]]
            else:
                # For MS MARCO, use as-is
                encode = [int(x) for x in encode.split(",")]
            
            # Debug: Print first 3 docids only
            if DEBUG and idx < 3:
                print(f"  Sample {idx}: docid={docid}")
                if dataset == "nq":
                    print(f"    Original: {original_encode}")
                    print(f"    After padding removal: {encode}")
                else:
                    print(f"    Encoded: {encode}")
            
            encoded_docids.append(encode)
            encode_str = ','.join([str(x) for x in encode])
            
            if encode_str not in encode_2_docid:
                encode_2_docid[encode_str] = [docid]
            else:
                encode_2_docid[encode_str].append(docid)
    
    print(f"Total encoded docids loaded: {len(encoded_docids)}")
    print(f"Total unique encodings: {len(encode_2_docid)}")
    
    if DEBUG:
        print(f"Sample encoding keys (first 3):")
        for i, key in enumerate(list(encode_2_docid.keys())[:3]):
            print(f"  {i+1}. {key} -> {encode_2_docid[key]}")
    print("="*80 + "\n")
    
    return encoded_docids, encode_2_docid

def evaluate_beamsearch():
    '''
        function: Generate the document identifiers with constrained beam search, and evaluate the ranking results.
    '''
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    
    model = T5ForPretrain(pretrain_model, args)
    save_model = load_model(args.save_path)
    model.load_state_dict(save_model)
    model = model.to(device)
    model.eval()
    
    myevaluator = evaluator()
    
    # Detect dataset from docid_path
    dataset = "nq" if "nq" in args.docid_path.lower() else "msmarco"
    print(f"Detected dataset: {dataset}\n")
    
    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path, dataset=dataset)
    docid_trie = Trie([[0] + item for item in encoded_docid])
    print(f"Trie built with {len(encoded_docid)} docids\n")

    def prefix_allowed_tokens_fn(batch_id, sent): 
        outputs = docid_trie.get(sent.tolist())
        if len(outputs) == 0:
            return [tokenizer.pad_token_id]
        return outputs
    
    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    if os.path.exists(args.test_file_path):
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluate on the {args.test_file_path}.")
        
        logger.write(f"{localtime} Evaluate on the {args.test_file_path}.\n")
        test_data = load_data(args.test_file_path)
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        truth, prediction, inputs = [], [], []
        
        # Counters for debugging
        total_queries = 0
        successful_matches = 0
        failed_matches = 0
        not_found_docids = set()  # Track unique not-found docids
        
        print("Start evaluating... \n")
        
        for i, testing_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Evaluating"):
            with torch.no_grad():
                for key in testing_data.keys():
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            
            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"]
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"]
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
            
            for j in range(input_ids.shape[0]):
                total_queries += 1
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                
                # ONLY print detailed debug for first 2 queries
                debug_this_query = DEBUG and total_queries <= 2
                
                if debug_this_query:
                    query_term = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    print("\n" + "-"*80)
                    print(f"QUERY {total_queries} DEBUG")
                    print("-"*80)
                    print(f"Query: {query_term}")
                    print(f"Ground truth: {labels[j] if args.use_docid_rank == 'True' else docid2string(testing_data['docid_labels'][j].cpu().numpy().tolist())}")
                    print(f"\nGenerated {len(batch_output)} candidates:")
                
                batch_has_match = False
                for beam_idx, docid in enumerate(batch_output):
                    docid_string = docid2string(docid)
                    
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid_string)
                        if debug_this_query and beam_idx < 5:  # Only print first 5 beams
                            print(f"  Beam {beam_idx+1}: {docid_string}")
                    else:
                        # Check if this docid exists in our dictionary
                        if docid_string in encode_2_docid:
                            docid_list = encode_2_docid[docid_string]
                            if len(docid_list) > 1:
                                random.shuffle(docid_list)
                                doc_rank.extend(docid_list)
                            else:
                                doc_rank.extend(docid_list)
                            
                            batch_has_match = True
                            if debug_this_query and beam_idx < 5:
                                print(f"  Beam {beam_idx+1}: {docid_string} -> FOUND: {docid_list}")
                        else:
                            if debug_this_query and beam_idx < 5:
                                print(f"  Beam {beam_idx+1}: {docid_string} -> NOT FOUND")
                            failed_matches += 1
                            not_found_docids.add(docid_string)
                
                if batch_has_match:
                    successful_matches += 1
                    
                if debug_this_query:
                    print(f"\nFinal ranking (top 5): {doc_rank[:5]}")
                    print("-"*80)
                
                prediction.append(doc_rank)

        print("\n" + "="*80)
        print("EVALUATION STATISTICS")
        print("="*80)
        print(f"Total queries processed: {total_queries}")
        print(f"Queries with at least one match: {successful_matches}")
        print(f"Match rate: {successful_matches/total_queries*100:.2f}%")
        print(f"Total failed docid lookups: {failed_matches}")
        print(f"Unique not-found docids: {len(not_found_docids)}")
        
        if not_found_docids and len(not_found_docids) <= 10:
            print(f"\nSample not-found docids:")
            for docid in list(not_found_docids)[:10]:
                print(f"  {docid}")
        print("="*80 + "\n")

        print("Computing metrics...")
        result_df = myevaluator.evaluate_ranking(truth, prediction)
        # Extracting metrics
        _mrr10 = result_df['MRR@10'].values.mean()
        _mrr = result_df['MRR'].values.mean()
        _ndcg10 = result_df['NDCG@10'].values.mean()
        _ndcg20 = result_df['NDCG@20'].values.mean()
        _ndcg100 = result_df['NDCG@100'].values.mean()
        _map20 = result_df['MAP@20'].values.mean()
        _p1 = result_df['P@1'].values.mean()
        _p10 = result_df['P@10'].values.mean()
        _p20 = result_df['P@20'].values.mean()
        _p100 = result_df['P@100'].values.mean()
        _r1 = result_df['R@1'].values.mean()
        _r10 = result_df['R@10'].values.mean()
        _r100 = result_df['R@100'].values.mean()
        _r1000 = result_df['R@1000'].values.mean()
        _hit1 = result_df['Hit@1'].values.mean()
        _hit5 = result_df['Hit@5'].values.mean()
        _hit10 = result_df['Hit@10'].values.mean()
        _hit100 = result_df['Hit@100'].values.mean()

        localtime = time.asctime(time.localtime(time.time()))
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"MRR@10: {_mrr10:.4f} | MRR: {_mrr:.4f}")
        print(f"P@1: {_p1:.4f} | P@10: {_p10:.4f} | P@20: {_p20:.4f}")
        print(f"R@1: {_r1:.4f} | R@10: {_r10:.4f} | R@100: {_r100:.4f} | R@1000: {_r1000:.4f}")
        print(f"Hit@1: {_hit1:.4f} | Hit@5: {_hit5:.4f} | Hit@10: {_hit10:.4f} | Hit@100: {_hit100:.4f}")
        print("="*80 + "\n")
        
        logger.write(f"{localtime} mrr@10:{_mrr10}, mrr:{_mrr}, p@1:{_p1}, p@10:{_p10}, p@20:{_p20}, p@100:{_p100}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}, r@1000:{_r1000}, hit@1:{_hit1}, hit@5:{_hit5}, hit@10:{_hit10}, hit@100:{_hit100}\n")
        csv_path = args.log_path.replace(".log", ".csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
if __name__ == '__main__':
    if args.operation == "testing":
        evaluate_beamsearch()