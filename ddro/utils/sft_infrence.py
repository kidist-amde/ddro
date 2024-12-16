import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from trie import Trie
from pretrain_dataset import PretrainDataForT5
import argparse
from utils import load_model
from tqdm import tqdm
from T5ForPretrain import T5ForPretrain

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

def load_encoded_docid(docid_path):
    """
        function: load encoded docid data from the docid_path
        return:
            encoded_docids: list of all encoded document identifiers.
            encode_2_docid: dict from encoded document identifiers to original unique id.
    """
    encode_2_docid = {}
    encoded_docids = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            # since I added padding when I generate the ids , I need to remove the padding
            encode_list = encode.split(",")
            encode= [int(x) for x in encode_list if x not in ["0", "1"]]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid

# # FOR MSMARCO DATA THE DOC ID MUST BE UPLOADED WITH THIS FUNCTION
# def load_encoded_docid(docid_path):
#     """
#         function: load encoded docid data from the docid_path
#         return:
#             encoded_docids: list of all encoded document identifiers.
#             encode_2_docid: dict from encoded document identifiers to original unique id.
#     """
#     encode_2_docid = {}
#     encoded_docids = []
#     with open(docid_path, "r") as fr:
#         for line in fr:
#             docid, encode = line.strip().split("\t")
#             docid = docid.lower()
#             encode = [int(x) for x in encode.split(",")]
#             encoded_docids.append(encode)
#             encode = ','.join([str(x) for x in encode])
#             if encode not in encode_2_docid:
#                 encode_2_docid[encode] = [docid]
#             else:
#                 encode_2_docid[encode].append(docid)
#     return encoded_docids, encode_2_docid

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def docid2string(docid):
    """Converts a document ID to a string."""
    x_list = []
    for x in docid:
        if x != 0:
            x_list.append(str(x))
        if x == 1:
            break
    return ",".join(x_list)

def generate_top_results(args):
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)

    # Adjust model's embedding size to match the checkpoint
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)
    model = T5ForPretrain(pretrain_model, args)

    # Load saved model checkpoint
    save_model = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(save_model, strict=False)  # Allow partial loading
    model = model.to(device)
    model.eval()

    # Load encoded doc IDs and Trie
    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])

    def prefix_allowed_tokens_fn(batch_id, sent):
        allowed_tokens = docid_trie.get(sent.tolist())
        if not allowed_tokens:  # Fallback to ensure no empty list
            return [tokenizer.pad_token_id]
        return allowed_tokens

    def docid2string(docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)

    # Load test data
    test_data = load_data(args.test_file_path)
    test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = []

    # Generate results
    print("Start generating top 1000 results...")
    for i, testing_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Generating"):
        with torch.no_grad():
            for key in testing_data.keys():
                if key in ["query_id", "doc_id"]:
                    continue
                testing_data[key] = testing_data[key].to(device)

            input_ids = testing_data["input_ids"]
            outputs = model.generate(
                input_ids,
                max_length=args.max_docid_length + 1,
                num_return_sequences=1000,
                num_beams=1000,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

            for j in range(input_ids.shape[0]):
                batch_output = outputs[j * 1000:(j + 1) * 1000].cpu().numpy().tolist()
                doc_ranks = [docid2string(docid) for docid in batch_output]
                results.append({
                    "input": tokenizer.decode(input_ids[j], skip_special_tokens=True),
                    "doc_ranks": doc_ranks
                })

    # Save results
    output_path = os.path.join(args.output_path, "top_1000_results.json")
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_path, "w") as f:
        import json
        json.dump(results, f, indent=4)

    print(f"Top 1000 results saved to {output_path}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate top 1000 results from the model.")
    parser.add_argument("--pretrain_model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--test_file_path", type=str, required=True, help="Path to the testing file.")
    parser.add_argument("--docid_path", type=str, required=True, help="Path to the encoded document IDs.")
    parser.add_argument("--output_path", type=str, default="./output", help="Path to save the output results.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--max_docid_length", type=int, default=1, help="Maximum document ID length.")
    parser.add_argument("--add_doc_num", type=int, default=0, help="Number of document IDs to add to vocabulary.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--dataset_script_dir", type=str, help="Path to the dataset script directory.")
    parser.add_argument("--dataset_cache_dir", type=str, help="Path to the dataset cache directory.")
    parser.add_argument("--use_origin_head", default="False", type=str, help="Whether to use the original lm_head from the pretrained model.")
    parser.add_argument("--dataset_script_dir", type=str, default="resources/dataset_scripts", help="Path to the dataset script directory.")


    args = parser.parse_args()

    # Generate top 1000 results
    generate_top_results(args)
