import os
import time
import torch
import random
import argparse
from utils import load_model, set_seed
from tqdm import tqdm
from trie import Trie
from evaluate import evaluator
from collections import defaultdict
from torch.utils.data import DataLoader
import sys

sys.path.append('/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro')
from pretrain.T5ForPretrain import T5ForPretrain
from pretrain_dataset import PretrainDataForT5
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set device
device = torch.device("cuda:0")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0.1, type=float, help="The ratio of warmup steps.")
parser.add_argument("--output_every_n_step", default=5000, type=int, help="Steps to output training info.")
parser.add_argument("--save_every_n_epoch", default=2, type=int, help="Epochs to save the model.")
parser.add_argument("--operation", default="training", type=str, help="Operation to perform: training/testing.")
parser.add_argument("--use_docid_rank", default="False", type=str, help="Use docid for ranking.")
parser.add_argument("--load_ckpt", default="False", type=str, help="Load checkpoint if available.")

# Paths
parser.add_argument("--save_path", default="./model/", type=str, help="Path to save models.")
parser.add_argument("--log_path", default="./log/", type=str, help="Path for logs.")
parser.add_argument("--doc_file_path", type=str, help='Path of original sentence data.')
parser.add_argument("--docid_path", type=str, help='Path of encoded docid.')
parser.add_argument("--train_file_path", type=str, help="Path of the training file.")
parser.add_argument("--test_file_path", type=str, help="Path of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="Path of the pretrained model checkpoint.")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="Path to load a checkpoint.")
parser.add_argument("--dataset_script_dir", type=str, help="Path of the dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="Path of the dataset cache.")

# Model hyperparameters
parser.add_argument("--add_doc_num", type=int, help="Number of special tokens added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="Max input sequence length.")
parser.add_argument("--max_docid_length", type=int, default=1, help="Max docid sequence length.")
parser.add_argument("--use_origin_head", default="False", type=str, help="Load the lm_head from the pretrained model.")
parser.add_argument("--num_beams", default=10, type=int, help="Number of beams for beam search.")
# Additional parser argument for gradient accumulation
parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="Number of steps to accumulate gradients.")


args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()

print("batch_size:", args.batch_size)
print("start a new run with args:", args)

# Setup logging
logger = open(args.log_path, "a")
logger.write("\n")
logger.write(f"start a new running with args: {args}\n")

# Initialize tokenizer and model, resize embeddings for added tokens
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)

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
            encode = [int(x) for x in encode.split(",")]
            encoded_docids.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docids, encode_2_docid
    
def train_model(train_data):
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    pretrain_model.resize_token_embeddings(pretrain_model.config.vocab_size + args.add_doc_num)

    model = T5ForPretrain(pretrain_model, args)
    
    if args.load_ckpt == "True":
        save_model = load_model(os.path.join(args.load_ckpt_path))
        model.load_state_dict(save_model)
        print("Successfully load checkpoint from ", args.load_ckpt_path)
    
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data)

def train_step(model, train_data):
    with torch.no_grad():
        for key in train_data.keys():
            if key in ["query_id", "doc_id"]:
                continue
            train_data[key] = train_data[key].to(device)
    input_ids = train_data["input_ids"]
    attention_mask = train_data["attention_mask"]
    labels = train_data["docid_labels"]

    loss = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return loss

def fit(model, X_train):
    print("start training...")
    train_dataset = PretrainDataForT5(X_train, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)  # Lower initial learning rate
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)  # Increased warm-up ratio
    os.makedirs(args.save_path, exist_ok=True)

    # Layer-freezing logic based on T5 naming conventions, with DataParallel fix
    num_hidden_layers = model.module.config.num_hidden_layers if isinstance(model, torch.nn.DataParallel) else model.config.num_hidden_layers
    for name, param in model.named_parameters():
        if 'block' in name:  # T5 uses 'block' to define encoder and decoder layers
            try:
                layer_index = int(name.split('block')[1].split('.')[1])  # Extracts numeric layer index
                if layer_index < num_hidden_layers - 3:  # Freeze lower layers
                    param.requires_grad = False
            except (IndexError, ValueError):
                print(f"Warning: Could not parse layer index from parameter name '{name}'")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        for i, training_data in enumerate(tqdm(train_dataloader)):
            loss = train_step(model, training_data)
            loss = loss.mean()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            if i % args.output_every_n_step == 0:
                localtime = time.asctime(time.localtime(time.time()))
                print(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}")
                logger.write(f"{localtime} step: {i}, lr: {args.learning_rate}, loss: {loss.item()}\n")
                logger.flush()
            avg_loss += loss.item()
        cnt = len(train_dataset) // args.batch_size + 1
        print("Average loss:{:.6f} ".format(avg_loss / cnt))
        logger.write("Average loss:{:.6f} \n".format(avg_loss / cnt))

        if (epoch+1) % args.save_every_n_epoch == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"model_{epoch}.pkl"))
            print(f"Save the model in {args.save_path}")
    logger.close()


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

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])

    def prefix_allowed_tokens_fn(batch_id, sent): 
        return docid_trie.get(sent.tolist())

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
        test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) # 构建训练集
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        truth, prediction, inputs = [], [], []

        for i, testing_data in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                for key in testing_data.keys():
                    if key in ["query_id", "doc_id"]:
                        continue
                    testing_data[key] = testing_data[key].to(device)
            
            input_ids = testing_data["input_ids"]
            if args.use_docid_rank == "False":
                labels = testing_data["docid_labels"] # encoded docid
                truth.extend([[docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
            else:
                labels = testing_data["query_id"] # docid
                truth.extend([[docid] for docid in labels])
            
            inputs.extend(input_ids)

            outputs = model.generate(input_ids, max_length=args.max_docid_length+1, num_return_sequences=args.num_beams, num_beams=args.num_beams, do_sample=False, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

            for j in range(input_ids.shape[0]):
                query_term = tokenizer.decode(input_ids[j], skip_special_tokens=True).split()
                doc_rank = []
                batch_output = outputs[j*args.num_beams:(j+1)*args.num_beams].cpu().numpy().tolist()
                for docid in batch_output:
                    if args.use_docid_rank == "False":
                        doc_rank.append(docid2string(docid))
                    else:
                        docid_list = encode_2_docid[docid2string(docid)]
                        if len(docid_list) > 1:
                            random.shuffle(docid_list)
                            doc_rank.extend(docid_list)
                        else:
                            doc_rank.extend(docid_list)                           
                prediction.append(doc_rank)
        result_df = myevaluator.evaluate_ranking(truth, prediction)

        # Extract and log evaluation metrics
        metrics = {
            'MRR@10': result_df['MRR@10'][0], 'MRR': result_df['MRR'][0],
            'P@1': result_df['P@1'][0], 'P@10': result_df['P@10'][0],
            'P@20': result_df['P@20'][0], 'P@100': result_df['P@100'][0],
            'R@1': result_df['R@1'][0], 'R@10': result_df['R@10'][0],
            'R@100': result_df['R@100'][0], 'R@1000': result_df['R@1000'][0],
            'Hit@1': result_df['Hit@1'][0], 'Hit@5': result_df['Hit@5'][0],
            'Hit@10': result_df['Hit@10'][0], 'Hit@100': result_df['Hit@100'][0]
        }
        localtime = time.asctime(time.localtime(time.time()))
        print(f"Evaluation Metrics: {metrics}")
        logger.write(f"Evaluation Metrics: {metrics}\n")
        logger.flush()

if __name__ == '__main__':
    if args.operation == "training":
        train_data = load_data(args.train_file_path)
        set_seed()
        train_model(train_data) # start training
    
    if args.operation == "testing":
        evaluate_beamsearch() # evaluate the model