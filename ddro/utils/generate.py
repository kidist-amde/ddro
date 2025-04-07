import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from trie_url import Trie
import IPython
# Configuration
model_path = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/outputs-sft-NQ/dpo/dpo_ckp_url_title_5epoch_lr5e-7_NewTripls/"
docid_path = "resources/ENCODED_DOC_IDs/t5_url_msmarco.txt"
output_dir = "generated_results"
num_beams = 10
max_new_tokens = 64
length_penalty = 1.0

# Create output directory
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "generation_log.txt")

# Load pretrained model and tokenizer
print("Loading model and tokenizer...")
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def load_encoded_docid(docid_path):
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

encoded_docids, encode_2_docid = load_encoded_docid(docid_path)
docid_trie = Trie([[0] + item for item in encoded_docids])

def prefix_allowed_tokens_fn(batch_id, sent):
    allowed_tokens = docid_trie.get(sent.tolist())
    if not allowed_tokens:
        return [tokenizer.pad_token_id]
    return allowed_tokens

def generate_doc_url(query_text):
    prompt = f"Find the most relevant document URL for: {query_text}\n\nDoc Url:"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=256
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            early_stopping=True,
            num_return_sequences=5,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )
    
    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    cleaned_output = [text.replace(prompt, "").replace("<pad>", "").strip() for text in generated_output]
    
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Query: {query_text}\n")
        log_file.write("\n".join(cleaned_output) + "\n\n")
    
    print("Query:", query_text)
    for url in cleaned_output:
        print("URL:", url)

# Example usage
query_text = "calcium is known as?"
query_text = " what does the name halima mean"
generate_doc_url(query_text)
print(f"Results saved to {log_file_path}")
IPython.embed()


