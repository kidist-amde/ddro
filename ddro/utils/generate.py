from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm.auto import tqdm 
import json, torch, os

# Define paths
model_path = "/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/outputs-sft-NQ/dpo/dpo_ckp_url_title_5epoch_lr5e-7_NewTripls/"
output_dir = "generated_results"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "generation_log.txt")

# Load the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

def generate_doc_url(query_text):
    prompt = f"Query: {query_text}\n\nDoc Url:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        outputs = model.generate(**inputs, num_beams=32, max_new_tokens=64, num_return_sequences=10)
    generated_output = list(map(lambda text: text.replace(prompt, ""), tokenizer.batch_decode(outputs, skip_special_tokens=True)))
    
    # Save results to log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Query: {query_text}\n")
        log_file.write("\n".join(generated_output) + "\n\n")
    
    print("Query:", query_text)
    print("\n".join(generated_output))

query_text = "calcium is known as?"
generate_doc_url(query_text)
print(f"Results saved to {log_file_path}")
