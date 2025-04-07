from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the paths
save_directory = 'resources/transformer_models/t5-base'

# Load the tokenizer and save it
tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.save_pretrained(save_directory)

# Load the model and save it
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
