file_path = "ddro/resources/datasets/processed/msmarco-data/hard_negatives/msmarco_train_triples"

# Counting the number of lines in the file
count = 0
try:
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            count += 1
    print(f"Number of documents in the file: {count}")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
