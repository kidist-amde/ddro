import argparse
import pandas as pd
import gzip
from sentence_transformers import SentenceTransformer
import torch

def compute_embeddings(merged_file_path, embedding_output_file, model_name='sentence-transformers/gtr-t5-base', batch_size=32):
    """
    Compute T5-based embeddings for the document content in the gzipped merged file.
    
    Args:
        merged_file_path (str): Path to the gzipped merged file containing document fields.
        embedding_output_file (str): Path to save the computed embeddings.
        model_name (str): Pre-trained model to use for embedding (default: gtr-t5-base).
        batch_size (int): Batch size for encoding (default: 32).
    """
    
    with gzip.open(merged_file_path, 'rt') as f:
        print(f"Reading the merged file from {merged_file_path}...")
        merged_df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', \
                                                                'short_answer', 'title', 'abstract', \
                                                                'content', 'document_url', 'doc_tac', 'language'])

    # Initialize the SentenceTransformer model (GTR-T5)
    model = SentenceTransformer(model_name)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, using CPU")

    # Extract the document content ('doc_tac') to generate embeddings for each document
    doc_tac_list = merged_df['doc_tac'].tolist()

    # Generate embeddings for each document using the GTR-T5 model
    print(f"Generating embeddings for {len(doc_tac_list)} documents using the model '{model_name}'...")
    embeddings = model.encode(doc_tac_list, batch_size=batch_size, show_progress_bar=True, \
                            device='cuda' if torch.cuda.is_available() else 'cpu')

    # Save embeddings to file in the specified format
    print(f"Saving embeddings to {embedding_output_file}...")
    with open(embedding_output_file, 'w') as f:
        for idx, embedding in enumerate(embeddings):
            doc_id = merged_df['id'][idx]  # Get the document ID
            embedding_str = ','.join(map(str, embedding))  # Convert embedding array to comma-separated string
            f.write(f'[{doc_id}]\t{embedding_str}\n')  # Write in the desired format: [doc_id] \t embedding


    print(f"Embeddings have been successfully saved to {embedding_output_file}.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Compute T5-based embeddings for NQ document content")

    # Arguments for the script
    parser.add_argument('--merged_file', type=str, required=True, help="Path to the gzipped merged TSV file containing the document content")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the computed embeddings")
    parser.add_argument('--model_name', type=str, default='sentence-transformers/gtr-t5-base', help="Pre-trained model to use for embeddings (default: 'gtr-t5-base')")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for encoding the embeddings (default: 32)")

    # Parse arguments
    args = parser.parse_args()

    # Compute and save embeddings
    compute_embeddings(args.merged_file, args.output_file, args.model_name, args.batch_size)
