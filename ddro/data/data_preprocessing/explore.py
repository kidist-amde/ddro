import argparse
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Argument parser for command-line inputs
parser = argparse.ArgumentParser(description="Explore and summarize the nq_merged.tsv.gz file")
parser.add_argument("--file_path", default="resources/datasets/raw/nq-data/nq_merged.tsv.gz", type=str, help="Path to the gzipped TSV file")
args = parser.parse_args()

def load_file(file_path):
    """
    Load the merged file (TSV or gzipped TSV).
    """
    if file_path.endswith(".gz"):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])
    else:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['query', 'id', 'long_answer', 'short_answer', 'title', 'abstract', 'content', 'document_url', 'doc_tac', 'language'])
    
    return df

def explore_data(df):
    """
    Explore the dataset:
    - Display missing values
    - Provide summary statistics
    - Show column types and sample values
    """
    print("\nBasic Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isna().sum())

    # print("\nSummary Statistics:")
    # print(df.describe(include='all'))

    print("\nSample Data:")
    print(df.head(2))

def plot_missing_values(df):
    """
    Plot missing values as a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_column_distributions(df):
    """
    Plot distributions for columns to understand the spread of data.
    """
    # Plot for 'id' column (check if it's unique)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['id'].astype(str), kde=False, bins=50)
    plt.title('Document ID Distribution')
    plt.xticks(rotation=45)
    plt.show()

    # Plot for the 'language' column (categorical data)
    plt.figure(figsize=(6, 4))
    sns.countplot(df['language'])
    plt.title('Language Distribution')
    plt.show()

    # Plot for the 'title' column (missing vs non-missing)
    plt.figure(figsize=(6, 4))
    df['title_missing'] = df['title'].isna()
    sns.countplot(df['title_missing'])
    plt.title('Missing Titles Count')
    plt.show()

def main():
    # Load the file
    df = load_file(args.file_path)

    # Explore the dataset
    explore_data(df)

#     # Plot missing values heatmap
#     plot_missing_values(df)

#     # Plot distributions of specific columns
#     plot_column_distributions(df)

if __name__ == "__main__":
    main()
