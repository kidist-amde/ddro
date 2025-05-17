!/bin/bash

# MS MARCO  Document ranking dataset
# (https://microsoft.github.io/msmarco/Datasets.html#document-ranking-dataset)


# # Create a directory for raw datasets if it doesn't exist
mkdir -p resources/dataset/raw/msmarco-data
cd resources/dataset/raw/msmarco-data

# # Download the query files
# echo "Downloading the query files"
wget  "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz"
wget  "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz"

# # Download the qrels files
# echo "Downloading the qrels files"
wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz"
wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"

# # Download the lookup files
# echo "Downloading the lookup files"
# wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz"

# # Download the top100 files
# echo "Downloading the top100 files"
# wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz"
# wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz"

# Download the document files
echo "Downloading the document files"
wget "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"


