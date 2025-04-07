
#  DDRO Data Preprocessing

This directory contains unified scripts for preprocessing data for **Direct Document Relevance Optimization (DDRO)**. It supports both **MS MARCO** and **Natural Questions (NQ)** datasets for generating training triples, computing dense embeddings, sampling documents, and converting formats.

---


## Available Scripts

```bash
ddro/data/preprocessing/
├── negative_sampling.py                # BM25-based hard negative sampling for MS MARCO or NQ
├── generate_doc_embeddings.py          # Compute dense GTR-T5 embeddings for documents (MS MARCO, NQ)
├── sample_msmarco_dataset.py           # Create top/random subsets of MSMARCO based on relevance frequency
├── convert_tsv_to_json_array.py        # Convert MSMARCO TSV docs to a flat JSON array
├── convert_nq_to_msmarco_format.py     # Reformat NQ data into MSMARCO-style queries and qrels
├── process_nq_dataset.py               # Extract and clean fields from raw NQ into structured format
├── create_nq_triples.py                # Generate BM25-based training triples for NQ
└── README.md                           # You are here
```

---
##  Example Usage

### BM25 Negative Sampling

```bash
python negative_sampling.py \
  --dataset [msmarco|nq] \
  --rank_file path/to/bm25.txt \
  --qrels_file path/to/qrels.tsv.gz \
  --queries_file path/to/queries.tsv.gz \
  --docs_file path/to/docs.tsv \
  --output_path triples.tsv \
  --num_negative_per_query 5
```

### Dense Document Embedding

```bash
python generate_doc_embeddings.py \
  --input_path data/msmarco-docs.jsonl \
  --output_path doc_embeddings.tsv \
  --dataset msmarco \
  --batch_size 128
```

---

##  Output Formats

- **Triples** (TSV):  
  `<query_id> <query_text> <pos_url> <pos_docid> <neg_url> <neg_docid>`
  
- **Embeddings** (TSV):  
  `[docid] \t 0.123,0.456,...`

- **Documents** (JSONL):  
  `{ "id": ..., "contents": ... }`

---

## 🗃️ Expected Directory Structure

```
resources/
├── datasets/
│   ├── raw/
│   │   ├── msmarco-data/
│   │   └── nq-data/
│   └── processed/
│       ├── msmarco-data/
│       └── nq-data/
```


Maintained with ❤️ by the DDRO team.

