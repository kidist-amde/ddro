# 🧱 DDRO Data Preparation & Instance Generation

This directory contains unified scripts for preparing and transforming data for **Direct Document Relevance Optimization (DDRO)**. It covers:

* Preprocessing MS MARCO and Natural Questions datasets
* Generating dense document embeddings
* DocID encoding (URL, PQ, Atomic)
* Creating training and evaluation instances

---


## 📁 Available Scripts — `src/data/data_prep/`

This folder contains all core scripts for preprocessing datasets, generating document embeddings, encoding docids, and preparing training/evaluation instances for DDRO.

---

### 🧼 Dataset Preprocessing

Scripts for transforming and sampling datasets:

```bash
data_prep/
├── convert_tsv_to_json_array.py        # Convert MS MARCO .tsv to flat JSONL format
├── sample_top300k_msmarco_documents.py # Sample top or random 300K MS MARCO documents
├── negative_sampling.py                # BM25-based hard negative mining
├── doc2query_query_generator.py        # Generate pseudoqueries using doc2query-T5
├── generate_doc_embeddings.py          # Generate GTR-T5 dense embeddings
```

---

### 🧪 Natural Questions (NQ) Utilities

Located under the `nq/` subdirectory:

```bash
nq/
├── process_nq_dataset.py               # Clean, flatten and merge original NQ files
├── convert_nq_to_msmarco_format.py     # Convert NQ format → MS MARCO query-passage format
├── create_nq_triples.py                # Create BM25-based NQ training triples
```

---

### 📦 Instance Generation

Scripts for preparing model training and evaluation input instances:

```bash
data_prep/
├── generate_encoded_docids.py          # Encode documents into docids (PQ, URL, etc.)
├── generate_train_data_wrapper.py      # Wrapper to generate all training instances
├── generate_train_instances.py         # Create pretrain, pseudoquery, and finetune inputs
├── generate_eval_data_wrapper.py       # Wrapper for evaluation data preparation
├── generate_eval_instances.py          # Format eval data for testing SFT/DDRO
```

---

### 📓 Demos & Explorations

Notebook demos for docid formats and encodings:

```bash
data_prep/
├── pq_docid_demo.ipynb                 # Visualize PQ-based docid encoding
├── rq_docid_demo.ipynb                 # Visualize ranking-quality based docid
├── url_title_docid_demo.ipynb          # URL+title docid construction example
```

---

### 🗂️ This README

```bash
data_prep/
└── README.md                           # You’re here!
```


## 📊 Data Preparation

DDRO supports both **MS MARCO** and **Natural Questions (NQ)** benchmarks.

### ✅ MS MARCO: Sample Top-300K Subset

```bash
sbatch src/scripts/preprocess/sample_top_docs.sh
```

📌 Generates: `resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz`
(JSONL format, sentence-tokenized, ranked by qrels frequency)

---

## 🔢 DocID Representations

You can either **generate** document ID (`docid`) representations locally or **download** pre-computed ones from 🤗 [Hugging Face](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5).

Place downloaded files in:

```bash
resources/datasets/processed/msmarco-data/encoded_docid/
```

#### 📄 Example: `url_docid` format

```text
[d108472] 594,858,7,17,4624,5,287,1
[d1842]   3,89,9,1824,3105,4440,...,1677,1
```

---

### 🛠️ Generating DocID Representations Locally

To generate PQ-based docids (**used in this project**):

#### 1️⃣ Generate T5 Document Embeddings

⚠️ **Attention:** Ensure the script sets the dataset to `"msmarco"`:

```bash
sbatch src/scripts/preprocess/generate_doc_embeddings.sh
```

#### 2️⃣ Encode Document IDs

Use the script below to generate `pq` docids (other types are also supported):

```bash
sbatch src/scripts/preprocess/generate_encoded_ids.sh
```

⚠️  Pass --encoding pq (default) or --encoding url to specify the format.

---

#### 📄 Example: `pq_docid` format

```text
[d108472] 32211,32518,32782,33144,33382,...
[d1842]   32177,32471,32844,33053,33163,...
```
---


## 🛠 Training Instance Generation

To train the **Phase 1: Supervised Fine-Tuning (SFT)** model, we generate three types of `input → docid` training instances. These follow a **curriculum-based progression** aligned with document relevance modeling.

---

### 🎯 SFT Training Stages

| Stage                     | Input → Target        | Purpose                              |
| ------------------------- | --------------------- | ------------------------------------ |
| **1. Pretraining**        | `doc → docid`         | General content understanding        |
| **2. Search Pretraining** | `pseudoquery → docid` | Learn retrieval-style query behavior |
| **3. Finetuning**         | `query → docid`       | Supervised learning from qrels       |

---

### ✏️ Pseudo Query Generation with `docTTTTTquery`

To enable search pretraining, generate pseudo queries from raw documents using a `docT5query` model. The model generates **10 queries per document** to enhance document-query pair coverage for retrieval tasks.

---

#### 🔧 Finetune `docTTTTTquery` on NQ or MS MARCO

Use the following script to finetune `docT5query` on the **Natural Questions (NQ)** dataset. You can apply the same procedure for **MS MARCO** document ranking:

```bash
bash src/scripts/preprocess/finetune_docTTTTTTquery.sh
```


#### 🛠 Generate Pseudo Queries

Once the model is finetuned, generate 10 pseudo-queries per document using:

```bash
python src/scripts/preprocess/pseudo_queries_generator.sh
```


#### 📥 Or download from Hugging Face:

**[DDRO 🤗 Collection](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5)**

Place the downloaded queries under:

```
ddro/resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt  
ddro/resources/datasets/processed/nq-data/nq_pseudo_query_10.txt
```

Expected format (`[docid] <TAB> pseudo_query`):

```
[d301595]	what age should a child know what they like to eat  
[d301595]	what is the average age for a child to be independent  
...
```

---

### ⚙️ Step 1: Generate Raw Supervision Signals

Use the following entry-point script to generate both **training** and **evaluation** instances for MS MARCO and NQ. 

```bash
sbatch src/scripts/preprocess/generate_train_and_eval_instances.sh
```

**Outputs:**

* `passage.jsonl` — raw document text
* `sampled_terms.jsonl` — sampled key phrases
* `fake_query.jsonl` — pseudo queries
* `query.jsonl` — real queries from qrels
* `eval_data_top_300k/query_dev.jsonl` — development evaluation data

---

### ⚙️ Step 2: Merge into Curriculum Format

Format data into `input → docid` pairs for training.

Make sure your `--cur_data` argument is one of the following stages:

* `general_pretrain`
* `search_pretrain`
* `finetune`

```bash
sbatch src/scripts/preprocess/generate_3stage_train_data.sh
```

**Produces:**

* `general_pretrain.t5_128_10.{encoding}.300k.json`
* `search_pretrain.t5_128_10.{encoding}.300k.json`
* `finetune.t5_128_10.{encoding}.300k.json` <br>*(where `{encoding}` can be `pq`, `url`, etc.)*

| Mode (`--cur_data`) | Input Files                                                  | Output File                       | Description                     |
| ------------------- | ------------------------------------------------------------ | --------------------------------- | ------------------------------- |
| `general_pretrain`  | `passage.jsonl + sampled_terms.jsonl + enhanced_docid.jsonl` | `general_pretrain.t5_128_10.json` | Raw document → docid            |
| `search_pretrain`   | `fake_query.jsonl`                                           | `search_pretrain.t5_128_10.json`  | Pseudo query → docid            |
| `finetune`          | `query.jsonl`                                                | `finetune.t5_128_10.json`         | Real query → docid (from qrels) |

---

### 📚 Natural Questions (NQ)

To prepare the **Natural Questions (NQ)** dataset in MS MARCO-style format and generate training-ready encodings, follow the steps below.

---

#### 🧹 Step 1: Preprocess and Convert to MS MARCO Format

Run the following scripts to clean and reformat NQ into MS MARCO-style layout:

```bash
sbatch src/scripts/preprocess/preprocess_nq_dataset.sh               # Cleans and merges raw NQ data
sbatch src/scripts/preprocess/convert_nq_to_msmarco_format.sh        # Converts to MS MARCO-style format
```

> 📝 After generating the MS MARCO-style dataset, follow the same steps described for MS MARCO above. Be sure to **replace the data paths** and **set the dataset type to `nq`** in all relevant scripts.

---
### ⚠️ NQ dev-split clarification  
We ship **two ways** to build the NQ train / dev files:

| Mode | Flag | Dev queries | Page overlap |
|------|------|-------------|--------------|
| **Strict isolation** (default) | `--strict_split true` | ~1.7 k | none |
| **Canonical NQ** (original) | `--strict_split false` | 7 830 | allowed |

Use the strict mode when you need zero document leakage between train and dev (closed-book QA, retrieval).  
Use the canonical mode to match scores reported in most NQ papers.  
Both options are produced by `process_nq_dataset.py`; the script now logs how many dev queries are removed when `--strict_split` is `true`.

---

## 🔍 BM25 Retrieval (via Pyserini)

BM25 is used in this project for:

1. **Sparse baseline evaluation**
2. **Hard negative mining** for contrastive and pairwise training (e.g., DDRO)

---

### ⚙️ Environment Setup

```bash
conda env create -f pyserini.yml
conda activate pyserini
pip install -r pyserini.txt
```

---

### 🔄 Convert MS MARCO TSV → JSONL

Convert MS MARCO `.tsv.gz` corpus into a Pyserini-compatible JSONL format:

```bash
python src/data/data_prep/convert_tsv_to_json_array.py
```

* **Input:** `msmarco-docs.tsv.gz`
* **Output:** `msmarco-docs.jsonl`

  ```json
  {"id": "docid", "contents": "title + body"}
  ```

---

### 📦 Index & Retrieve with Pyserini

Use SLURM scripts to index and retrieve:

```bash
sbatch src/scripts/bm25/run_bm25_retrieval_msmarco.sh
```
run: src/data/data_prep/nq/convert_json_array_to_jsonl.ipynb

```bash 
sbatch src/scripts/bm25/run_bm25_retrieval_nq.sh
```

Each script:

* Indexes the document corpus using Pyserini
* Performs BM25 retrieval with optimized hyperparameters (`k1`, `b`)
* Saves output in MS MARCO format

---

## 🔁 Negative Sampling for Triplet Generation

BM25 top-k runs are used to sample **hard negatives** for training.

Generate training triplets:

```bash
sbatch src/scripts/preprocess/generate_msmarco_triples.sh
sbatch src/scripts/preprocess/create_nq_triples.sh
```

Alternatively, for MS MARCO download the official 100 negatives per query:

📥 [msmarco-doctrain-top100.gz](https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz)

Then generate triplets using 
```bash
python src/data/data_prep/generate_msmarco_triples.py
``` 
Which was adopted from the the original script from the Microsoft repo:
[msmarco-doctriples.py](https://github.com/microsoft/TREC-2019-Deep-Learning/blob/master/utils/msmarco-doctriples.py)

---

Maintained with ❤️ by the **DDRO authors**
*This repo is under active development — thank you for your support!*

