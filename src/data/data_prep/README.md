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

### ✏️ Pseudo Query Generation (docTTTTTquery)

To enable search pretraining, generate pseudo queries from raw documents using a `docT5query` model, producing 10 queries per document.

#### 🧪 Finetune `docTTTTTquery` on NQ or MS MARCO:

```bash
bash scripts/run_finetune_docTTTTTquery.sh
```

#### 🛠 Generate queries:

```bash
python src/data/data_prep/doc2query_query_generator.py
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

### ✅ Step 1: Generate Raw Supervision Signals

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


## 🧲 Contrastive Triplets (For Phase 2: DDRO)

After training the SFT model, we move to **Phase 2: Direct Document Relevance Optimization (DDRO)**, which fine-tunes the model with a **pairwise ranking objective** using contrastive triplets.

Each triplet contains:

* A query
* A **positive** document ID
* One or more **negative** document IDs

---

### 📦 Triplet Generation

To create these contrastive training triplets:

#### 🔹 Natural Questions (NQ)

```bash
python ddro/src/data/dataprep/create_nq_triples.py
```

---

#### 🔹 MS MARCO

1. Download the official top-100 BM25 retrievals:
   📥 [msmarco-doctrain-top100.gz](https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz)

2. Run the triplet generation script:

```bash
python src/data/dataprep/generate_msmarco_triples.py
```

> You may also adapt the script to use your **own BM25 ranking outputs**.
---

### ✅ Natural Questions (NQ)

To prepare the **Natural Questions (NQ)** dataset in MS MARCO-style format and generate training-ready encodings, follow the steps below.

---

#### 🧹 Step 1: Preprocess and Convert to MS MARCO Format

Run the following scripts to clean and reformat NQ into MS MARCO-style layout:

```bash
bash scripts/preprocess/preprocess_nq_dataset.sh               # Cleans and merges raw NQ data
bash scripts/preprocess/convert_nq_to_msmarco_format.sh        # Converts to MS MARCO-style format
```

> After generating the MS MARCO-style dataset, follow the same steps described for MS MARCO above.
> Be sure to **replace the data paths** and **set the dataset type to `nq`** in all relevant scripts.

---

#### ⚙️ Step 2: Generate Document IDs and Training Instances

Make sure that the `--dataset` or `--data_type` argument in each script is set to `"nq"`.

```bash
sbatch src/scripts/preprocess/generate_doc_embeddings.sh    # Step 1: Compute document embeddings (for PQ)
bash scripts/preprocess/generate_encoded_ids.sh             # Step 2: Generate and save encoded doc IDs (PQ, URL, etc.)
```

> 📝 Only **PQ** and **URL-based docid encodings** are reported in our paper.

---

Maintained with ❤️ by the DDRO authors.
*This repo is under active development — thanks for your patience!*

---
