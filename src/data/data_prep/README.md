# 🧱 DDRO Data Preparation & Instance Generation

This directory contains unified scripts for preparing and transforming data for **Direct Document Relevance Optimization (DDRO)**. It covers:

* Preprocessing MS MARCO and Natural Questions datasets
* Generating dense document embeddings
* DocID encoding (URL_TITLE, PQ)
* Creating training and evaluation instances

---


## 📁 Layout — `src/data/`

This folder contains all core scripts for preprocessing datasets, generating document embeddings, encoding docids, and preparing training/evaluation instances for DDRO.

```
src/data/
├─ data_prep/
│  ├─ bm25_negative_sampling_msmarco.py      # BM25 hard negatives for MS MARCO
│  ├─ build_t5_data/
│  │  ├─ generate_train_instances.py         # Pretrain / pseudoquery / finetune builders
│  │  ├─ generate_eval_instances.py          # Eval builders for SFT / DDRO
│  │  ├─ gen_train_data_pipline.py           # Wrapper: full training-data build
│  │  └─ gen_eval_data_pipline.py            # Wrapper: full eval-data build
│  ├─ convert_tsv_to_json_array.py           # MS MARCO .tsv → JSON array (see notes)
│  ├─ generate_doc_embeddings.py             # Dense doc embeddings (e.g., for PQ)
│  ├─ generate_encoded_docids.py             # DocID encoders (PQ, URL‑title, …)
│  ├─ generate_msmarco_triples.py            # (q, pos, neg) triples for MS MARCO
│  ├─ generate_pseudo_queries.py             # docTTTTTquery‑style pseudo‑queries
│  ├─ negative_sampling.py                   # Generic negative sampling utilities
│  ├─ nq/
│  │  ├─ bm25_negative_Sampling_NQ.py        # BM25 hard negatives for NQ
│  │  ├─ convert_json_array_to_jsonl.ipynb   # JSON array → JSONL helper
│  │  ├─ convert_nq_to_msmarco_format.py     # Map NQ → MS MARCO‑like schema (optional)
│  │  └─ process_nq_dataset.py               # Clean / flatten / merge NQ
│  ├─ pq_docid_demo.ipynb                    # PQ encoding demo
│  ├─ rq_docid_demo.ipynb                    # Ranking‑quality docID demo
│  └─ url_title_docid_demo.ipynb             # URL+title docID demo
└─ data_scripts/
   ├─ csv_builder.py                         # Helpers used by builders
   └─ json_builder.py                        # Helpers used by builders
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

### ⚙️ Step 1: Generate Training Data by Stage

**Option 1: Use the batch script (recommended for SLURM clusters):**
```bash
sbatch src/scripts/preprocess/generate_3stage_train_data.sh
```

**Option 2: Run individual commands directly:**

Make sure to specify your encoding type (`url_title` or `pq`) for each stage:

**Stage 1 - General Pretraining:**
```bash
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data general_pretrain --encoding "url_title"
```

**Stage 2 - Search Pretraining:**
```bash
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data search_pretrain --encoding "url_title"
```

**Stage 3 - Fine-tuning:**
```bash
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data finetune --encoding "url_title"
```

⚠️ **Important**: Make sure to use the same `--encoding` type across all stages. Change `"url_title"` to `"pq"` if using PQ encoding.

---

### ⚙️ Step 2: Generate Evaluation Data

**Option 1: Use the batch script (recommended for SLURM clusters):**
```bash
sbatch src/scripts/preprocess/generate_eval_data.sh
```

**Option 2: Run the command directly:**
```bash
python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py --encoding "url_title"
```

⚠️ **Important**: Use the same `--encoding` type as used in Step 1. Change `"url_title"` to `"pq"` if using PQ encoding.

---

### 📁 Output Structure

**Training Data Outputs:**
* `resources/datasets/processed/msmarco-data/train_data_top_300k/`
  * `pretrain.t5_128_10.{encoding}.json` — General pretraining data (passages + sampled terms + enhanced docids)
  * `search_pretrain.t5_128_10.{encoding}.json` — Search pretraining data (pseudo queries)
  * `finetune.t5_128_1.{encoding}.json` — Fine-tuning data (real queries from qrels)

**Evaluation Data Outputs:**
* `resources/datasets/processed/msmarco-data/eval_data_top_300k/`
  * `query_dev.{encoding}.jsonl` — Development evaluation data

---

### 🔧 Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--encoding` | Document ID encoding method | `pq`, `url_title` |
| `--cur_data` | Training stage to generate | `general_pretrain`, `search_pretrain`, `finetune` |
| `--max_seq_length` | Maximum sequence length | Default: 128 |

---

### 📋 Data Generation Details

**General Pretraining** combines three data types:
- **Passages**: Raw document text chunked into passages → docid
- **Sampled Terms**: TF-IDF selected terms from documents → docid  
- **Enhanced DocIDs**: Document ID transformations → docid

**Search Pretraining** uses:
- **Pseudo Queries**: Artificially generated queries → docid

**Fine-tuning** uses:
- **Real Queries**: Actual user queries from qrels → docid

All data is formatted as `input → docid` pairs for training the model to map inputs to document identifiers.

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

