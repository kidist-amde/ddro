

# 🧱 DDRO Data Preparation & Instance Generation

This directory contains unified scripts for preparing and transforming data for **Direct Document Relevance Optimization (DDRO)**. It covers:

* Preprocessing MS MARCO and Natural Questions datasets
* Generating dense document embeddings
* DocID encoding (URL, PQ, Atomic)
* Creating training and evaluation instances

---

## 📁 Available Scripts

### Dataset Preprocessing

```bash
dataprep/
├── negative_sampling.py                # BM25-based hard negative sampling (MS MARCO, NQ)
├── generate_doc_embeddings.py          # Generate dense GTR-T5 embeddings
├── sample_msmarco_dataset.py           # Create top/random MS MARCO subsets
├── convert_tsv_to_json_array.py        # Convert MS MARCO TSV to flat JSON
├── convert_nq_to_msmarco_format.py     # Convert NQ to MS MARCO-style format
├── process_nq_dataset.py               # Clean and extract NQ fields
├── create_nq_triples.py                # Create BM25-based NQ triplets
└── README.md                           # You're here!
```

### Instance Generation

```bash
dataprep/
├── generate_encoded_docids.py          # Encode documents into docid formats
├── generate_eval_data_wrapper.py       # Entry for evaluation data generation
├── generate_eval_instances.py          # Core script for eval instance creation
├── generate_train_data_wrapper.py      # Entry for training data generation
├── generate_train_instances.py         # Core script for training instances
├── nq_doc2query_query_generator.py     # Generate pseudo queries via doc2query-T5
├── url_title_docid_demo.ipynb          # URL+title docid encoding demo
├── pq_docid_demo.ipynb                 # Product Quantization (PQ) docid demo
```

---

## 📊 Data Preparation

DDRO supports both **MS MARCO** and **Natural Questions (NQ)** benchmarks.

### ✅ MS MARCO: Sample Top-300K Subset

```bash
bash scripts/preprocess/sample_top_docs.sh
```

📌 Generates: `resources/datasets/processed/msmarco-docs-sents.top.300k.json`
(JSONL format, sentence-tokenized, ranked by qrels frequency)

---

### 🔢 DocID Representations

You can either **generate** docid representations locally or **download** pre-encoded files from 🤗 [Hugging Face](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5):

Place them under:

```bash
resources/datasets/processed/msmarco-data/encoded_docid/
```

Example `url_docid` format:

```
[d108472] 594,858,7,17,4624,5,287,1
[d1842]   3,89,9,1824,3105,4440,...,1677,1
```

---

To generate **PQ docids**, first compute document embeddings:

```bash
sbatch scripts/preprocess/generate_msmarco_t5_embeddings.sh
sbatch scripts/preprocess/generate_msmarco_encoded_ids.sh
```

Example `pq_docid` format:

```
[d108472] 32211,32518,32782,33144,33382,...
[d1842]   32177,32471,32844,33053,33163,...
```

---
### ✅ Natural Questions

To prepare NQ data for training and evaluation, run:

```bash
bash scripts/preprocess/preprocess_nq_dataset.sh               # Cleans and merges NQ
bash scripts/preprocess/convert_nq_to_msmarco_format.sh        # Converts to MS MARCO-style format
```
First, convert the original NQ data to match the MS MARCO query-passage format:

```bash
bash src/scripts/preprocess/convert_nq_to_msmarco_format.sh
```

Then generate the document IDs and training instances:

```bash
bash scripts/preprocess/compute_nq_t5_embeddings.sh   # Compute document embeddings for NQ using TR-T5 (required for PQ ID assignment)
bash scripts/preprocess/generate_nq_encoded_ids.sh    # Generate and save encoded document IDs (URL, PQ, Atomic, Summary formats)
```

---
### ✏️ Pseudo Query Generation (DocTTTTTQuery)

To fine-tune on NQ:

```bash
bash ./scripts/run_finetune_docTTTTTquery.sh
```

To generate queries:

```bash
python ./src/data/dataprep/doc2query_query_generator.py
```

**Or download pre-generated queries from Hugging Face:**
📥 [DDRO HF Collection](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5)


Save under:

```bash
ddro/resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt
ddro/resources/datasets/processed/nq-data/nq_pseudo_query_10.txt
```

Expected format:


Each line should contain a document ID and one of its generated queries, tab-separated:

```
[d301595]	what age should a child know what they like to eat
[d301595]	what age should a child know what they like to do
[d301595]	what is the average age for a child to be independent
[d301595]	what age should a child start writing letters
[d301595]	what age should a child start playing with themselves
[d301595]	what age should a child know what they like

...
```

Your section is already clear and well-structured! Here's a lightly polished version to ensure consistency, flow, and professional tone—just a few minor edits to tighten phrasing, clarify script purposes, and improve formatting:

---

## 🛠 Training Instance Generation

To train the reference model (**Phase 1: Supervised Fine-Tuning, SFT**), we generate three types of **supervised instances** for next-token prediction across the following stages:

### 🎯 Training Stages

1. **General Pretraining**
   Train on raw document content to predict `docid`:
   → `doc → docid`

2. **Search Pretraining**
   Train on synthetic pseudo queries to predict `docid`:
   → `pseudoquery → docid`

3. **Finetuning**
   Train on real queries paired with gold documents from qrels:
   → `query → docid`

Each stage provides a progressively stronger retrieval signal to guide relevance generation.

---

### ⚙️ Script: `generate_train_data_wrapper.py`

This script supports all three training stages via the `--cur_data` flag:

| `--cur_data` Value | Input Source         | Purpose                        |
| ------------------ | -------------------- | ------------------------------ |
| `general_pretrain` | Document contents    | General language understanding |
| `search_pretrain`  | Pseudo queries       | Search-aligned supervision     |
| `finetune`         | Real queries + qrels | Final relevance optimization   |

---

### 🚀 Generating Training + Evaluation Data

Use the following entry-point scripts to generate both **training** and **evaluation** instances for MS MARCO and NQ. These scripts internally invoke the wrapper and handle all required paths and configurations.

#### 📘 MS MARCO

```bash
bash ./src/scripts/preprocess/generate_msmarco_eval_and_train_data.sh
```

#### 📗 Natural Questions (NQ)

```bash
bash ./src/scripts/preprocess/generate_nq_eval_and_train_data.sh
```

> These scripts automatically generate data for all three stages (`general_pretrain`, `search_pretrain`, and `finetune`) as well as the corresponding evaluation files.

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

You may also adapt the script to use your **own BM25 ranking outputs**.

---

Maintained with ❤️ by the DDRO authors.
