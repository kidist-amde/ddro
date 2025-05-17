# 🧱 DDRO Data Preparation & Instance Generation

This directory contains **unified scripts** for preparing and transforming data for **Direct Document Relevance Optimization (DDRO)**. It includes:

* Raw data preprocessing (MS MARCO, Natural Questions)
* Dense embedding generation
* DocID encoding (URL, PQ, Atomic)
* Training & evaluation instance creation

---

## 📁 Available Scripts

### 📦 Dataset Preprocessing
```bash
ddro/dataprepr/
├── negative_sampling.py                # BM25-based hard negative sampling for MS MARCO or NQ
├── generate_doc_embeddings.py          # Compute dense GTR-T5 embeddings for documents (MS MARCO, NQ)
├── sample_msmarco_dataset.py           # Create top/random subsets of MSMARCO based on relevance frequency
├── convert_tsv_to_json_array.py        # Convert MSMARCO TSV docs to a flat JSON array
├── convert_nq_to_msmarco_format.py     # Reformat NQ data into MSMARCO-style queries and qrels
├── process_nq_dataset.py               # Extract and clean fields from raw NQ into structured format
├── create_nq_triples.py                # Generate BM25-based training triples for NQ
└── README.md                           # You are here
```

### 📄 Instance Generation

```bash
src/data/dataprep/
├── generate_encoded_docids.py            # Encode docids for retrieval
├── generate_eval_data_wrapper.py         # Entry point for evaluation data
├── generate_eval_instances.py            # Core script for eval generation
├── generate_train_data_wrapper.py        # Entry point for training data
├── generate_train_instances.py           # Core script for training generation
├── nq_doc2query_query_generator.py       # Generate pseudo queries using doc2query-T5
├── url_title_docid_demo.ipynb            # Demo: URL+title-based docid encoding
├── pq_docid_demo.ipynb                   # Demo: Product Quantization (PQ) docid encoding
```


---

## 📊 Data Preparation

DDRO is evaluated on both **MS MARCO** and **Natural Questions (NQ)** datasets.

### ✅ MS MARCO: Sample Top-300K Subset

```bash
bash scripts/preprocess/sample_top_docs.sh
```

📌 Generates: `resources/datasets/processed/msmarco-docs-sents.top.300k.json`
(JSONL format, sentence-tokenized, ranked by qrels frequency)

---

### 🔢 DocID Representations

You can download document ID representations for both datasets used in this paper from our 🤗 Hugging Face repo:
👉🏽 [ddro-docids](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5)

Place them under:

```
resources/datasets/processed/msmarco-data/encoded_docid/
```

But you can also generate them locally. To generate **URL-based docids**, run:

```bash
bash ddro/src/scripts/preprocess/generate_msmarco_encoded_ids.sh
```

Example `url_docid` format:

```
[d108472] 594,858,7,17,4624,5,287,1
[d1842]   3,89,9,1824,3105,4440,...,1677,1
```

---

### 🧠 Generating PQ DocIDs

To generate **PQ-based docids**, you must first compute document embeddings using a `GTR-T5` encoder (e.g., `sentence-transformers/gtr-t5-base`):

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

Then generate the document IDs and training instances:

```bash
bash scripts/preprocess/compute_nq_t5_embeddings.sh   # Compute document embeddings for NQ using TR-T5 (required for PQ ID assignment)
bash scripts/preprocess/generate_nq_encoded_ids.sh    # Generate and save encoded document IDs (URL, PQ, Atomic, Summary formats)
```

---

### ✏️  Pseudo Query Generation

We generate pseudo queries using the [**DocTTTTTQuery**](https://github.com/castorini/docTTTTTquery) framework.
The model is fine-tuned separately for each dataset (MS MARCO and NQ) to better match their domain.

#### 🛠️ Fine-tuning the Model

To fine-tune the DocTTTTTQuery model on the NQ dataset, run:

```bash
bash ./scripts/run_finetune_docTTTTTquery.sh
```

#### 🧾 Query Generation Script

Once fine-tuned, use the following script to generate pseudo queries for each document:

```python
python ./src/data/dataprep/doc2query_query_generator.py
```

This script will generate 10 queries per document and output them in the expected format.

#### 📥 Pre-generated Queries

You can also download pre-generated pseudo queries from our 🤗 [Hugging Face collection](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5) and place them in:

```text
ddro/resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt
ddro/resources/datasets/processed/nq-data/nq_pseudo_query_10.txt
```

#### 🧾 Output Format

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


## 🛠 Training Instance Generation

To train the reference model (**Phase 1: Supervised Fine-Tuning (SFT)**), we generate three types of **supervised signal instances** for next-token prediction training across three stages:

### 🎯 Training Stages

1. **General Pretraining**
   Train the model to predict `docid` from **document content**
   → `doc → docid`

2. **Search Pretraining**
   Train the model to predict `docid` from **pseudo queries**
   → `pseudoquery → docid`

3. **Finetuning**
   Train the model using **real queries paired with gold documents**
   → `query → docid` (from QRELs)

Each stage aligns with a different training signal to gradually improve the model's relevance generation.

---

### ⚙️ Script: `generate_train_data_wrapper.py`

This script generate training instances for all three stages by specifying the `--cur_data` flag:

| `--cur_data` Option | Input Source         | Description                       |
| ------------------- | -------------------- | --------------------------------- |
| `general_pretrain`  | Document contents    | For unsupervised content modeling |
| `search_pretrain`   | Pseudo queries       | For intermediate search alignment |
| `finetune`          | Real queries + qrels | For final relevance optimization  |

---

### 🚀 Generate Instances

Use the following scripts to generate both **training and evaluation** instances for each dataset. These scripts internally call the wrapper and handle all necessary dataset paths and output configurations.

#### 📘 For **MS MARCO**:

Run the following command to generate data for **both training and evaluation**:

```bash
bash ./src/scripts/preprocess/generate_msmarco_eval_and_train_data.sh
```

#### 📗 For **Natural Questions (NQ)**:

```bash
bash ddro/src/scripts/preprocess/generate_nq_eval_and_train_data.sh
```

> These scripts prepare inputs for all three training stages: `general_pretrain`, `search_pretrain`, and `finetune`.
> As well as, they generate evaluation data required for testing the model.

---


Maintained with ❤️ by the DDRO authors.
