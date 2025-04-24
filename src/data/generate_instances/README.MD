# DDRO Data Generation Scripts

This module provides all the necessary scripts for generating training and evaluation instances for **Direct Document Relevance Optimization (DDRO)**, supporting datasets like MS MARCO and Natural Questions (NQ).

## 📂 Structure

```bash
ddro/data/generate_instances/
├── generate_encoded_docids.py            # Encode document IDs for retrieval
├── generate_eval_data_wrapper.py         # Wrapper for generating evaluation instances
├── generate_eval_instances.py            # Core script for eval data generation
├── generate_train_data_wrapper.py        # Wrapper for generating & merging training instances
├── generate_train_instances.py           # Core script for training data generation
├── nq_doc2query_query_generator.py       # Generate pseudo-queries using doc2query-T5
├── url_title_docid_demo.ipynb            # Demo: Generate doc IDs using URLs and titles
├── pq_docid_demo.ipynb                   # Demo: Encode and decode doc IDs with Product Quantization
└── README.md                              # You are here
```

## 🔧 Training Data Generation

Use `generate_train_data_wrapper.py` to create pretraining or fine-tuning data. It internally calls `generate_train_instances.py` based on `--cur_data`:

### Modes:
- `general_pretrain` → Generates: `passage`, `sampled_terms`, `enhanced_docid`
- `search_pretrain`  → Generates: `synthetic_query`
- `finetune`         → Generates: `query`

### Example:
```bash
python generate_train_data_wrapper.py \
  --cur_data general_pretrain \
  --scale top_300k \
  --encoding url
```

This produces:
- Merged file: `train_data_top_300k/general_pretrain.t5_128_10.url.300k.json`
- Uses: `generate_train_instances.py`

---

##  Evaluation Data Generation

Use `generate_eval_data_wrapper.py` to generate dev/test evaluation instances. It calls `generate_eval_instances.py`.

```bash
python generate_eval_data_wrapper.py \
  --scale top_300k \
  --encoding url
```

Produces:
- File: `test_data_top_300k/query_dev.t5_128_1.url.300k.json`

---

##  Document ID Encoding

Encode document IDs (needed before training/eval):
```bash
python generate_encoded_docids.py
```
This generates a mapping of `[docid] → token_id`.

---

##  Pseudo-Query Generation 
Use `nq_doc2query_query_generator.py` to generate queries using a `doc2query-T5` model:
```bash
python nq_doc2query_query_generator.py \
  --input_file path/to/docs.jsonl \
  --checkpoint_path castorini/doc2query-t5-large-msmarco \
  --output_path output/queries.json
```

---

## Shared Requirements
- Python ≥ 3.8
- `transformers`, `tqdm`, `torch`, `numpy`

```bash
pip install -r requirements.txt
```

---

##  Notes
- Document IDs are wrapped as `[docid]` tokens in vocabulary
- Tokenizers are extended for domain-specific document IDs
- Outputs are in `.json` or `.jsonl` formats (one instance per line)

---


---

Maintained with ❤️ by the DDRO authors.
