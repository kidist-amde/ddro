# DDRO Dataset Download Scripts

This folder contains scripts to download datasets and models required for **Direct Document Relevance Optimization (DDRO)**.

---

## 📂 Contents
```bash
download/
├── download_msmarco_datasets.sh   # Download MS MARCO dataset (docs, qrels, queries)
├── download_nq_datasets.sh        # Download Natural Questions (NQ) data
├── download_t5_model.py           # Fetch T5 model/tokenizer from HuggingFace
└── README.md                      # You're here
```

---

##  Dataset Download Scripts

###  MS MARCO
Run this script to download MS MARCO passage/document-level files and qrels:
```bash
bash download_msmarco_datasets.sh
```

###  Natural Questions
Run this to download preprocessed NQ documents and supervision:
```bash
bash download_nq_datasets.sh
```

###  Pretrained T5 Model
This Python script downloads a specified HuggingFace model + tokenizer locally:
```bash
python download_t5_model.py --model_name_or_path castorini/doc2query-t5-large-msmarco --save_dir models/t5
```

---

## ☁️ Google Cloud SDK (Optional)
Some MS MARCO datasets are hosted on GCS buckets (gs://...). To download via `gsutil`, install the Google Cloud SDK:

### Installation (Linux Example)
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-439.0.0-linux-x86_64.tar.gz

# Extract and install
tar -xf google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Activate
source ~/.bashrc  # or ~/.zshrc

# Initialize
gcloud init
```

###  Verify Installation
```bash
gcloud version
gsutil version
```



---

Maintained with ❤️ by the DDRO authors.
