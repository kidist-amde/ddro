# ⬇️ DDRO Dataset & Model Download Scripts

This folder contains scripts to download all datasets and pretrained models required for training and evaluating **Direct Document Relevance Optimization (DDRO)**.

---

## 📁 Contents

```bash
download/
├── download_msmarco_datasets.sh     # Download MS MARCO documents, qrels, queries
├── download_nq_datasets.sh          # Download Natural Questions (NQ) documents and qrels
├── download_t5_model.py             # Download T5 model/tokenizer from Hugging Face
└── README.md                        # You're here!
```

---

## 📦 Dataset Download Instructions

### 📘 MS MARCO

Downloads passage-level documents, qrels, and dev queries:

```bash
bash download_msmarco_datasets.sh
```

---

### 📗 Natural Questions (NQ)

Downloads preprocessed NQ documents and relevance annotations:

```bash
bash download_nq_datasets.sh
```

---

## 🧠 Pretrained T5 Model

Download a t5-base pretrained model and tokenizer locally:
```bash
python download_t5_model.py
```

---

## 📂 Expected Directory Structure

After downloading, your project structure should include:

```
resources/
├── datasets/
│   └── raw/
│       ├── msmarco-data/     
│       └── nq-data/          
└── transformer_models/
    └── t5-base/              
```

> ✅ Make sure scripts point to these exact paths.

---

## ☁️ Optional: Google Cloud SDK for GCS Downloads

Some MS MARCO files are hosted on `gs://` buckets. To use `gsutil`:

### Install Google Cloud SDK (Linux Example)

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-439.0.0-linux-x86_64.tar.gz

# Extract and install
tar -xf google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Activate and initialize
source ~/.bashrc  # or ~/.zshrc
gcloud init
```

### Verify installation:

```bash
gcloud version
gsutil version
```

---

Maintained with ❤️ by the DDRO authors.
