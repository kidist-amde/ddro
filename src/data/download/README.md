Here’s a cleaner, final version of your sub-README that improves clarity, consistency, and tone while keeping the structure and content intact:

---

# ⬇️ DDRO Dataset & Model Download Scripts

This folder contains scripts to download all datasets and pretrained models required for training and evaluating **Direct Document Relevance Optimization (DDRO)**.

---

## 📁 Contents

```bash
download/
├── download_msmarco_datasets.sh     # Download MS MARCO documents, qrels, and queries
├── download_nq_datasets.sh          # Download Natural Questions (NQ) documents and qrels
├── download_t5_model.py             # Download the T5-base model and tokenizer
└── README.md                        # You're here!
```

---

## 📦 Dataset Download Instructions

### MS MARCO

Downloads passage-level documents, queries, and qrels:

```bash
bash download_msmarco_datasets.sh
```

---

### Natural Questions (NQ)

Downloads preprocessed NQ documents and relevance annotations:

```bash
bash download_nq_datasets.sh
```

---

## 🧠 Pretrained T5 Model

Downloads the `t5-base` model and tokenizer from Hugging Face:

```bash
python download_t5_model.py
```

---

## 📂 Expected Directory Structure

After running the scripts, your file structure should look like:

```bash
resources/
├── datasets/
│   └── raw/
│       ├── msmarco-data/
│       └── nq-data/
└── transformer_models/
    └── t5-base/
```

> ✅ Ensure all paths match the expected structure for downstream scripts to work properly.

---

## ☁️ (Optional) Google Cloud SDK for `gs://` Downloads

Some MS MARCO files are hosted on Google Cloud Storage. To access them:

### Install Google Cloud SDK (Linux example)

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

Then initialize:

```bash
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
