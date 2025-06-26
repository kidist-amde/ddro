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

## ☁️ (Prerequisite) Google Cloud SDK for `gs://` Downloads

To download the Natural Questions (NQ) dataset, you need to have the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and gsutil available in your environment.


### Install Google Cloud SDK (Linux example)
One-time setup (if not already installed):

```bash
# Download and install the SDK
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-460.0.0-linux-x86_64.tar.gz
tar -xvzf google-cloud-sdk-460.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
```

> 📌 When prompted, allow the installer to update your `.bashrc` or `.zshrc`.

### 🔄 Initialize and authenticate

```bash
# Reload your shell and initialize

source ~/.bashrc  # or ~/.zshrc
gcloud init
```

> 📌 When prompted during gcloud init, you can select any existing project (no billing required for public downloads).

### Verify installation

```bash
gcloud --version
gsutil --version
```

You can now run:

```bash
bash ./src/data/download/download_nq_datasets.sh
```

---

© 2025 Kidist Amde Mekonnen Made with ❤️ at IRLab, University of Amsterdam.

