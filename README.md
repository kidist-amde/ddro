# DDRO: Direct Document Relevance Optimization for for Generative Information Retrieval

This repository contains the official implementation of our SIGIR 2025 paper:  
📄 **[Lightweight and Direct Document Relevance Optimization for Generative IR (DDRO)](https://arxiv.org/abs/2504.05181)**

---

## 🤖 Motivation

**Misalignment in Learning Objectives:**  
Gen-IR models are typically trained via next-token prediction (cross-entropy loss) over docids.  
While effective for language modeling, this objective:
- 🎯 Optimizes **token-level generation**
- ❌ Not designed for **document-level ranking**

As a result, Gen-IR models are not directly optimized for **learning-to-rank**, which is the core requirement in IR systems.

---

## 🎯 What DDRO Does

In this work, we ask:

> _How can Gen-IR models directly learn to rank documents, instead of just predicting the next token?_

We propose **DDRO**:  
**Lightweight and Direct Document Relevance Optimization for Gen-IR**

### ✅ Key Contributions:
- Aligns training objective with ranking by using **pairwise preference learning**
- Trains the model to **prefer relevant documents over non-relevant ones**
- Bridges the gap between **autoregressive training** and **ranking-based optimization**
- Requires **no reinforcement learning or reward modeling**

---
<img src="src/arc_images/DDRO.drawio.png" alt="DDRO Image" width="600"/>




### 🧠 Learning Objectives in DDRO

We optimize DDRO in two phases:

---

#### 📘 Phase 1: Supervised Fine-Tuning (SFT)

Learn to generate the correct **docid** sequence given a query by minimizing the autoregressive token-level cross-entropy loss:
<!-- 
$$
\mathcal{L}_{\text{SFT}} = -\sum \log p_\theta(\text{docid}_i \mid \text{docid}_{<i}, q)
$$ -->

 - <img src="src/arc_images/loss_ntp.png" alt="DDRO Image" width="300"/>


Maximize the likelihood of generating the correct docid given a query:

<!-- $$
\boxed{
\max_{\pi} \,\, \mathbb{E}_{(q, \text{docid}) \sim \mathcal{D}} \left[
\log \pi(\text{docid} \mid q)
\right]
}
$$ -->

 - <img src="src/arc_images/objective_ntp.png" alt="DDRO Image" width="300"/>
---

#### 📗 Phase 2: Pairwise Ranking Optimization (DDRO Loss)

This phase improves the **ranking quality** of generated document identifiers by applying a **pairwise learning-to-rank objective** inspired by **Direct Preference Optimization (DPO)**.

📄 *Rafailov et al., 2023 — [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)*

<!-- $$
\mathcal{L}_{\text{DDRO}}(\pi_\theta; \pi^{\text{ref}}) = - \mathbb{E}_{(q, \text{docid}^+, \text{docid}^-) \sim \mathcal{D}} 
\left[
\log \sigma \left(
\beta \log \frac{\pi_\theta(\text{docid}^+ \mid q)}{\pi^{\text{ref}}(\text{docid}^+ \mid q)} -
\beta \log \frac{\pi_\theta(\text{docid}^- \mid q)}{\pi^{\text{ref}}(\text{docid}^- \mid q)}
\right)
\right]
$$ -->
 - <img src="src/arc_images/dpo_loss.png" alt="DDRO Image" width="800"/>

### 📖 Description

This **Direct Document Relevance Optimization (DDRO)** loss guides the model to **prefer relevant documents (`docid⁺`) over less relevant ones (`docid⁻`)** by comparing how both the current model and a frozen reference model score each document:

* `docid⁺`: A relevant document for the query `q`
* `docid⁻`: A non-relevant or less relevant document
* $\pi_\theta$: The current model being optimized
* $\pi^{\text{ref}}$: A frozen reference model (typically trained with SFT in Phase 1)
* $\beta$: A temperature-like scaling factor to control sensitivity
* $\sigma$: Sigmoid function, to map scores to \[0,1] preference space

Encourage the model to rank relevant docid⁺ higher than non-relevant docid⁻  :


<!-- $$
\boxed{
\max_{\pi} \,\, \mathbb{E}_{(q, \text{docid}^+, \text{docid}^-) \sim \mathcal{D}} \left[
\log \sigma \left(
\beta \log \frac{\pi(\text{docid}^+ \mid q)}{\pi_{\text{ref}}(\text{docid}^+ \mid q)} -
\beta \log \frac{\pi(\text{docid}^- \mid q)}{\pi_{\text{ref}}(\text{docid}^- \mid q)}
\right)
\right]
}
$$ -->
 - <img src="src/arc_images/dpo_objective.png" alt="DDRO Image" width="500"/>

### ✅ Usage

This loss is used **after** the SFT phase to **fine-tune the ranking behavior** of the model. Instead of just generating `docid`, the model now **learns to rank `docid⁺` higher than `docid⁻`** in a preference-aligned manner.



---

### ✅ Why It Works

- Directly **encourages higher generation scores for relevant documents**
- Uses **contrastive ranking** rather than token-level generation
- Avoids reward modeling or RL while remaining efficient and scalable


---
---

### 💡 Why DDRO is Different from Standard DPO

While our optimization is inspired by the DPO framework [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290), its adaptation to **Generative Information Retrieval (GenIR)** is **non-trivial**:

- In contrast to open-ended preference alignment, our task involves **structured docid generation** under **beam decoding constraints**
- Our model uses an **encoder-decoder** architecture rather than decoder-only
- The objective is **document-level ranking**, not open-ended preference generation

This required **novel integration** of preference optimization into **retrieval-specific pipelines**, making DDRO uniquely suited for GenIR.



## 📁 Project Structure

```bash
src/
├── data/                # Data downloading, preprocessing, and docid instance generation
├── pretrain/            # DDRO model training and evaluation logic (incl. ddro)
├── scripts/             # Entry-point shell scripts for SFT, ddro, BM25, and preprocessing
├── utils/               # Core utilities (tokenization, trie, metrics, trainers)
├── ddro.yml             # Conda environment (for training DDRO)
├── pyserini.yml         # Conda environment (for BM25 retrieval with Pyserini)
├── README.md            # You're here!
└── requirements.txt     # Additional Python dependencies
```
<h5><span style="color:Yellow;">➡️ Each subdirectory includes a detailed README.md with instructions.</span></h5>

---

## 🛠️ Setup & Dependencies
### 1. Install Environment

Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/kidist-amde/ddro.git
   conda env create -f ddro.yml
   conda activate ddro
   ```

### 2. Download Datasets and Pretrained Model
We use MS MARCO document (top-300k) and Natural Questions (NQ-320k) datasets, and a pretrained T5 model.


   ```bash
   bash   ./src/data/download/download_msmarco_datasets.sh
   bash   ./src/data/download/download_nq_datasets.sh
   python ./src/data/download/download_t5_model.py
   ```
📂 For details and download links, refer to: [src/data/download/README.md](https://github.com/kidist-amde/ddro/tree/main/src/data/download#readme)


### 3. Expected Directory Structure
Once downloaded, your resources/ directory should look like this:
   ```
   resources/
   ├── datasets/
   │   ├── msmarco-data/
   │   └── nq-data/
   └── transformer_models/
       └── t5-base/
   ```

---

## Data Preparation
DDRO evaluated both on **Natural Questions (NQ)** and **MS MARCO** datasets. 

We provide preprocessing scripts for:
- Cleaning and formatting datasets
- Generating dense document embeddings
- Encoding documents into discrete docid representations (e.g., PQ, URL ids, etc ..)
- Creating training and evaluation instances

---

### 📚 Natural Questions & MS MARCO

To prepare both datasets for training and evaluation:

➡️ See: [`src/data/preprocessing/README.md`](https://github.com/kidist-amde/ddro/tree/main/src/data/preprocessing#readme)  

---

## 🔁 Training Pipeline

(Phase 1) We first train a **Supervised Fine-Tuning (SFT) model** using **next-token prediction** across three stages:

1. **Pretraining** on document content (`doc → docid`)
2. **Search Pretraining** on pseudo queries (`pseudoquery → docid`)
3. **Finetuning** on real queries using supervised pairs from qrels (with gold docids) (`query → docid`)

This results in a **seed model** trained to autoregressively generate document identifiers.


You can run all stages with a single command:

```bash
python utils/run_training_pipeline.py --encoding pq
```
📍 The --encoding flag supports formats like pq, url, atomic, summary.

➡️ For detailed instructions, configuration options, and individual stage execution: See [pretrain/README.md](https://github.com/kidist-amde/ddro/blob/main/src/pretrain/README.md)

---

## 🔍 BM25 Retrieval Setup (via Pyserini)

We use BM25 for two key purposes in DDRO:

1. **Sparse Baseline Comparison**  
   BM25 serves as a strong term-based baseline in our experiments, allowing comparison against dense and generative retrievers.

2. **Negative Sampling for Training**  
   BM25 results are used to sample hard negatives for training contrastive and pairwise objectives.

---

### ⚙️ Setup Instructions

To run BM25 retrieval using Pyserini:

```bash
conda env create -f pyserini.yml
conda activate pyserini
pip install -r pyserini.txt
```

Then index and retrieve with:
```bash
bash scripts/bm25/run_bm25_retrieval_nq.sh
bash scripts/bm25/run_bm25_retrieval_msmarco.sh
```

---

## DDRO Training (Phase 2: Pairwise Optimization)

After training the SFT model (Phase 1), we apply **Phase 2: Direct Document Relevance Optimization**, which fine-tunes the model using a **pairwise ranking objective**.

This stage optimizes the model to **prefer relevant documents over non-relevant ones**, bridging the gap between autoregressive generation and ranking-based retrieval.

We implement this using [Hugging Face’s `DPOTrainer`](https://github.com/huggingface/trl), adapted for document ID generation.

➡️ Scripts are available in: [`scripts/ddro/`](./src/scripts/ddro/)

Run training and evaluation:

```bash
bash scripts/ddro/run_ddro_training.sh
bash scripts/ddro/run_test_ddro.sh

```

---


📂 Evaluation logs and metrics are saved to:
```
logs/
outputs/
```
###### ➡️ For configuration options and example training pairs, see: pretrain/train_ddro_encoder_decoder.py
---

---

## 📚 Datasets Used

We evaluate DDRO on two standard retrieval benchmarks:

- 📘 [MS MARCO Document Ranking](https://microsoft.github.io/msmarco/)
- 📗 [Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions)

---

## 📂 Preprocessed Data & Model Checkpoints (Hugging Face 🤗)

We release all training resources to support reproducibility:

### 🧾 Document DocID Encodings
- 👉🏽 [ddro-docids](https://huggingface.co/datasets/kiyam/ddro-docids)  
  Encoded docid representations (PQ, URL, Atomic, etc.)

### ❓ Pseudo Queries (for Search Pretraining)
- 👉🏽 [ddro-pseudo-queries](https://huggingface.co/datasets/kiyam/ddro-pseudo-queries)  
  Synthetic queries generated using DocT5Query.

### 🧠 Model Checkpoints
- 👉🏽 [DDRO Generative IR Collection](https://huggingface.co/collections/kiyam/ddro-generative-document-retrieval-680f63f2e9a72033598461c5)  
  Includes models trained on MS MARCO and NQ with both PQ and TU encoding strategies.

### 📄 Preprocessed MS MARCO Subset
- 👉🏽 [Top-300K MS MARCO Corpus](https://huggingface.co/datasets/kiyam/ddro-msmarco-doc-dataset-300k)

---

## 🙏 Acknowledgments

We gratefully acknowledge the following open-source projects:

- [ULTRON](https://github.com/smallporridge/WebUltron)
- [HuggingFace TRL](https://github.com/huggingface/trl)
- [NCI (Neural Corpus Indexer)](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)
- [docTTTTTquery](https://github.com/castorini/docTTTTTquery)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 📌 Citation

```bibtex
@article{mekonnen2025lightweight,
  title={Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval},
  author={Mekonnen, Kidist Amde and Tang, Yubao and de Rijke, Maarten},
  journal={arXiv preprint arXiv:2504.05181},
  year={2025}
}
```
---

## 📬 Contact

For questions, please open an [issue](https://github.com/kidist-amde/DDRO-Direct-Document-Relevance-Optimization/issues).


