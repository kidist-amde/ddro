#!/bin/sh
#SBATCH --job-name=gen_train_data_raw
#SBATCH --time=1-00:00:00
#SBATCH --mem=128gb
#SBATCH -c 48
#SBATCH --output=logs-slurm/gen_train_and_eval_data_PQID_raw-%j.out

set -e  # Exit on first error

source /home/kmekonn/.bashrc
conda activate ddro_env

# === Path config ===
MODEL_PATH="t5-base"
DOC_PATH="resources/datasets/processed/msmarco-docs-sents.top.300k.json.gz"
DOCID_PATH="resources/datasets/processed/msmarco-data/encoded_docid/pq_docid.txt"          # target docid (label)
SOURCE_DOCID_PATH="resources/datasets/processed/msmarco-data/encoded_docid/url_docid.txt"  # source docid
QUERY_PATH="resources/datasets/raw/msmarco-data/msmarco-doctrain-queries.tsv.gz"
QUERY_PATH_DEV="resources/datasets/raw/msmarco-data/msmarco-docdev-queries.tsv.gz"
QRELS_PATH="resources/datasets/raw/msmarco-data/msmarco-doctrain-qrels.tsv.gz"
QRELS_PATH_DEV="resources/datasets/raw/msmarco-data/msmarco-docdev-qrels.tsv.gz"
FAKE_QUERY_PATH="/ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro/resources/datasets/processed/msmarco-data/msmarco_pseudo_query_10.txt"
OUTPUT_BASE="resources/datasets/processed/msmarco-data/train_data_top_300k"
OUTPUT_BASE_dev="resources/datasets/processed/msmarco-data/eval_data_top_300k"
SAMPLE_PER_DOC=5

mkdir -p "$OUTPUT_BASE"

# === [1/5] Passage => DocID ===
echo "[1/5] Generating passage samples..."
python src/data/data_prep/generate_train_instances.py \
  --pretrain_model_path "$MODEL_PATH" \
  --data_path "$DOC_PATH" \
  --docid_path "$DOCID_PATH" \
  --fake_query_path "$FAKE_QUERY_PATH" \
  --output_path "$OUTPUT_BASE/passage.jsonl" \
  --sample_for_one_doc $SAMPLE_PER_DOC \
  --max_seq_length 512 \
  --current_data passage

# === [2/5] TF-IDF Terms => DocID ===
echo "[2/5] Generating TF-IDF term samples..."
python src/data/data_prep/generate_train_instances.py \
  --pretrain_model_path "$MODEL_PATH" \
  --data_path "$DOC_PATH" \
  --docid_path "$DOCID_PATH" \
  --fake_query_path "$FAKE_QUERY_PATH" \
  --output_path "$OUTPUT_BASE/sampled_terms.jsonl" \
  --max_seq_length 64 \
  --current_data sampled_terms

# === [3/5] Enhanced DocID (URL => PQ) ===
echo "[3/5] Generating enhanced docid samples..."
python src/data/data_prep/generate_train_instances.py \
  --pretrain_model_path "$MODEL_PATH" \
  --data_path "$DOC_PATH" \
  --docid_path "$DOCID_PATH" \
  --source_docid_path "$SOURCE_DOCID_PATH" \
  --output_path "$OUTPUT_BASE/enhanced_docid.jsonl" \
  --max_seq_length 64 \
  --current_data enhanced_docid

# === [4/5] Fake Query => DocID ===
echo "[4/5] Generating fake query samples..."
python src/data/data_prep/generate_train_instances.py \
  --pretrain_model_path "$MODEL_PATH" \
  --data_path "$DOC_PATH" \
  --docid_path "$DOCID_PATH" \
  --fake_query_path "$FAKE_QUERY_PATH" \
  --output_path "$OUTPUT_BASE/fake_query.jsonl" \
  --max_seq_length 64 \
  --current_data fake_query

# === [5/5] Real Query => DocID ===
echo "[5/5] Generating real query samples..."
python src/data/data_prep/generate_train_instances.py \
  --pretrain_model_path "$MODEL_PATH" \
  --data_path "$DOC_PATH" \
  --docid_path "$DOCID_PATH" \
  --query_path "$QUERY_PATH" \
  --qrels_path "$QRELS_PATH" \
  --output_path "$OUTPUT_BASE/query.jsonl" \
  --max_seq_length 64 \
  --current_data query

python src/data/data_prep/generate_eval_instances.py \
  --pretrain_model_path $MODEL_PATH \
  --data_path $DOC_PATH \
  --docid_path $DOCID_PATH \
  --query_path $QUERY_PATH_DEV \
  --qrels_path $QRELS_PATH_DEV \
  --output_path $OUTPUT_BASE_dev/query_dev.jsonl \
  --max_seq_length 128 \
  --current_data query_dev



echo "=== DONE: $(date)"
