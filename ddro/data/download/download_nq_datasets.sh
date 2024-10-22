#!/bin/bash

# Googel Natural Questions dataset (NQ-dataset)
# (https://ai.google.com/research/NaturalQuestions/download)

mkdir -p resources/datasets/raw/nq-data
cd resources/datasets/raw/nq-data

# gcloud auth login
gsutil cp gs://natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz .
gsutil cp gs://natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz .



