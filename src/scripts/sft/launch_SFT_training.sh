#!/bin/sh

#  The --encoding flag supports formats like pq, url, atomic, summary.

# Run DDRO training pipeline
python utils/run_training_pipeline.py \
    --encoding url 
    --scale top_300k
