#!/bin/sh


echo "Generating evaluation data..."

python src/data/data_prep/build_t5_data/gen_eval_data_pipline.py --encoding "url_title"

