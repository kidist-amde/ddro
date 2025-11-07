#!/bin/bash


# make sure to change the encoding to either "url_title" or "pq"

echo "generating 3-stage training data"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data general_pretrain --encoding "url_title" 
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data search_pretrain --encoding "url_title"
python src/data/data_prep/build_t5_data/gen_train_data_pipline.py --cur_data finetune --encoding "url_title"
