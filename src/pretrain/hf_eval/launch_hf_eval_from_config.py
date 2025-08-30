


python eval_hf_docid_ranking.py \
    --hf_model_name kiyam/ddro-msmarco-tu \
    --use_fp16 \
    --per_gpu_batch_size 4 \
    --num_beams 20 \
    --docid_path ./docids.txt \
    --test_file_path ./test.json \
    --log_path ./results.log \
    --dataset_script_dir ./scripts \
    --dataset_cache_dir ./cache \
    --doc_file_path ./docs.json