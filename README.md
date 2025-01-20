### Data Preparation

Download MS MARCO and Natural Questions datasets using the provided shell script.

#### Prerequisites
Ensure you have the following tools installed:
- `wget` for MS MARCO
- `[gcloud` and `gsutil`]() for Natural Questions 
- Addtionaly Download and store  T5-base model in `./resources/transformer_models/t5-base/`

#### Instructions
Run the script to download the datasets and the model:
```bash
bash ./data/download/download_msmarco_datasets.sh
bash ./data/download/download_nq_datasets.sh
python ./data/download/download_t5_model.py 
```
And place the dataset in the spcfied folders.
- MS MARCO: `./resources/dataset/msmarco-data/raw`
- Natural Questions: `./resources/dataset/nq-data/raw`
- T5-model: `./resources/transformer_models`

#### Data preprocessing 
# NQ Dataset Preparation 
- Processes, cleans, and extracts information from raw Natural Questions (NQ) train and dev datasets, saves the processed train and dev datasets separately as `.gz` files, and merges them into a single deduplicated dataset, also saved as a `.gz` file.

```bash
bash ../scripts/data_processing_scripts/NQ_dataset_processing.sh
```
- Convert NQ datasets into MS MARCO format by generating queries, qrels, and document metadata for retrieval tasks.

```bash
bash ../scripts/data_processing_scripts/NQ_create_msmarco_format_dataset.sh 
```
- PreCompute nq dataset T5 embeding 

 ```bash
bash ../scripts/data_processing_scripts/compute_nq_t5_embeddings.sh
```

- Generate NQ encodded doc ids 

 ```bash
bash ../scripts/generate_instances_scripts/generate_nq_t5_encoded_ids.sh
```
- Generate T5 traning and eval files

 ```bash
bash ../scripts/generate_instances_scripts/run_generate_three_stage_nq_eval_data.sh
bash ../scripts/generate_instances_scripts/run_generate_three_stage_nq_train_data.sh
```

# DDRO TRANING DATA
- Nq first stage indexing and retrival 
```bash
bash ./sbatch scripts/data_processing_scripts run_nq_bm25_indexing_and_retrieval.sh 
```
- create NQ triples 
```bash
bash sbatch ./scripts/data_processing_scripts/NQ_create_tripls.sh  
```
# MS-Marco-dataset
- Sample the datasets 
```bash
bash ./scripts/run_msmarco_dataset_smapling.sh
```
- Compute the embedding of each document based on pre-trained  T5-based encoder (GTR) 

```bash
bash ./scripts/generate_nq_t5_embeddings.sh
bash ./scripts/generate_msmarco_t5_embeddings.sh
```

- Generate encodeed doc ids 
  - You can genrate 3 diffrent docid by changing the input file path and the encoding type 
                - Atomic 
                - URL
                - PQ
```bash
bash ./scripts/generate_nq_t5_encoded_ids.sh
```
#### Query Generation
- uses[ docTTTTTquery](https://github.com/castorini/docTTTTTquery) checkpoint to generate synthetic queries. If you finetune docTTTTTquery checkpoint, the query generation files can make the retrieval result even better. We show how to finetune the model. The following command will finetune the model for 4k iterations to predict queries. We assume you put the tsv training file in gs://your_bucket/qcontent_train_512.csv (download from above). Also, change your_tpu_name, your_tpu_zone, your_project_id, and your_bucket accordingly. For each document, we generate 10 pseudo queries for training. You can also generate pseudo queries with your own settings, then output the result in the following format.


```bash
[d0]    which of the following is the most common form of email marketing
[d0]    which of the following is not a form of email marketing
[d0]    which of the following is not a type of email marketing
...
[d1]    when do we find out who the mother is in how i met your mother
[d1]    who is the mother in how i met your mother
...
...
```
- Fine-tune docTTTTTquery model on the NQ dataset.

```bash
bash ./scripts/run_finetune_docTTTTTquery.sh
```
- For sampling the negative docs for the NQ-Dataset
- [Pyserini: Detailed Installation Guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
Here’s a more concise version:
Or simply 
## Setup for BM25 Retrieval

1. **Create Environment**:
   ```bash
   conda env create -f pysinir.yml
   conda activate pyserini
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r pysinir.txt
   ```

Now you’re ready to run the BM25 retrieval task.

--- 

For three-stage training, we generate training data by:
cd starter_script
python gen_train_data.py --encoding pq --scale top_300k --cur_data general_pretrain
python gen_train_data.py --encoding pq --scale top_300k --cur_data search_pretrain
python gen_train_data.py --encoding pq --scale top_300k --cur_data finetune

Train your model

#MS-MARCO

1. sanple 300k top docs 
2. generate embedding for msmarco all collection  (compute embeding)
3. generate encodded docid 
- pq 
- url
- atomic 
- SUMARY 
    - [docid]    token_id_1,token_id_2,...,token_id_n

4. generate  three-stage training, for SFT
5. generate triples for DPO

#### Datasets:
- [MS MARCO](https://microsoft.github.io/msmarco/Datasets.html#document-ranking-dataset)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)



## Acknowledgments
This repository incorporates code adapted from the following sources:  

- [ULTRON](https://github.com/smallporridge/WebUltron/tree/main)  
- [HF TRL library](https://github.com/huggingface/trl), which includes support for the [`DPOTrainer`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) used for fine-tuning language models  
- [NCI](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI/blob/main/Data_process/NQ_dataset/NQ_dataset_Process.ipynb) for preprocessing the NQ dataset  

Credit is due to the authors of these projects for their valuable contributions.

## Contact
For any questions or concerns, please contact me via email at **kidistamdie@gmail.com**.
