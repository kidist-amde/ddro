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
- Sample the datasets 
```bash
bash ./scripts/run_msmarco_dataset_smapling.sh
bash ./scripts/run_nq_merge.sh
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
#### Datasets:
- [MS MARCO](https://microsoft.github.io/msmarco/Datasets.html#document-ranking-dataset)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)
