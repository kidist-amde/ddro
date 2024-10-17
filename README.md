### Data Preparation

Download MS MARCO and Natural Questions datasets using the provided shell script.

#### Prerequisites
Ensure you have the following tools installed:
- `wget` for MS MARCO
- `[gcloud` and `gsutil`]() for Natural Questions 
- Addtionaly Download and store  T5-base model in `./resources/transformer_models/t5-base/`

#### Instructions
Run the script to download the datasets:
```bash
bash run_scripts/download_msmarco_and_nq_datasets.sh
```

- MS MARCO: `./resources/dataset/msmarco-data/`
- Natural Questions: `./resources/dataset/nq-data/`

#### Datasets:
- [MS MARCO](https://microsoft.github.io/msmarco/Datasets.html#document-ranking-dataset)
- [Natural Questions](https://ai.google.com/research/NaturalQuestions)
