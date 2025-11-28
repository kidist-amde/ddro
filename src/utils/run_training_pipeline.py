import os
import json
import argparse
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default="pq", type=str, help="DocID encoding method: atomic, pq, or url")
    parser.add_argument("--scale", default="top_300k", type=str, help="Dataset scale: top_300k or rand_300k")
    return parser.parse_args()


def load_config(encoding: str, scale: str):
    config_file_path = "scripts/config.json"
    with open(config_file_path, "r") as file:
        config_data = json.load(file)

    config_data["atomic"]["add_doc_num"] = config_data["doc_num"][scale]
    config = config_data[encoding]
    return config, config["add_doc_num"][scale]


def run_stage(stage_name: str, model: str, load_model: str, all_data: str, cur_data: str,
              stage: str, load_ckpt: str, operation: str, max_seq_length: int, epoch: int,
              encoding: str, add_doc_num: int, max_docid_length: int, use_origin_head: str,
              top_or_rand: str, scale: str):

    code_dir = "ddro"
    log_dir = f"logs/{stage}"
    save_dir = f"outputs/{model}_{top_or_rand}_{scale}_{encoding}_{all_data}"
    train_path = f"resources/datasets/processed/msmarco-data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json"
    test_path = f"resources/datasets/processed/msmarco-data/test_data_{top_or_rand}_{scale}/"
    doc_file_path = f"resources/datasets/processed/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json"
    docid_path = f"resources/datasets/processed/msmarco-data/encoded_docid/t5_512_{encoding}_{top_or_rand}.{scale}.txt"
    model_ckpt_path = f"outputs/{load_model}_{top_or_rand}_{scale}_{encoding}_pretrain/model_final.pkl"

    command = [
        "python", f"{code_dir}/utils/run_t5_trainer.py",
        "--epoch", str(epoch),
        "--per_gpu_batch_size", "128",
        "--learning_rate", "1e-3",
        "--save_path", save_dir,
        "--log_path", f"{log_dir}/{stage}.{model}.{top_or_rand}.{scale}.{encoding}.{all_data}.log",
        "--doc_file_path", doc_file_path,
        "--pretrain_model_path", "resources/transformer_models/t5-base",
        "--docid_path", docid_path,
        "--train_file_path", train_path,
        "--test_file_path", test_path,
        "--dataset_script_dir", "data/data_scripts",
        "--dataset_cache_dir", "negs_tutorial_cache",
        "--add_doc_num", str(add_doc_num),
        "--max_seq_length", str(max_seq_length),
        "--max_docid_length", str(max_docid_length),
        "--use_origin_head", use_origin_head,
        "--output_every_n_step", "5000",
        "--save_every_n_epoch", "2",
        "--operation", operation
    ]

    if load_ckpt == "True":
        command.extend(["--load_ckpt", load_ckpt, "--load_ckpt_path", model_ckpt_path])

    print(f"Running stage: {stage_name}")
    subprocess.run(command, check=True)


def main():
    args = parse_arguments()
    encoding = args.encoding
    scale = args.scale
    top_or_rand, scale_val = scale.split("_")

    config, add_doc_num = load_config(encoding, scale)
    max_docid_length = config["max_docid_length"]
    use_origin_head = config["use_origin_head"]

    run_stage(
        stage_name="Content-to-DocID Pretraining",
        model="t5_128_10",
        load_model="t5_128_10",
        all_data="pretrain",
        cur_data="pretrain",
        stage="pretrain",
        load_ckpt="False",
        operation="training",
        max_seq_length=128,
        epoch=10,
        encoding=encoding,
        add_doc_num=add_doc_num,
        max_docid_length=max_docid_length,
        use_origin_head=use_origin_head,
        top_or_rand=top_or_rand,
        scale=scale_val
    )

    run_stage(
        stage_name="PseudoQuery-to-DocID Pretraining",
        model="t5_128_10",
        load_model="t5_128_10",
        all_data="pretrain_search",
        cur_data="search_pretrain",
        stage="search_pretrain",
        load_ckpt="True",
        operation="training",
        max_seq_length=64,
        epoch=20,
        encoding=encoding,
        add_doc_num=add_doc_num,
        max_docid_length=max_docid_length,
        use_origin_head=use_origin_head,
        top_or_rand=top_or_rand,
        scale=scale_val
    )

    run_stage(
        stage_name="Query-to-DocID Finetuning",
        model="t5_128_1",
        load_model="t5_128_10",
        all_data="pretrain_search_finetune",
        cur_data="finetune",
        stage="finetune",
        load_ckpt="True",
        operation="training",
        max_seq_length=64,
        epoch=10,
        encoding=encoding,
        add_doc_num=add_doc_num,
        max_docid_length=max_docid_length,
        use_origin_head=use_origin_head,
        top_or_rand=top_or_rand,
        scale=scale_val
    )

    print("Training pipeline completed.")


if __name__ == '__main__':
    main()
