import os
import json
import gzip
import nltk
import argparse
from tqdm import tqdm as base_tqdm
from collections import defaultdict

nltk.data.path.append("/ivi/ilps/personal/kmekonn/projects")

class LoggingTqdm(base_tqdm):
    def __init__(self, *args, log_every=100000, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_every = log_every
        self._logger = logger
        self._last_logged = 0

    def update(self, n=1):
        super().update(n)
        if self.n - self._last_logged >= self.log_every:
            self._last_logged = self.n
        if self._logger:
            if self.total:
                self._logger.info(f"{self.desc}: {self.n}/{self.total} ({self.n / self.total:.2%})")
            else:
                self._logger.info(f"{self.desc}: {self.n} items processed.")


def tqdm_log(*args, logger=None, **kwargs):
    return LoggingTqdm(*args, logger=logger, **kwargs)

def setup_logger(log_file):
    import logging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

def smart_open(path, mode):
    return gzip.open(path, mode) if path.endswith('.gz') else open(path, mode)

def generate_top_documents(doc_click_count, input_path, output_path, logger=None):
    count = 0
    with smart_open(input_path, "rt") as fr, smart_open(output_path, "wt") as fw:
        for line in tqdm_log(fr, desc="Filtering top documents", logger=logger):
            docid = json.loads(line)["docid"]
            if doc_click_count[docid] <= 0:
                continue
            fw.write(line)
            count += 1
    if logger:
        logger.info(f"Total top documents written: {count}")
    print(f"Total top documents written: {count}")

def main(args):
    logger = setup_logger(args.log_file) if args.log_file else None

    doc_click_count = defaultdict(int)
    id_to_content = {}

    with smart_open(args.doc_file, "rt") as fin:
        for i, line in tqdm_log(enumerate(fin), desc="Reading documents", logger=logger):
            cols = line.split("\t")
            if len(cols) != 4:
                continue
            docid, url, title, body = cols
            sents = nltk.sent_tokenize(body)
            id_to_content[docid] = {"docid": docid, "url": url, "title": title, "body": body, "sents": sents}
            doc_click_count[docid] = 0

    if logger:
        logger.info(f"Total unique documents: {len(doc_click_count)}")
    print("Total number of unique documents:", len(doc_click_count))

    with smart_open(args.qrels_file, "rt") as fr:
        for line in tqdm_log(fr, desc="Parsing qrels", logger=logger):
            queryid, _, docid, _ = line.strip().split()
            doc_click_count[docid] += 1

    sorted_click_count = sorted(doc_click_count.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most clicked:", sorted_click_count[:5])

    with smart_open(args.output_file, "wt") as fout:
        for docid, _ in sorted_click_count:
            if docid not in id_to_content:
                continue
            fout.write(json.dumps(id_to_content[docid]) + "\n")

    generate_top_documents(
        doc_click_count,
        input_path=args.output_file,
        output_path=args.top_output_file,
        logger=logger
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_file", type=str, required=True)
    parser.add_argument("--qrels_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_output_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()
    main(args)


