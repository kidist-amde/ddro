import os
import json
import gzip
import nltk
import argparse
import tempfile
import shutil
import subprocess
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

def decompress_if_needed(path):
    """Decompress file to temporary file if compressed"""
    if path.endswith('.gz'):
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        with gzip.open(path, 'rt') as fin:
            shutil.copyfileobj(fin, temp_file)
        temp_file.close()
        return temp_file.name
    return path

def main(args):
    logger = setup_logger(args.log_file) if args.log_file else None
    temp_files_to_clean = []
    
    try:
        # Handle compressed input
        doc_file_path = args.doc_file
        if doc_file_path.endswith('.gz'):
            if logger:
                logger.info("Decompressing document file...")
            doc_file_path = decompress_if_needed(args.doc_file)
            temp_files_to_clean.append(doc_file_path)

        # Step 1: Initialize click counts
        doc_click_count = defaultdict(int)
        with open(doc_file_path, 'rt') as fin:
            for line in tqdm_log(fin, desc="Initializing doc counts", logger=logger):
                docid = line.split('\t', 1)[0]
                doc_click_count[docid] = 0

        # Step 2: Process qrels
        with smart_open(args.qrels_file, 'rt') as fr:
            for line in tqdm_log(fr, desc="Processing qrels", logger=logger):
                parts = line.strip().split()
                if len(parts) >= 4:
                    docid = parts[2]
                    doc_click_count[docid] += 1

        # Identify top documents
        top_docids = {docid for docid, count in doc_click_count.items() if count > 0}
        if logger:
            logger.info(f"Top documents count: {len(top_docids)}")
        print(f"Found {len(top_docids)} documents with clicks")

        # Step 3: Create temporary file for sorting
        temp_top_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_files_to_clean.append(temp_top_file.name)
        
        with open(doc_file_path, 'rt') as fin:
            for line in tqdm_log(fin, desc="Processing documents", logger=logger):
                docid = line.split('\t', 1)[0]
                if docid in top_docids:
                    click_count = doc_click_count[docid]
                    temp_top_file.write(f"{click_count}\t{line}")

        temp_top_file.close()

        # Step 4: Sort using Unix command
        sorted_temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        temp_files_to_clean.append(sorted_temp_file.name)
        sorted_temp_file.close()
        
        sort_cmd = [
            'sort',
            '--field-separator=\t',
            '--key=1,1nr',
            '--buffer-size=10%',  # Use 10% of available RAM
            '--parallel=4',       # Use 4 cores
            '--output=' + sorted_temp_file.name,
            temp_top_file.name
        ]
        
        if logger:
            logger.info("Sorting documents...")
        subprocess.run(sort_cmd, check=True)

        # Step 5: Write final output
        with smart_open(args.top_output_file, 'wt') as fout, \
             open(sorted_temp_file.name, 'rt') as fin:
            
            for line in tqdm_log(fin, desc="Writing output", logger=logger):
                _, _, original_line = line.partition('\t')
                docid, url, title, body = original_line.split('\t', 3)
                sents = nltk.sent_tokenize(body)
                doc_struct = {
                    "docid": docid,
                    "url": url,
                    "title": title,
                    "body": body,
                    "sents": sents
                }
                fout.write(json.dumps(doc_struct) + "\n")

        if logger:
            logger.info(f"Finished processing {len(top_docids)} documents")

    finally:
        # Clean up temporary files
        for path in temp_files_to_clean:
            try:
                os.unlink(path)
            except Exception as e:
                if logger:
                    logger.error(f"Error deleting temp file {path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_file", type=str, required=True)
    parser.add_argument("--qrels_file", type=str, required=True)
    parser.add_argument("--top_output_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()
    main(args)