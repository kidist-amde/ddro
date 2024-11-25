import csv
import random
import gzip
import os
from collections import defaultdict
from typing import Dict, List

# Initialize the subset variable
subset = "train"

# Load the query strings for each topicid into a dictionary
querystring: Dict[str, str] = {}
with gzip.open(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-queries.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for topicid, querystring_of_topicid in tsvreader:
        querystring[topicid] = querystring_of_topicid

# Initialize and load the document offsets from a lookup file
docoffset: Dict[str, int] = {}
with gzip.open("resources/datasets/raw/msmarco-data/msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for docid, _, offset in tsvreader:
        docoffset[docid] = int(offset)

# Load the relevance judgments for each topicid into a dictionary
qrel: Dict[str, List[str]] = defaultdict(list)
with gzip.open(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-qrels.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter=" ")
    for topicid, _, docid, rel in tsvreader:
        assert rel == "1"
        qrel[topicid].append(docid)

def getcontent(docid: str, f) -> str:
    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    return line.rstrip()

def get_url(docid: str, f) -> str:
    f.seek(docoffset[docid])
    line = f.readline()
    _docid, url, title, body = line.split("\t")
    assert _docid == docid, f"Looking for {docid}, found {line}"
    return url

def generate_triples(outfile: str, triples_to_generate: int) -> Dict[str, int]:
    stats = defaultdict(int)
    unjudged_rank_to_keep = random.randint(1, 100)
    already_done_a_triple_for_topicid = set()

    with gzip.open(f"resources/datasets/raw/msmarco-data/msmarco-doc{subset}-top100.gz", 'rt', encoding='utf8') as top100f, \
            gzip.open("resources/datasets/raw/msmarco-data/msmarco-docs.tsv.gz", 'rt', encoding="utf8") as f, \
            open(outfile, 'w', encoding="utf8") as out:
        for line in top100f:
            topicid, _, unjudged_docid, rank, _, _ = line.split()

            if topicid in already_done_a_triple_for_topicid or int(rank) != unjudged_rank_to_keep:
                stats['skipped'] += 1
                continue

            unjudged_rank_to_keep = random.randint(1, 100)
            already_done_a_triple_for_topicid.add(topicid)

            assert topicid in querystring
            assert topicid in qrel
            assert unjudged_docid in docoffset

            positive_docid = random.choice(qrel[topicid])
            assert positive_docid in docoffset

            if unjudged_docid in qrel[topicid]:
                stats['docid_collision'] += 1
                continue

            stats['kept'] += 1

            out.write(f"{topicid}\t{querystring[topicid]}\t{get_url(positive_docid, f)}\t{positive_docid}\t{get_url(unjudged_docid, f)}\t{unjudged_docid}\n")
            
            triples_to_generate -= 1
            if triples_to_generate <= 0:
                return stats
        return stats

outfile = f"resources/datasets/processed/msmarco-data/triples_{subset}_data.tsv"
logfile = f"logs-slurm/other-logs/triples_{subset}_generation.log"
triples_to_generate = float('inf')
stats = generate_triples(outfile, triples_to_generate)

with open(logfile, 'w', encoding='utf8') as log:
    for key, val in stats.items():
        log.write(f"{key}\t{val}\n")
    log.write("\nInterpretation of Statistics:\n")
    log.write("skipped: Number of entries ignored during triple generation. Skipping occurs if:\n")
    log.write("         - A triple has already been generated for the same topic (topicid) to avoid duplication.\n")
    log.write("         - The document rank does not match the randomly selected rank to ensure dataset diversity.\n\n")
    log.write("kept: Number of triples successfully generated, each containing:\n")
    log.write("      - A query (topicid and its query text).\n")
    log.write("      - A positive document (judged relevant to the query).\n")
    log.write("      - A random negative document (from the top 100 results, not judged relevant).\n\n")
    log.write("docid_collision: Number of times a randomly selected 'unjudged' document was found to be relevant.\n")
    log.write("                 These cases are excluded to ensure the negative example is irrelevant to the query.\n")
print(f"Log file saved at: {logfile}")