#!/usr/bin/env bash
set -euo pipefail

python scripts/build_mixed_benchmark.py \
  --queries \
    data/hotpot_dev_distractor_1000_seed42_queries.json \
    data/2wikimultihopqa_dev_1000_seed42_queries.json \
    data/nq_dev_500_seed42_queries.json \
  --corpora \
    data/hotpot_dev_distractor_1000_seed42_corpus.json \
    data/2wikimultihopqa_dev_1000_seed42_corpus.json \
    data/nq_dev_500_seed42_corpus.json \
  --queries-output data/mixed_hotpot2wiki_nq_2500_seed42_queries.json \
  --corpus-output data/mixed_hotpot2wiki_nq_2500_seed42_corpus.json
