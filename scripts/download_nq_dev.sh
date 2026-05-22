#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python scripts/download_nq_dev.py \
  --dataset-name "${NQ_HF_DATASET:-google-research-datasets/natural_questions}" \
  --config "${NQ_HF_CONFIG:-dev}" \
  --split "${NQ_HF_SPLIT:-validation}" \
  --output "${1:-data/nq_dev.jsonl.gz}"
