#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/prepare_hotpot_shared_corpus.sh <subset-size> [seed]"
  exit 1
fi

SUBSET_SIZE="$1"
SEED="${2:-42}"

python scripts/prepare_shared_corpus_dataset.py \
  --input data/hotpot_dev_distractor_v1.json \
  --queries-output "data/hotpot_dev_distractor_${SUBSET_SIZE}_seed${SEED}_queries.json" \
  --corpus-output "data/hotpot_dev_distractor_${SUBSET_SIZE}_seed${SEED}_corpus.json" \
  --subset-size "${SUBSET_SIZE}" \
  --seed "${SEED}" \
  --dataset-name hotpotqa \
  --workload multi-hop
