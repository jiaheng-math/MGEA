#!/usr/bin/env bash
set -euo pipefail

SUBSET_SIZE="${1:-500}"
SEED="${2:-42}"
INPUT="${3:-data/nq_dev.jsonl.gz}"
MAX_PASSAGE_TOKENS="${4:-384}"

python scripts/prepare_nq_shared_corpus.py \
  --input "${INPUT}" \
  --queries-output "data/nq_dev_${SUBSET_SIZE}_seed${SEED}_queries.json" \
  --corpus-output "data/nq_dev_${SUBSET_SIZE}_seed${SEED}_corpus.json" \
  --subset-size "${SUBSET_SIZE}" \
  --seed "${SEED}" \
  --max-passage-tokens "${MAX_PASSAGE_TOKENS}"
