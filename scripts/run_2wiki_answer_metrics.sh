#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_answer_metrics_from_retrieval.sh \
  "${1:-results/study_2wiki_hipporag_colbert_shared}" \
  "2wikimultihopqa"
