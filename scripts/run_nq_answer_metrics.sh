#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_answer_metrics_from_retrieval.sh \
  "${1:-results/study_nq_hipporag_colbert_500}" \
  "nq"
