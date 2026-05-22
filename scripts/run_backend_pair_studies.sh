#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_backend_pair_studies.sh <dataset: hotpot|2wiki|nq> [python-bin]"
  exit 1
fi

DATASET="$1"
PYTHON_BIN="${2:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$DATASET" in
  hotpot)
    CONFIGS=(
      "$ROOT_DIR/configs/study_hotpot_hipporag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_hotpot_hipporag_bge_large_shared.yaml"
      "$ROOT_DIR/configs/study_hotpot_lightrag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_hotpot_lightrag_bge_large_shared.yaml"
    )
    ;;
  2wiki)
    CONFIGS=(
      "$ROOT_DIR/configs/study_2wiki_hipporag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_2wiki_hipporag_bge_large_shared.yaml"
      "$ROOT_DIR/configs/study_2wiki_lightrag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_2wiki_lightrag_bge_large_shared.yaml"
    )
    ;;
  nq)
    CONFIGS=(
      "$ROOT_DIR/configs/study_nq_hipporag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_nq_hipporag_bge_large_shared.yaml"
      "$ROOT_DIR/configs/study_nq_lightrag_colbert_shared.yaml"
      "$ROOT_DIR/configs/study_nq_lightrag_bge_large_shared.yaml"
    )
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 1
    ;;
esac

for config_path in "${CONFIGS[@]}"; do
  echo "[backend-pairs] running: $config_path"
  "$PYTHON_BIN" -m src.study_main --config "$config_path"
done
