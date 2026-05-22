#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "No active conda environment. Run bash scripts/setup_conda_env.sh first, then conda activate graph-routing."
  exit 1
fi

if [[ -z "${HIPPORAG_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set HIPPORAG_API_KEY or OPENAI_API_KEY before running."
  exit 1
fi

if [[ -n "${HIPPORAG_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${HIPPORAG_API_KEY}"
fi

python -m src.study_main --config configs/study_hotpot_hipporag_shared.yaml "$@"
