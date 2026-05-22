#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_study.sh <config-path>"
  exit 1
fi

python -m src.study_main --config "$1"
