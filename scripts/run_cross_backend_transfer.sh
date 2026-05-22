#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_cross_backend_transfer.sh <config-path>"
  exit 1
fi

python -m src.cross_backend_main --config "$1"
