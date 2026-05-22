#!/usr/bin/env bash
set -euo pipefail

python -m src.study_main --config configs/study_mixed_hotpot2wiki_nq_colbert_shared.yaml
