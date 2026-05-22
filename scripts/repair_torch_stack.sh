#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Restoring torch / torchvision for HippoRAG + vLLM compatibility..."
"${PYTHON_BIN}" -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu124 \
  --trusted-host download.pytorch.org \
  torch==2.5.1 torchvision==0.20.1
