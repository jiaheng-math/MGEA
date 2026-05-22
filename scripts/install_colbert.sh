#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYPI_MIRROR_URL="${PYPI_MIRROR_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYPI_MIRROR_HOST="${PYPI_MIRROR_HOST:-pypi.tuna.tsinghua.edu.cn}"

echo "Installing ColBERT-compatible setuptools..."
"${PYTHON_BIN}" -m pip install --upgrade "setuptools<81"

echo "Installing ColBERT from official PyPI without touching the existing torch stack..."
if "${PYTHON_BIN}" -m pip install --upgrade \
  --index-url https://pypi.org/simple \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --no-deps \
  "colbert-ai>=0.2.19"; then
  :
else
  echo "Official PyPI install failed, falling back to GitHub source..."
  "${PYTHON_BIN}" -m pip install --upgrade \
    --trusted-host github.com \
    --trusted-host codeload.github.com \
    --no-deps \
    "git+https://github.com/stanford-futuredata/ColBERT.git"
fi

echo "Installing ColBERT runtime dependencies that are skipped by --no-deps..."
"${PYTHON_BIN}" -m pip install --upgrade \
  --index-url "${PYPI_MIRROR_URL}" \
  --trusted-host "${PYPI_MIRROR_HOST}" \
  "ujson>=5"

echo "Patching installed ColBERT for stable indexing..."
"${PYTHON_BIN}" scripts/patch_colbert_stability.py
