#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${1:-graph-routing}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH."
  exit 1
fi

# Ignore user-level channel config (for example stale ~/.condarc mirrors such as pkgs/free)
# and use the project's intended channel explicitly.
CONDA_ENV_CMD=(conda env)
CONDA_CHANNEL_ARGS=(--override-channels -c "${CONDA_CHANNEL}")

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_ENV_CMD[@]}" update "${CONDA_CHANNEL_ARGS[@]}" -n "${ENV_NAME}" -f environment.yml --prune
else
  "${CONDA_ENV_CMD[@]}" create "${CONDA_CHANNEL_ARGS[@]}" -n "${ENV_NAME}" -f environment.yml
fi

conda run -n "${ENV_NAME}" env PYTHON_BIN=python bash scripts/install_colbert.sh

echo "Conda environment created: ${ENV_NAME}"
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
