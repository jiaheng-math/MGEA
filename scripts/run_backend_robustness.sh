#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/backend_robustness_template.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRE_EXACT_SHARED="${REQUIRE_EXACT_SHARED:-1}"

echo "[backend-robustness] root: $ROOT_DIR"
echo "[backend-robustness] python: $PYTHON_BIN"
echo "[backend-robustness] config: $CONFIG_PATH"

VALIDATE_ARGS=(--config "$CONFIG_PATH")
if [[ "$REQUIRE_EXACT_SHARED" == "1" ]]; then
  VALIDATE_ARGS+=(--require-exact-shared)
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/validate_backend_artifacts.py" "${VALIDATE_ARGS[@]}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_backend_robustness.py" --config "$CONFIG_PATH"
"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_backend_robustness.py" --config "$CONFIG_PATH"
