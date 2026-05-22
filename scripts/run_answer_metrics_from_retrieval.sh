#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:?Usage: bash scripts/run_answer_metrics_from_retrieval.sh <result-dir> [dataset-name]}"
DATASET_NAME="${2:-$(basename "${RESULT_DIR}")}"

MODEL="${QA_MODEL:-gpt-4.1}"
BASE_URL="${QA_BASE_URL:-}"
API_KEY_ENV="${QA_API_KEY_ENV:-OPENAI_API_KEY}"
TOP_K="${QA_TOP_K:-5}"
METHODS="${QA_METHODS:-auto}"
POLICY="${QA_POLICY:-strict}"
NUM_FOLDS="${QA_NUM_FOLDS:-5}"
SEED="${QA_RANDOM_SEED:-42}"
THRESHOLD="${QA_ROUTER_THRESHOLD:-0.5}"
TIE_POLICY="${QA_TIE_POLICY:-dense}"

PARETO_DIR="${RESULT_DIR}/pareto_${POLICY}_cv"
GEN_INPUT_PATH="${RESULT_DIR}/main_table_generation_input_top${TOP_K}_${POLICY}_cv.jsonl"
GEN_PATH="${RESULT_DIR}/main_table_generations_top${TOP_K}_${POLICY}_cv.jsonl"
QA_METRICS_PATH="${RESULT_DIR}/main_table_qa_metrics_top${TOP_K}_${POLICY}_cv.json"
QA_PER_SAMPLE_PATH="${RESULT_DIR}/main_table_qa_per_sample_top${TOP_K}_${POLICY}_cv.jsonl"

if [[ ! -f "${RESULT_DIR}/retrieval_results.jsonl" ]]; then
  echo "Missing ${RESULT_DIR}/retrieval_results.jsonl" >&2
  exit 1
fi

if [[ ! -f "${RESULT_DIR}/routing_rows.jsonl" ]]; then
  echo "Missing ${RESULT_DIR}/routing_rows.jsonl" >&2
  exit 1
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key env var: ${API_KEY_ENV}" >&2
  exit 1
fi

echo "=== ${DATASET_NAME}: OOF routing probabilities / Pareto ==="
PYTHONPATH=. python scripts/plot_routing_pareto.py \
  --input "${RESULT_DIR}/routing_rows.jsonl" \
  --output-dir "${PARETO_DIR}" \
  --policy "${POLICY}" \
  --label-k "${TOP_K}" \
  --top-k "${TOP_K}" \
  --num-folds "${NUM_FOLDS}" \
  --random-seed "${SEED}" \
  --metrics-json "${RESULT_DIR}/metrics.json"

echo "=== ${DATASET_NAME}: prepare main-table generation input ==="
python scripts/prepare_main_table_generation_input.py \
  --retrieval-results "${RESULT_DIR}/retrieval_results.jsonl" \
  --oof-predictions "${PARETO_DIR}/oof_predictions.jsonl" \
  --output "${GEN_INPUT_PATH}" \
  --top-k "${TOP_K}" \
  --threshold "${THRESHOLD}" \
  --tie-policy "${TIE_POLICY}" \
  --random-seed "${SEED}"

echo "=== ${DATASET_NAME}: batch generation for all main-table methods ==="
GENERATE_CMD=(python scripts/batch_generate_from_retrieval.py \
  --input "${GEN_INPUT_PATH}" \
  --output "${GEN_PATH}" \
  --model "${MODEL}" \
  --api-key-env "${API_KEY_ENV}" \
  --methods "${METHODS}" \
  --top-k "${TOP_K}" \
  --temperature 0 \
  --timeout 120 \
  --max-retries 5)
if [[ -n "${BASE_URL}" ]]; then
  GENERATE_CMD+=(--base-url "${BASE_URL}")
fi
"${GENERATE_CMD[@]}"

echo "=== ${DATASET_NAME}: main-table QA metrics ==="
python scripts/evaluate_generations.py \
  --input "${GEN_PATH}" \
  --output "${QA_METRICS_PATH}" \
  --per-sample-output "${QA_PER_SAMPLE_PATH}" \
  --methods "${METHODS}"

echo "Wrote:"
echo "- ${PARETO_DIR}/pareto_curve.png"
echo "- ${PARETO_DIR}/pareto_operating_points.json"
echo "- ${GEN_INPUT_PATH}"
echo "- ${GEN_PATH}"
echo "- ${QA_METRICS_PATH}"
