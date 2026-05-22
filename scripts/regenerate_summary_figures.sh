#!/usr/bin/env bash
set -euo pipefail

python scripts/plot_routing_pareto.py \
  --input results/study_hotpot_hipporag_colbert_500/routing_rows.jsonl \
  --output-dir results/study_hotpot_hipporag_colbert_500/pareto_strict_cv \
  --policy strict \
  --label-k 5 \
  --top-k 5 \
  --metrics-json results/study_hotpot_hipporag_colbert_500/metrics.json

python scripts/plot_routing_pareto.py \
  --input results/study_2wiki_hipporag_colbert_500/routing_rows.jsonl \
  --output-dir results/study_2wiki_hipporag_colbert_500/pareto_strict_cv \
  --policy strict \
  --label-k 5 \
  --top-k 5 \
  --metrics-json results/study_2wiki_hipporag_colbert_500/metrics.json

python scripts/plot_routing_pareto.py \
  --input results/study_nq_hipporag_colbert_500/routing_rows.jsonl \
  --output-dir results/study_nq_hipporag_colbert_500/pareto_strict_cv \
  --policy strict \
  --label-k 5 \
  --top-k 5 \
  --metrics-json results/study_nq_hipporag_colbert_500/metrics.json

python scripts/plot_budget_pareto.py \
  --retrieval-summary results/budgeted_correction_eval_oof_gate_v3.json \
  --hotpot-qa results/study_hotpot_hipporag_colbert_500/budgeted_qa_metrics_v2_core.json \
  --twowiki-qa results/study_2wiki_hipporag_colbert_500/budgeted_qa_metrics_v2_core.json \
  --output-dir results/budget_pareto

python scripts/summarize_evidence_exposure.py \
  --dataset HotpotQA results/study_hotpot_hipporag_colbert_500/budgeted_generation_input_v2.jsonl \
  --dataset 2WikiMHQA results/study_2wiki_hipporag_colbert_500/budgeted_generation_input_v2.jsonl \
  --output-csv results/evidence_exposure_budgeted.csv \
  --output-json results/evidence_exposure_budgeted.json

python scripts/plot_qa_budget_curve.py \
  --generations results/study_hotpot_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --oof-predictions results/study_hotpot_hipporag_colbert_500/pareto_strict_cv/oof_predictions.jsonl \
  --output-dir results/study_hotpot_hipporag_colbert_500/qa_budget_curve_strict_cv \
  --dataset-name HotpotQA

python scripts/plot_qa_budget_curve.py \
  --generations results/study_2wiki_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --oof-predictions results/study_2wiki_hipporag_colbert_500/pareto_strict_cv/oof_predictions.jsonl \
  --output-dir results/study_2wiki_hipporag_colbert_500/qa_budget_curve_strict_cv \
  --dataset-name 2WikiMHQA

python scripts/plot_qa_budget_curve.py \
  --generations results/study_nq_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --oof-predictions results/study_nq_hipporag_colbert_500/pareto_strict_cv/oof_predictions.jsonl \
  --output-dir results/study_nq_hipporag_colbert_500/qa_budget_curve_strict_cv \
  --dataset-name NQ
