# MGEA

Main-experiment code release for the paper project on cost-aware graph retrieval for question answering.

This repository is intentionally scoped to the experiments needed for the paper's main results: probe-aware graph invocation and marginal evidence-slot allocation. Generated results, caches, raw datasets, exploratory diagnostics, backend-management scripts, and failed/ablation side tracks are not included in the public release.

## Repository Layout

```text
MGEA/
  configs/          Main HotpotQA, 2Wiki, NQ, and mixed-workload configs
  data/             Local dataset location; only .gitkeep is tracked
  scripts/          Main data-prep, experiment, QA, plotting, and slot-allocation scripts
  src/              Core dataset, retrieval, feature, routing, and evaluation modules
  environment.yml   Conda environment definition
```

## Installation

```bash
bash scripts/setup_conda_env.sh
conda activate graph-routing
python -m spacy download en_core_web_sm
```

The setup script installs ColBERT separately and applies the included ColBERT stability patch used by the main experiments.

## Data Preparation

Place raw datasets under `data/`. Raw datasets are not tracked by git.

```bash
bash scripts/prepare_hotpot_shared_corpus.sh 500
bash scripts/prepare_2wiki_shared_corpus.sh 500
bash scripts/prepare_nq_shared_corpus.sh 500 42 data/nq_dev.jsonl.gz
bash scripts/build_mixed_hotpot2wiki_nq.sh
```

## Retrieval and Routing

Run a main experiment from a YAML config:

```bash
python -m src.study_main --config configs/study_hotpot_hipporag_colbert_shared.yaml
```

Convenience entry points:

```bash
bash scripts/run_hotpot_hipporag_colbert_smoke.sh
bash scripts/run_hotpot_hipporag_colbert_200.sh
bash scripts/run_hotpot_hipporag_colbert.sh
bash scripts/run_nq_hipporag_colbert_500.sh
bash scripts/run_mixed_hotpot2wiki_nq_colbert.sh
```

Tracked configs:

- `configs/study_hotpot_hipporag_colbert_shared.yaml`
- `configs/study_hotpot_hipporag_colbert_200.yaml`
- `configs/study_2wiki_hipporag_colbert_shared.yaml`
- `configs/study_nq_hipporag_colbert_500.yaml`
- `configs/study_mixed_hotpot2wiki_nq_colbert_shared.yaml`

Each run writes generated artifacts under `results/`, including `metrics.json`, `partial_metrics.json`, `routing_rows.jsonl`, and `retrieval_results.jsonl`. These outputs are not committed.

## QA Evaluation

Prepare generation inputs from saved retrieval and routing outputs:

```bash
python scripts/plot_routing_pareto.py \
  --input results/study_hotpot_hipporag_colbert_500/routing_rows.jsonl \
  --output-dir results/study_hotpot_hipporag_colbert_500/pareto_strict_cv \
  --policy strict \
  --label-k 5 \
  --top-k 5 \
  --metrics-json results/study_hotpot_hipporag_colbert_500/metrics.json

python scripts/prepare_main_table_generation_input.py \
  --retrieval-results results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \
  --oof-predictions results/study_hotpot_hipporag_colbert_500/pareto_strict_cv/oof_predictions.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_generation_input_top5_strict_cv.jsonl \
  --top-k 5 \
  --threshold 0.5 \
  --tie-policy dense
```

Generate and evaluate answers:

```bash
python scripts/batch_generate_from_retrieval.py \
  --input results/study_hotpot_hipporag_colbert_500/main_table_generation_input_top5_strict_cv.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --model gpt-4.1 \
  --methods auto \
  --top-k 5

python scripts/evaluate_generations.py \
  --input results/study_hotpot_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_qa_metrics_top5_strict_cv.json \
  --per-sample-output results/study_hotpot_hipporag_colbert_500/main_table_qa_per_sample_top5_strict_cv.jsonl \
  --methods auto
```

## Marginal Evidence-Slot Allocation

Materialize deep graph evidence:

```bash
python scripts/materialize_deep_retrieval.py \
  --config configs/study_hotpot_hipporag_colbert_shared.yaml \
  --top-k 20 \
  --prefix-retrieval results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/retrieval_results_deep20.jsonl
```

Run the slot allocation experiment:

```bash
python scripts/eval_adaptive_context_budget.py \
  --datasets hotpot 2wiki \
  --deep-retrieval-files \
    results/study_hotpot_hipporag_colbert_500/retrieval_results_deep20.jsonl \
    results/study_2wiki_hipporag_colbert_500/retrieval_results_deep20.jsonl \
  --base-k 5 \
  --large-k 8 10 \
  --slot-target-avg-k 7 \
  --slot-max-k 20 \
  --slot-per-query-cap 5 \
  --slot-feature-variants full text_rerank passage_rerank no_probe probe_only \
  --output results/marginal_slot_allocation_ablation_eval.json \
  --per-sample-output results/marginal_slot_allocation_ablation_per_sample.jsonl \
  --generation-input-output results/marginal_slot_allocation_ablation_generation_input.jsonl \
  --generation-methods slot_ablation
```

Reader sweep and summary:

```bash
python scripts/run_slot_reader_sweep.py \
  --generation-input results/marginal_slot_allocation_ablation_generation_input.jsonl \
  --models gpt-4.1 gpt-4o-mini \
  --output-dir results/slot_reader_sweep \
  --generation-seed 42

python scripts/summarize_slot_ablation.py \
  --allocation-report results/marginal_slot_allocation_ablation_eval.json \
  --qa-metrics results/slot_reader_sweep/qa_metrics_gpt-4.1.json \
  --include-probe-only
```

## Release Scope

The public repository tracks only main-experiment code and configs. It intentionally excludes:

- `results/`, cache directories, SQLite caches, and JSONL outputs
- raw datasets
- local virtual environments
- exploratory diagnostics and backend-management scripts
- superseded residual, counterfactual, iterative, masking, and robustness experiments
