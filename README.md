# MGEA

Code release for the paper project on cost-aware graph retrieval for question answering.

MGEA studies when an expensive graph retriever should be invoked after a first-pass dense retrieval, and how to allocate additional graph evidence slots under a fixed reader-context budget. The repository contains the training, retrieval, routing, evaluation, and plotting code needed to reproduce the experiments. Generated results, caches, model indexes, and raw datasets are intentionally excluded from version control.

## Repository Layout

```text
MGEA/
  configs/          YAML experiment configurations
  data/             local dataset location; only .gitkeep is tracked
  scripts/          data prep, experiment, evaluation, plotting, and diagnostics scripts
  src/              core retrieval, routing, feature, dataset, and evaluation modules
  environment.yml   conda environment definition
```

Ignored local artifacts include `results/`, `hipporag_cache/`, `colbert_cache/`, `lightrag_cache*/`, SQLite caches, JSONL generation outputs, virtual environments, and raw data files.

## Main Components

- Shared-corpus benchmark construction for HotpotQA, 2WikiMultiHopQA, Natural Questions, and mixed workloads.
- Dense retrieval with SentenceTransformers or ColBERTv2.
- Graph retrieval adapters for BM25, HippoRAG2, and LightRAG.
- Probe-aware routing that combines query features and dense-retrieval evidence features.
- Cross-backend routing transfer evaluation from saved `routing_rows.jsonl` artifacts.
- Batch answer generation and EM/F1 evaluation from saved retrieval outputs.
- Marginal evidence-slot allocation over deep HippoRAG evidence reservoirs.
- Plotting and summarization scripts for routing, budget, robustness, and slot-ablation analyses.

## Installation

Create the conda environment from the repository root:

```bash
bash scripts/setup_conda_env.sh
conda activate graph-routing
python -m spacy download en_core_web_sm
```

The setup script installs the base environment from `environment.yml`, installs ColBERT separately, and applies a local ColBERT stability patch. If ColBERT is reinstalled or upgraded later, rerun:

```bash
python scripts/patch_colbert_stability.py
```

Some experiments use OpenAI-compatible LLM or embedding endpoints through HippoRAG or LightRAG. Configure the relevant API environment variables before running those experiments.

## Data Preparation

Place raw datasets under `data/`. Raw datasets are not tracked by git.

Prepare HotpotQA:

```bash
bash scripts/prepare_hotpot_shared_corpus.sh 500
```

Prepare 2WikiMultiHopQA:

```bash
bash scripts/prepare_2wiki_shared_corpus.sh 500
```

Prepare Natural Questions:

```bash
bash scripts/prepare_nq_shared_corpus.sh 500 42 data/nq_dev.jsonl.gz
```

Build the mixed HotpotQA + 2Wiki + NQ benchmark after the individual shared-corpus files exist:

```bash
bash scripts/build_mixed_hotpot2wiki_nq.sh
```

## Running Retrieval and Routing Experiments

Run an experiment from a YAML config:

```bash
python -m src.study_main --config configs/study_hotpot_hipporag_colbert_shared.yaml
```

Equivalent wrapper:

```bash
bash scripts/run_study.sh configs/study_hotpot_hipporag_colbert_shared.yaml
```

Useful entry points:

```bash
bash scripts/run_hotpot_hipporag_colbert_smoke.sh
bash scripts/run_hotpot_hipporag_colbert_200.sh
bash scripts/run_hotpot_hipporag_colbert.sh
bash scripts/run_nq_hipporag_colbert_500.sh
bash scripts/run_mixed_hotpot2wiki_nq_colbert.sh
```

Each study writes generated artifacts under `results/`, including:

- `metrics.json`
- `partial_metrics.json`
- `routing_rows.jsonl`
- `retrieval_results.jsonl`

These files are experiment outputs and are not committed.

## Answer Generation and Evaluation

Prepare main-table generation input from saved retrieval and routing outputs:

```bash
python scripts/prepare_main_table_generation_input.py \
  --retrieval-results results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \
  --oof-predictions results/study_hotpot_hipporag_colbert_500/pareto_strict_cv/oof_predictions.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_generation_input_top5_strict_cv.jsonl \
  --top-k 5 \
  --threshold 0.5 \
  --tie-policy dense
```

Generate answers with a cached OpenAI-compatible reader:

```bash
python scripts/batch_generate_from_retrieval.py \
  --input results/study_hotpot_hipporag_colbert_500/main_table_generation_input_top5_strict_cv.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --model gpt-4.1 \
  --methods auto \
  --top-k 5
```

Evaluate generated answers:

```bash
python scripts/evaluate_generations.py \
  --input results/study_hotpot_hipporag_colbert_500/main_table_generations_top5_strict_cv.jsonl \
  --output results/study_hotpot_hipporag_colbert_500/main_table_qa_metrics_top5_strict_cv.json \
  --per-sample-output results/study_hotpot_hipporag_colbert_500/main_table_qa_per_sample_top5_strict_cv.jsonl \
  --methods auto
```

## Cross-Backend Transfer

Train routing models on one backend's saved routing rows and evaluate on another:

```bash
python -m src.cross_backend_main --config configs/cross_backend_transfer_hotpot.yaml
```

Wrapper:

```bash
bash scripts/run_cross_backend_transfer.sh configs/cross_backend_transfer_hotpot.yaml
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

Run reader sweeps and significance tests:

```bash
python scripts/run_slot_reader_sweep.py \
  --generation-input results/marginal_slot_allocation_ablation_generation_input.jsonl \
  --models gpt-4.1 gpt-4o-mini \
  --output-dir results/slot_reader_sweep \
  --generation-seed 42

python scripts/sig_test_qa.py \
  --input results/slot_reader_sweep/qa_per_sample_gpt-4.1.jsonl \
  --baseline slot_no_probe_graph_5_to_20_cap5_avg7 \
  --compare slot_graph_5_to_20_cap5_avg7,slot_probe_only_graph_5_to_20_cap5_avg7 \
  --n-boot 5000 \
  --seed 42
```

## Reproducibility Notes

- The repository tracks code and configuration only. Regenerate `results/` locally.
- ColBERT indexing is GPU-intensive for full shared-corpus runs. The smoke scripts are intended for environment checks.
- LightRAG experiments require the optional `lightrag-hku` package and configured LLM/embedding endpoints.
- The dense retriever fails loudly if a required encoder is unavailable unless `dense_allow_tfidf_fallback: true` is set in the config.
- For paper tables, regenerate the relevant result files first, then run the summarization and plotting scripts in `scripts/`.
