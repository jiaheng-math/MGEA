# MGEA

Anonymous-review artifact for the paper project on cost-aware graph retrieval
for question answering.

This snapshot is intentionally scoped. It keeps the repository structure,
environment definition, dataset preparation utilities, retrieval backend
wrappers, and non-sensitive QA evaluation utilities. The core unpublished MGEA
method is withheld from this anonymous review artifact and will be released in
the final public code package.

See `REVIEW_RELEASE.md` for the exact release scope.

## Repository Layout

```text
MGEA/
  configs/          Example HotpotQA, 2Wiki, NQ, and mixed-workload configs
  data/             Local dataset location; only .gitkeep is tracked
  scripts/          Data-prep and non-sensitive evaluation utilities
  src/              Dataset/retrieval helpers plus explicit placeholders
  REVIEW_RELEASE.md Anonymous-review release scope
  NOTICE            Review-use notice
  environment.yml   Conda environment definition
```

## Installation

```bash
bash scripts/setup_conda_env.sh
conda activate graph-routing
python -m spacy download en_core_web_sm
```

The setup script installs ColBERT separately and applies the included ColBERT
stability patch used by the main experiments.

## Data Preparation

Place raw datasets under `data/`. Raw datasets are not tracked by git.

```bash
bash scripts/prepare_hotpot_shared_corpus.sh 500
bash scripts/prepare_2wiki_shared_corpus.sh 500
bash scripts/prepare_nq_shared_corpus.sh 500 42 data/nq_dev.jsonl.gz
bash scripts/build_mixed_hotpot2wiki_nq.sh
```

## Withheld Components

The following files are explicit placeholders in this anonymous-review archive:

- `src/study_main.py`
- `src/features.py`
- `src/model.py`
- `scripts/plot_routing_pareto.py`
- `scripts/eval_adaptive_context_budget.py`
- `scripts/materialize_deep_retrieval.py`
- `scripts/prepare_main_table_generation_input.py`
- `scripts/run_slot_reader_sweep.py`
- `scripts/plot_qa_budget_curve.py`
- `scripts/plot_budget_pareto.py`
- `scripts/summarize_slot_ablation.py`
- `scripts/run_answer_metrics_from_retrieval.sh`

These files preserve module paths while avoiding disclosure of the unpublished
routing and marginal evidence-slot allocation implementation.

## Included Utilities

The archive still includes dataset preparation scripts, retrieval backend
wrappers, answer generation, answer evaluation, plotting utilities that do not
define the withheld method, and environment/config examples.

Generated outputs such as `results/`, cache directories, SQLite caches, JSONL
outputs, raw datasets, and local virtual environments are not included.
