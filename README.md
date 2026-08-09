# MGEA

Minimal implementation of **MGEA: Marginal Graph Evidence Allocation for
Efficient Graph-Augmented Retrieval**.

The repository contains the two decisions defined in the paper:

1. **Dense-probe triage** estimates whether graph retrieval should be invoked.
2. **Marginal slot allocation** keeps graph top-5 as the base context and scores
   graph ranks 6-20 by their conditional gain over that base.

MGEA consumes ranked retrieval results, so it is independent of graph indexing,
entity linking, and traversal. Any dense and graph retriever can be used upstream.

## Method

For a query `q`, let `G5(q)` be the fixed graph base and let ranks 6-20 form
the candidate pool. A tail passage is positive only when it covers gold evidence
that `G5(q)` does not already cover. The allocator uses the four feature groups
from the paper:

- local graph rank and score signals;
- dense-support signals;
- lexical novelty and redundancy relative to `G5(q)`;
- query-level dense probes and dense-graph overlap.

A standardized logistic regression predicts conditional marginal value. During
evaluation, every slot from the same query stays in the same fold. Selection is
global under an average context budget of 7, with at most 5 appended passages
per query.

The optional triage model uses question features together with dense top-5 score
confidence, entity coverage, and document diversity. Its labels are:

- `0`: dense top-5 already covers all gold evidence;
- `1`: dense top-5 is insufficient and graph top-5 improves coverage;
- discarded: dense is insufficient but graph top-5 does not improve it.

## Installation

```bash
conda env create -f environment.yml
conda activate mgea
pip install -e .
```

## Input

Use one JSON object per line. `gold_passage_ids` or `gold_titles` is required for
training and out-of-fold evaluation, but not for prediction.

```json
{
  "id": "q-001",
  "question": "Which city was the author of Book A born in?",
  "gold_passage_ids": ["book-a", "author-a"],
  "query_entities": ["Book A"],
  "retrieval": {
    "dense": [
      {"id": "book-a", "title": "Book A", "text": "...", "score": 18.2}
    ],
    "graph": [
      {"id": "book-a", "title": "Book A", "text": "...", "score": 0.91},
      {"id": "author-a", "title": "Author A", "text": "...", "score": 0.47}
    ]
  }
}
```

The ranked lists should contain at least 5 dense passages and 20 graph passages
for the paper setting. Passage IDs should use the same canonical document IDs as
the gold evidence. `query_entities` is optional; a lightweight fallback is used
when it is absent.

## Paper-style OOF run

```bash
mgea oof \
  --input data/retrieval.jsonl \
  --output outputs/oof_predictions.jsonl \
  --metrics outputs/oof_metrics.json \
  --base-k 5 \
  --max-k 20 \
  --target-avg-k 7 \
  --per-query-cap 5 \
  --folds 5 \
  --triage-rate 0.527 \
  --seed 42
```

This writes query-grouped OOF probabilities, selected tail passages, the MGEA
graph context, the triage decision, and the final reader context.

## Fit and predict

```bash
mgea fit \
  --input data/train_retrieval.jsonl \
  --model-output models/mgea.joblib

mgea predict \
  --input data/test_retrieval.jsonl \
  --model models/mgea.joblib \
  --output outputs/predictions.jsonl \
  --triage-rate 0.527
```

## Code layout

```text
src/mgea/schema.py          retrieval input and evidence matching
src/mgea/features.py        paper feature definitions
src/mgea/slot_allocator.py  marginal labels, grouped OOF, and budget allocation
src/mgea/triage.py          dense-probe graph-invocation triage
src/mgea/cli.py             oof, fit, and predict commands
tests/test_mgea.py          synthetic end-to-end checks
```

## Tests

```bash
python -m unittest discover -s tests -v
```
