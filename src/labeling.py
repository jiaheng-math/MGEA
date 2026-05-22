from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.dataset import QASample
from src.utils import RetrievedPassage, answer_in_passages


@dataclass(frozen=True)
class LabelResult:
    label: int | None
    dense_quality: float
    graph_quality: float


def build_oracle_label(
    sample: QASample,
    dense_results: list[RetrievedPassage],
    graph_results: list[RetrievedPassage],
) -> LabelResult:
    dense_quality = _retrieval_quality(sample, dense_results)
    graph_quality = _retrieval_quality(sample, graph_results)

    if dense_quality > graph_quality:
        return LabelResult(label=0, dense_quality=dense_quality, graph_quality=graph_quality)
    if graph_quality > dense_quality:
        return LabelResult(label=1, dense_quality=dense_quality, graph_quality=graph_quality)
    return LabelResult(label=None, dense_quality=dense_quality, graph_quality=graph_quality)


def summarize_labels(rows: list[dict]) -> dict[str, int]:
    label_counter = Counter(row["label"] for row in rows if row["label"] is not None)
    return {
        "valid_samples": int(sum(label_counter.values())),
        "label_0_count": int(label_counter.get(0, 0)),
        "label_1_count": int(label_counter.get(1, 0)),
    }


def _retrieval_quality(sample: QASample, retrieved: list[RetrievedPassage]) -> float:
    if sample.gold_passage_ids or sample.gold_titles:
        return _recall_at_k(sample, retrieved)
    return answer_in_passages(sample.answer, retrieved)


def _recall_at_k(sample: QASample, retrieved: list[RetrievedPassage]) -> float:
    if sample.gold_passage_ids:
        gold_targets = set(sample.gold_passage_ids)
        retrieved_targets = {passage.id for passage in retrieved}
    else:
        gold_targets = set(sample.gold_titles)
        retrieved_targets = {passage.title for passage in retrieved if passage.title}

    if not gold_targets:
        return 0.0
    hits = len(gold_targets & retrieved_targets)
    return hits / len(gold_targets)
