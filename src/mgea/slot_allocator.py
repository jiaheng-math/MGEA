from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import SLOT_FEATURE_NAMES, slot_features
from .schema import Example, Passage


@dataclass(frozen=True)
class SlotConfig:
    base_k: int = 5
    max_k: int = 20
    target_avg_k: float = 7.0
    per_query_cap: int = 5
    folds: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.base_k < 1:
            raise ValueError("base_k must be positive.")
        if self.max_k <= self.base_k:
            raise ValueError("max_k must be larger than base_k.")
        if self.target_avg_k < self.base_k:
            raise ValueError("target_avg_k cannot be smaller than base_k.")
        if self.per_query_cap < 0:
            raise ValueError("per_query_cap cannot be negative.")
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")


@dataclass(frozen=True)
class SlotRecord:
    qid: str
    pid: str
    rank: int
    features: tuple[float, ...]
    label: int | None


class SlotAllocator:
    """MGEA's conditional marginal evidence-slot allocator.

    The allocator keeps graph top-5 fixed, scores graph ranks 6-20 relative to
    that base context, and applies a global slot budget with a per-query cap.
    """

    feature_names = SLOT_FEATURE_NAMES

    def __init__(self, config: SlotConfig | None = None) -> None:
        self.config = config or SlotConfig()
        self.model: Any | None = None

    def candidates(self, example: Example, *, require_labels: bool = False) -> list[SlotRecord]:
        if require_labels and not example.gold_keys:
            raise ValueError(f"Example {example.id} has no gold evidence for slot labels.")
        base_ids = {passage.id for passage in example.graph[: self.config.base_k]}
        seen = set(base_ids)
        missing_gold = example.gold_keys - example.covered_gold(example.graph[: self.config.base_k])
        records: list[SlotRecord] = []
        for rank, passage in enumerate(example.graph[: self.config.max_k], start=1):
            if rank <= self.config.base_k or passage.id in seen:
                continue
            seen.add(passage.id)
            label = None
            if example.gold_keys:
                label = int(bool(passage.evidence_keys & missing_gold))
            records.append(
                SlotRecord(
                    qid=example.id,
                    pid=passage.id,
                    rank=rank,
                    features=tuple(
                        slot_features(
                            example,
                            passage,
                            base_k=self.config.base_k,
                            max_k=self.config.max_k,
                        )
                    ),
                    label=label,
                )
            )
        return records

    def fit(self, examples: Sequence[Example]) -> "SlotAllocator":
        records = [record for example in examples for record in self.candidates(example, require_labels=True)]
        if not records:
            raise ValueError("No tail candidates are available for slot training.")
        x = np.asarray([record.features for record in records], dtype=np.float64)
        y = np.asarray([int(record.label) for record in records], dtype=np.int64)
        self.model = _fit_classifier(x, y, self.config.random_seed, class_weight="balanced")
        return self

    def predict_scores(self, examples: Sequence[Example]) -> dict[str, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Call fit() or load a trained allocator before prediction.")
        output: dict[str, dict[str, float]] = {}
        for example in examples:
            records = self.candidates(example)
            if not records:
                output[example.id] = {}
                continue
            x = np.asarray([record.features for record in records], dtype=np.float64)
            probabilities = _positive_probabilities(self.model, x)
            output[example.id] = {
                record.pid: float(probability)
                for record, probability in zip(records, probabilities)
            }
        return output

    def allocate(
        self,
        examples: Sequence[Example],
        scores: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, list[str]]:
        scores = scores if scores is not None else self.predict_scores(examples)
        ranked_records: list[tuple[float, SlotRecord]] = []
        for example in examples:
            for record in self.candidates(example):
                ranked_records.append((float(scores.get(example.id, {}).get(record.pid, 0.0)), record))
        ranked_records.sort(key=lambda item: (-item[0], item[1].qid, item[1].rank, item[1].pid))

        desired = int(round(len(examples) * (self.config.target_avg_k - self.config.base_k)))
        maximum = len(examples) * self.config.per_query_cap
        slot_budget = max(0, min(desired, maximum, len(ranked_records)))
        selected: dict[str, list[tuple[int, str]]] = {example.id: [] for example in examples}
        counts: Counter[str] = Counter()
        for _, record in ranked_records:
            if sum(counts.values()) >= slot_budget:
                break
            if counts[record.qid] >= self.config.per_query_cap:
                continue
            selected[record.qid].append((record.rank, record.pid))
            counts[record.qid] += 1
        return {
            qid: [pid for _, pid in sorted(values)]
            for qid, values in selected.items()
        }

    def contexts(
        self,
        examples: Sequence[Example],
        selected: dict[str, list[str]],
    ) -> dict[str, list[Passage]]:
        contexts: dict[str, list[Passage]] = {}
        for example in examples:
            base = list(_dedupe_passages(example.graph[: self.config.base_k]))
            base_ids = {passage.id for passage in base}
            selected_ids = set(selected.get(example.id, []))
            tail = [
                passage
                for passage in example.graph[self.config.base_k : self.config.max_k]
                if passage.id in selected_ids and passage.id not in base_ids
            ]
            contexts[example.id] = base + list(_dedupe_passages(tail))
        return contexts

    def out_of_fold(self, examples: Sequence[Example]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        labeled = [example for example in examples if example.gold_keys and self.candidates(example)]
        if len(labeled) < 2:
            raise ValueError("At least two labeled queries with tail candidates are required for OOF.")
        query_labels = [int(any(record.label for record in self.candidates(example))) for example in labeled]
        folds = _query_folds(query_labels, self.config.folds, self.config.random_seed)
        scores: dict[str, dict[str, float]] = {}
        true_labels: list[int] = []
        predicted: list[float] = []
        fold_by_qid: dict[str, int] = {}

        for fold_index, (train_indices, test_indices) in enumerate(folds, start=1):
            train_records = [
                record
                for index in train_indices
                for record in self.candidates(labeled[int(index)], require_labels=True)
            ]
            x_train = np.asarray([record.features for record in train_records], dtype=np.float64)
            y_train = np.asarray([int(record.label) for record in train_records], dtype=np.int64)
            model = _fit_classifier(
                x_train,
                y_train,
                self.config.random_seed,
                class_weight="balanced",
            )
            for index in test_indices:
                example = labeled[int(index)]
                records = self.candidates(example, require_labels=True)
                x_test = np.asarray([record.features for record in records], dtype=np.float64)
                probabilities = _positive_probabilities(model, x_test)
                scores[example.id] = {
                    record.pid: float(probability)
                    for record, probability in zip(records, probabilities)
                }
                fold_by_qid[example.id] = fold_index
                true_labels.extend(int(record.label) for record in records)
                predicted.extend(float(probability) for probability in probabilities)

        metrics: dict[str, Any] = {
            "queries": len(labeled),
            "folds": len(folds),
            "feature_count": len(SLOT_FEATURE_NAMES),
            "candidates": len(true_labels),
            "positive_candidates": int(sum(true_labels)),
            "positive_rate": float(np.mean(true_labels)) if true_labels else 0.0,
            "fold_by_query": fold_by_qid,
        }
        if len(set(true_labels)) > 1:
            metrics["auc"] = float(roc_auc_score(true_labels, predicted))
            metrics["average_precision"] = float(average_precision_score(true_labels, predicted))
        else:
            metrics["auc"] = None
            metrics["average_precision"] = None
        return scores, metrics


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    random_seed: int,
    *,
    class_weight: str | None,
) -> Any:
    if len(x) == 0:
        raise ValueError("Cannot train on an empty feature matrix.")
    if len(set(y.tolist())) < 2:
        model: Any = DummyClassifier(strategy="prior")
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    model.fit(x, y)
    return model


def _positive_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(x.shape[0], dtype=np.float64)
    return probabilities[:, classes.index(1)]


def _query_folds(labels: Sequence[int], requested_folds: int, random_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = Counter(labels)
    if len(counts) > 1 and min(counts.values()) >= 2:
        n_splits = min(requested_folds, min(counts.values()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return list(splitter.split(np.zeros(len(labels)), labels))
    n_splits = min(requested_folds, len(labels))
    if n_splits < 2:
        raise ValueError("At least two queries are required for grouped OOF.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return list(splitter.split(np.zeros(len(labels))))


def _dedupe_passages(passages: Iterable[Passage]) -> Iterable[Passage]:
    seen: set[str] = set()
    for passage in passages:
        if passage.id in seen:
            continue
        seen.add(passage.id)
        yield passage
