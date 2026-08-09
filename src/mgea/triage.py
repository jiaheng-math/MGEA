from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import TRIAGE_FEATURE_NAMES, triage_features
from .schema import Example
from .slot_allocator import _fit_classifier, _positive_probabilities


@dataclass(frozen=True)
class TriageConfig:
    top_k: int = 5
    folds: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive.")
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")


class DenseProbeTriage:
    """Optional pre-graph invocation layer from Section 4.3 of the paper."""

    feature_names = TRIAGE_FEATURE_NAMES

    def __init__(self, config: TriageConfig | None = None) -> None:
        self.config = config or TriageConfig()
        self.model: Any | None = None

    def label(self, example: Example) -> int | None:
        if not example.gold_keys:
            return None
        dense_recall = example.recall(example.dense[: self.config.top_k])
        graph_recall = example.recall(example.graph[: self.config.top_k])
        if dense_recall >= 1.0:
            return 0
        if graph_recall > dense_recall:
            return 1
        return None

    def fit(self, examples: Sequence[Example]) -> "DenseProbeTriage":
        labeled = [(example, self.label(example)) for example in examples]
        labeled = [(example, label) for example, label in labeled if label is not None]
        if not labeled:
            raise ValueError("No dense-sufficient or graph-beneficial queries are available.")
        x = np.asarray(
            [triage_features(example, self.config.top_k) for example, _ in labeled],
            dtype=np.float64,
        )
        y = np.asarray([int(label) for _, label in labeled], dtype=np.int64)
        self.model = _fit_classifier(x, y, self.config.random_seed, class_weight=None)
        return self

    def predict_scores(self, examples: Sequence[Example]) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Call fit() or load a trained triage model before prediction.")
        x = np.asarray(
            [triage_features(example, self.config.top_k) for example in examples],
            dtype=np.float64,
        )
        probabilities = _positive_probabilities(self.model, x)
        return {
            example.id: float(probability)
            for example, probability in zip(examples, probabilities)
        }

    def select_invocations(
        self,
        examples: Sequence[Example],
        scores: dict[str, float] | None = None,
        *,
        target_rate: float = 0.527,
    ) -> dict[str, bool]:
        if not 0.0 <= target_rate <= 1.0:
            raise ValueError("target_rate must be between 0 and 1.")
        scores = scores if scores is not None else self.predict_scores(examples)
        invoke_count = int(round(len(examples) * target_rate))
        ranked = sorted(examples, key=lambda item: (-scores.get(item.id, 0.0), item.id))
        selected = {example.id for example in ranked[:invoke_count]}
        return {example.id: example.id in selected for example in examples}

    def out_of_fold(self, examples: Sequence[Example]) -> tuple[dict[str, float], dict[str, Any]]:
        labeled = [(example, self.label(example)) for example in examples]
        labeled = [(example, int(label)) for example, label in labeled if label is not None]
        if len(labeled) < 4:
            raise ValueError("At least four labeled queries are required for triage OOF.")
        labels = [label for _, label in labeled]
        counts = Counter(labels)
        if len(counts) < 2 or min(counts.values()) < 2:
            raise ValueError("Triage OOF requires at least two queries in each class.")
        n_splits = min(self.config.folds, min(counts.values()))
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_seed,
        )
        probabilities: dict[str, float] = {}
        predictions: list[float] = []
        true_labels: list[int] = []
        fold_by_qid: dict[str, int] = {}
        label_array = np.asarray(labels, dtype=np.int64)

        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1
        ):
            x_train = np.asarray(
                [triage_features(labeled[int(index)][0], self.config.top_k) for index in train_indices],
                dtype=np.float64,
            )
            y_train = label_array[train_indices]
            model = _fit_classifier(x_train, y_train, self.config.random_seed, class_weight=None)
            test_examples = [labeled[int(index)][0] for index in test_indices]
            x_test = np.asarray(
                [triage_features(example, self.config.top_k) for example in test_examples],
                dtype=np.float64,
            )
            fold_probabilities = _positive_probabilities(model, x_test)
            for index, example, probability in zip(test_indices, test_examples, fold_probabilities):
                probabilities[example.id] = float(probability)
                predictions.append(float(probability))
                true_labels.append(int(label_array[int(index)]))
                fold_by_qid[example.id] = fold_index

        # Ambiguous queries have no supervision label and therefore cannot be
        # assigned an OOF fold. They are still valid inference queries: fit on
        # all labeled queries and score only this unlabeled remainder.
        ambiguous = [example for example in examples if self.label(example) is None]
        if ambiguous:
            x_all = np.asarray(
                [triage_features(example, self.config.top_k) for example, _ in labeled],
                dtype=np.float64,
            )
            final_model = _fit_classifier(
                x_all, label_array, self.config.random_seed, class_weight=None
            )
            x_ambiguous = np.asarray(
                [triage_features(example, self.config.top_k) for example in ambiguous],
                dtype=np.float64,
            )
            ambiguous_probabilities = _positive_probabilities(final_model, x_ambiguous)
            probabilities.update(
                {
                    example.id: float(probability)
                    for example, probability in zip(ambiguous, ambiguous_probabilities)
                }
            )

        binary_predictions = [int(probability >= 0.5) for probability in predictions]
        metrics = {
            "queries": len(labeled),
            "folds": n_splits,
            "feature_count": len(TRIAGE_FEATURE_NAMES),
            "dense_sufficient": counts.get(0, 0),
            "graph_beneficial": counts.get(1, 0),
            "ambiguous_inference_queries": len(ambiguous),
            "auc": float(roc_auc_score(true_labels, predictions)),
            "accuracy_at_0.5": float(accuracy_score(true_labels, binary_predictions)),
            "fold_by_query": fold_by_qid,
        }
        return probabilities, metrics
