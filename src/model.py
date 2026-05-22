from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelOutput:
    auc: float
    accuracy: float
    probabilities: list[float]
    predictions: list[int]


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[str]
    test_ids: list[str]
    split_mode: Literal["train_test", "resubstitution"]


@dataclass(frozen=True)
class CVFold:
    train_ids: list[str]
    test_ids: list[str]
    fold_index: int


def make_split(
    sample_ids: list[str],
    labels: list[int],
    train_ratio: float,
    random_seed: int,
) -> SplitResult:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need at least two classes for logistic regression.")

    class_counts = {label: labels.count(label) for label in unique_labels}
    if min(class_counts.values()) < 2:
        return SplitResult(
            train_ids=list(sample_ids),
            test_ids=list(sample_ids),
            split_mode="resubstitution",
        )

    stratify = labels
    train_ids, test_ids = train_test_split(
        sample_ids,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=stratify,
    )
    return SplitResult(
        train_ids=list(train_ids),
        test_ids=list(test_ids),
        split_mode="train_test",
    )


def train_and_evaluate(
    train_matrix: np.ndarray,
    train_labels: list[int],
    test_matrix: np.ndarray,
    test_labels: list[int],
    random_seed: int,
) -> ModelOutput:
    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=random_seed, max_iter=1000)),
        ]
    )
    classifier.fit(train_matrix, train_labels)
    probabilities = classifier.predict_proba(test_matrix)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    auc = 0.5 if len(set(test_labels)) < 2 else roc_auc_score(test_labels, probabilities)
    accuracy = accuracy_score(test_labels, predictions)
    return ModelOutput(
        auc=float(auc),
        accuracy=float(accuracy),
        probabilities=probabilities.tolist(),
        predictions=predictions.astype(int).tolist(),
    )


def make_cv_folds(
    sample_ids: list[str],
    labels: list[int],
    num_folds: int,
    random_seed: int,
) -> list[CVFold]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need at least two classes for cross-validation.")
    class_counts = {label: labels.count(label) for label in unique_labels}
    if min(class_counts.values()) < num_folds:
        raise ValueError(
            f"Not enough samples in the minority class for {num_folds}-fold CV. "
            f"Class counts: {class_counts}"
        )

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    sample_ids_array = np.asarray(sample_ids)
    labels_array = np.asarray(labels)
    folds: list[CVFold] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(sample_ids_array, labels_array), start=1):
        folds.append(
            CVFold(
                train_ids=sample_ids_array[train_idx].tolist(),
                test_ids=sample_ids_array[test_idx].tolist(),
                fold_index=fold_index,
            )
        )
    return folds
