from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.features import probe_feature_names, query_feature_names
from src.model import make_cv_folds, make_split
from src.study_main import (
    matrix_from_rows,
    summarize_router_folds,
    summarize_router_predictions,
)
from src.utils import ensure_dir, load_yaml, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-backend routing transfer evaluation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_random_seed(int(config["random_seed"]))

    project_root = Path(args.config).resolve().parent.parent
    output_dir = ensure_dir(resolve_path(project_root, str(config["output_dir"])))
    train_rows_path = resolve_path(project_root, str(config["train_rows_path"]))
    test_rows_path = resolve_path(project_root, str(config["test_rows_path"]))

    train_rows_all = {row["id"]: row for row in load_jsonl(train_rows_path) if row.get("label") is not None}
    test_rows_all = {row["id"]: row for row in load_jsonl(test_rows_path) if row.get("label") is not None}
    common_ids = sorted(set(train_rows_all) & set(test_rows_all))
    if len(common_ids) < 4:
        raise ValueError("Need at least 4 overlapping labeled queries for cross-backend evaluation.")

    train_rows = [train_rows_all[sample_id] for sample_id in common_ids]
    test_rows = [test_rows_all[sample_id] for sample_id in common_ids]
    if len({int(row["label"]) for row in train_rows}) < 2 or len({int(row["label"]) for row in test_rows}) < 2:
        raise ValueError("Both train and test backends need both label classes on the overlapping sample set.")

    evaluation_mode = str(config.get("evaluation_mode", "train_test")).lower()
    if evaluation_mode == "cv":
        summary = evaluate_cv_transfer(
            train_rows=train_rows,
            test_rows=test_rows,
            num_folds=int(config.get("num_folds", 5)),
            random_seed=int(config["random_seed"]),
            top_k_values=[int(value) for value in config.get("top_k_values", [3, 5])],
        )
    else:
        summary = evaluate_split_transfer(
            train_rows=train_rows,
            test_rows=test_rows,
            train_ratio=float(config.get("train_ratio", 0.8)),
            random_seed=int(config["random_seed"]),
            top_k_values=[int(value) for value in config.get("top_k_values", [3, 5])],
        )

    payload = {
        "config": {
            "train_rows_path": str(train_rows_path),
            "test_rows_path": str(test_rows_path),
            "output_dir": str(output_dir),
            "random_seed": int(config["random_seed"]),
            "evaluation_mode": evaluation_mode,
            "train_ratio": float(config.get("train_ratio", 0.8)),
            "num_folds": int(config.get("num_folds", 5)),
            "top_k_values": [int(value) for value in config.get("top_k_values", [3, 5])],
        },
        "overlap_summary": {
            "overlap_queries": len(common_ids),
            "train_backend": sorted({str(row.get("graph_backend", "unknown")) for row in train_rows}),
            "test_backend": sorted({str(row.get("graph_backend", "unknown")) for row in test_rows}),
        },
        "transfer_metrics": summary,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def evaluate_split_transfer(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_ratio: float,
    random_seed: int,
    top_k_values: list[int],
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in train_rows]
    sample_ids = [str(row["id"]) for row in train_rows]
    split = make_split(sample_ids=sample_ids, labels=labels, train_ratio=train_ratio, random_seed=random_seed)
    train_id_set = set(split.train_ids)
    test_id_set = set(split.test_ids)
    train_subset = [row for row in train_rows if row["id"] in train_id_set]
    test_subset = [row for row in test_rows if row["id"] in test_id_set]
    return {
        "split_mode": split.split_mode,
        "train_samples": len(train_subset),
        "test_samples": len(test_subset),
        "methods": train_transfer_models(train_subset, test_subset, top_k_values, random_seed),
    }


def evaluate_cv_transfer(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_folds: int,
    random_seed: int,
    top_k_values: list[int],
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in train_rows]
    sample_ids = [str(row["id"]) for row in train_rows]
    folds = make_cv_folds(sample_ids=sample_ids, labels=labels, num_folds=num_folds, random_seed=random_seed)
    per_method: dict[str, list[dict[str, Any]]] = {"query_only": [], "probe_only": [], "query_plus_probe": []}
    for fold in folds:
        train_id_set = set(fold.train_ids)
        test_id_set = set(fold.test_ids)
        train_subset = [row for row in train_rows if row["id"] in train_id_set]
        test_subset = [row for row in test_rows if row["id"] in test_id_set]
        method_metrics = train_transfer_models(train_subset, test_subset, top_k_values, random_seed)
        for method_name, metrics in method_metrics.items():
            per_method[method_name].append(metrics)

    return {
        "split_mode": "cross_validation",
        "num_folds": num_folds,
        "methods": {
            method_name: summarize_router_folds(fold_metrics, top_k_values)
            for method_name, fold_metrics in per_method.items()
        },
    }


def train_transfer_models(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    top_k_values: list[int],
    random_seed: int,
) -> dict[str, Any]:
    from src.model import train_and_evaluate

    query_names = query_feature_names()
    probe_names = probe_feature_names()
    y_train = [int(row["label"]) for row in train_rows]
    y_test = [int(row["label"]) for row in test_rows]

    outputs = {
        "query_only": train_and_evaluate(
            matrix_from_rows(train_rows, query_names),
            y_train,
            matrix_from_rows(test_rows, query_names),
            y_test,
            random_seed,
        ),
        "probe_only": train_and_evaluate(
            matrix_from_rows(train_rows, probe_names),
            y_train,
            matrix_from_rows(test_rows, probe_names),
            y_test,
            random_seed,
        ),
        "query_plus_probe": train_and_evaluate(
            matrix_from_rows(train_rows, query_names + probe_names),
            y_train,
            matrix_from_rows(test_rows, query_names + probe_names),
            y_test,
            random_seed,
        ),
    }
    return {
        method_name: summarize_router_predictions(
            method_name=method_name,
            test_rows=test_rows,
            predictions=output.predictions,
            probabilities=output.probabilities,
            top_k_values=top_k_values,
            auc=output.auc,
            accuracy=output.accuracy,
        )
        for method_name, output in outputs.items()
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


if __name__ == "__main__":
    main()
