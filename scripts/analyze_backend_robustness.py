from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from statistics import mean
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import probe_feature_names, query_feature_names
from src.model import make_cv_folds, make_split, train_and_evaluate
from src.utils import ensure_dir, load_yaml


FEATURE_SETS = {
    "query_only": query_feature_names(),
    "probe_only": probe_feature_names(),
    "query_plus_probe": query_feature_names() + probe_feature_names(),
}


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    name: str
    dense_backend: str
    graph_backend: str
    routing_rows_path: Path
    metrics_json_path: Path | None

    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze multi-backend routing robustness from saved routing rows.")
    parser.add_argument("--config", required=True, help="YAML config listing backend pairs and artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    project_root = Path(args.config).resolve().parent.parent
    output_dir = ensure_dir(resolve_path(project_root, str(config["output_dir"])))
    evaluation_mode = str(config.get("evaluation_mode", "train_test")).lower()
    random_seed = int(config.get("random_seed", 42))
    train_ratio = float(config.get("train_ratio", 0.8))
    num_folds = int(config.get("num_folds", 5))
    top_k_values = [int(value) for value in config.get("top_k_values", [3, 5])]
    feature_correlation_mode = str(config.get("feature_correlation_mode", "same_graph_dense_swap")).lower()

    pair_specs = load_pair_specs(config=config, project_root=project_root)
    pair_rows = {pair.key: load_labeled_rows(pair.routing_rows_path) for pair in pair_specs}
    pair_metrics = {
        pair.key: (load_json(pair.metrics_json_path) if pair.metrics_json_path and pair.metrics_json_path.exists() else None)
        for pair in pair_specs
    }

    overlap_rows = summarize_overlaps(pair_specs, pair_rows)
    transfer_rows = analyze_transfer_matrix(
        pair_specs=pair_specs,
        pair_rows=pair_rows,
        evaluation_mode=evaluation_mode,
        top_k_values=top_k_values,
        random_seed=random_seed,
        train_ratio=train_ratio,
        num_folds=num_folds,
    )
    correlation_rows = analyze_probe_correlations(
        pair_specs=pair_specs,
        pair_rows=pair_rows,
        mode=feature_correlation_mode,
    )
    consistency_rows = analyze_routing_consistency(
        pair_specs=pair_specs,
        pair_rows=pair_rows,
        pair_metrics=pair_metrics,
        top_k_values=top_k_values,
        evaluation_mode=evaluation_mode,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    write_csv(output_dir / "overlap_summary.csv", overlap_rows)
    write_csv(output_dir / "backend_transfer_matrix.csv", transfer_rows)
    write_csv(output_dir / "probe_feature_correlation.csv", correlation_rows)
    write_csv(output_dir / "routing_consistency_summary.csv", consistency_rows)

    payload = {
        "config": {
            "evaluation_mode": evaluation_mode,
            "random_seed": random_seed,
            "train_ratio": train_ratio,
            "num_folds": num_folds,
            "top_k_values": top_k_values,
            "feature_correlation_mode": feature_correlation_mode,
            "output_dir": str(output_dir),
        },
        "pairs": [pair_to_dict(pair) for pair in pair_specs],
        "num_overlap_rows": len(overlap_rows),
        "num_transfer_rows": len(transfer_rows),
        "num_correlation_rows": len(correlation_rows),
        "num_consistency_rows": len(consistency_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_pair_specs(config: dict[str, Any], project_root: Path) -> list[PairSpec]:
    specs: list[PairSpec] = []
    for dataset_cfg in config.get("datasets", []):
        dataset_name = str(dataset_cfg["name"])
        for pair_cfg in dataset_cfg.get("pairs", []):
            routing_rows_path = resolve_path(project_root, str(pair_cfg["routing_rows_path"]))
            metrics_raw = pair_cfg.get("metrics_json_path")
            metrics_path = resolve_path(project_root, str(metrics_raw)) if metrics_raw else None
            specs.append(
                PairSpec(
                    dataset=dataset_name,
                    name=str(pair_cfg["name"]),
                    dense_backend=str(pair_cfg["dense_backend"]),
                    graph_backend=str(pair_cfg["graph_backend"]),
                    routing_rows_path=routing_rows_path,
                    metrics_json_path=metrics_path,
                )
            )
    if not specs:
        raise ValueError("No backend pairs found in config.")
    return specs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labeled_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in load_jsonl(path) if row.get("label") is not None]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def pair_to_dict(pair: PairSpec) -> dict[str, Any]:
    return {
        "dataset": pair.dataset,
        "name": pair.name,
        "dense_backend": pair.dense_backend,
        "graph_backend": pair.graph_backend,
        "routing_rows_path": str(pair.routing_rows_path),
        "metrics_json_path": str(pair.metrics_json_path) if pair.metrics_json_path else None,
    }


def summarize_overlaps(pair_specs: list[PairSpec], pair_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, dataset_pairs in group_pairs_by_dataset(pair_specs).items():
        id_sets = {pair.key: {str(row["id"]) for row in pair_rows[pair.key]} for pair in dataset_pairs}
        for left, right in itertools.combinations(dataset_pairs, 2):
            overlap = sorted(id_sets[left.key] & id_sets[right.key])
            rows.append(
                {
                    "dataset": dataset,
                    "left_pair": left.name,
                    "right_pair": right.name,
                    "left_dense_backend": left.dense_backend,
                    "left_graph_backend": left.graph_backend,
                    "right_dense_backend": right.dense_backend,
                    "right_graph_backend": right.graph_backend,
                    "left_num_rows": len(id_sets[left.key]),
                    "right_num_rows": len(id_sets[right.key]),
                    "overlap_queries": len(overlap),
                }
            )
    return rows


def analyze_transfer_matrix(
    *,
    pair_specs: list[PairSpec],
    pair_rows: dict[str, list[dict[str, Any]]],
    evaluation_mode: str,
    top_k_values: list[int],
    random_seed: int,
    train_ratio: float,
    num_folds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, dataset_pairs in group_pairs_by_dataset(pair_specs).items():
        for train_pair, test_pair in itertools.product(dataset_pairs, dataset_pairs):
            metrics_by_method = evaluate_pair_transfer(
                train_rows=pair_rows[train_pair.key],
                test_rows=pair_rows[test_pair.key],
                evaluation_mode=evaluation_mode,
                top_k_values=top_k_values,
                random_seed=random_seed,
                train_ratio=train_ratio,
                num_folds=num_folds,
            )
            for method_name, method_metrics in metrics_by_method.items():
                row = {
                    "dataset": dataset,
                    "train_pair": train_pair.name,
                    "test_pair": test_pair.name,
                    "train_dense_backend": train_pair.dense_backend,
                    "train_graph_backend": train_pair.graph_backend,
                    "test_dense_backend": test_pair.dense_backend,
                    "test_graph_backend": test_pair.graph_backend,
                    "method": method_name,
                    **method_metrics,
                }
                rows.append(row)
    return rows


def evaluate_pair_transfer(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    evaluation_mode: str,
    top_k_values: list[int],
    random_seed: int,
    train_ratio: float,
    num_folds: int,
) -> dict[str, dict[str, Any]]:
    train_by_id = {str(row["id"]): row for row in train_rows}
    test_by_id = {str(row["id"]): row for row in test_rows}
    common_ids = sorted(set(train_by_id) & set(test_by_id))

    base = {
        "overlap_queries": len(common_ids),
        "split_mode": evaluation_mode,
    }
    if len(common_ids) < 4:
        return {name: {**base, "error": "Need at least 4 overlapping labeled queries."} for name in FEATURE_SETS}

    aligned_train = [train_by_id[sample_id] for sample_id in common_ids]
    aligned_test = [test_by_id[sample_id] for sample_id in common_ids]
    train_labels = [int(row["label"]) for row in aligned_train]
    test_labels = [int(row["label"]) for row in aligned_test]
    if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
        return {
            name: {
                **base,
                "error": "Both train and test pairs need both label classes on overlapping queries.",
                "train_label_0_count": train_labels.count(0),
                "train_label_1_count": train_labels.count(1),
                "test_label_0_count": test_labels.count(0),
                "test_label_1_count": test_labels.count(1),
            }
            for name in FEATURE_SETS
        }

    if evaluation_mode == "cv":
        return evaluate_transfer_cv(
            train_rows=aligned_train,
            test_rows=aligned_test,
            top_k_values=top_k_values,
            random_seed=random_seed,
            num_folds=num_folds,
        )
    return evaluate_transfer_split(
        train_rows=aligned_train,
        test_rows=aligned_test,
        top_k_values=top_k_values,
        random_seed=random_seed,
        train_ratio=train_ratio,
    )


def evaluate_transfer_split(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    top_k_values: list[int],
    random_seed: int,
    train_ratio: float,
) -> dict[str, dict[str, Any]]:
    train_labels = [int(row["label"]) for row in train_rows]
    sample_ids = [str(row["id"]) for row in train_rows]
    split = make_split(sample_ids=sample_ids, labels=train_labels, train_ratio=train_ratio, random_seed=random_seed)
    train_id_set = set(split.train_ids)
    test_id_set = set(split.test_ids)
    train_subset = [row for row in train_rows if str(row["id"]) in train_id_set]
    test_subset = [row for row in test_rows if str(row["id"]) in test_id_set]
    return train_transfer_models(
        train_subset=train_subset,
        test_subset=test_subset,
        top_k_values=top_k_values,
        random_seed=random_seed,
        split_mode=split.split_mode,
        overlap_queries=len(sample_ids),
    )


def evaluate_transfer_cv(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    top_k_values: list[int],
    random_seed: int,
    num_folds: int,
) -> dict[str, dict[str, Any]]:
    train_labels = [int(row["label"]) for row in train_rows]
    sample_ids = [str(row["id"]) for row in train_rows]
    class_counts = {label: train_labels.count(label) for label in sorted(set(train_labels))}
    if min(class_counts.values()) < num_folds:
        return {
            method_name: {
                "overlap_queries": len(sample_ids),
                "split_mode": "cross_validation",
                "error": f"Not enough samples in the minority class for {num_folds}-fold CV.",
            }
            for method_name in FEATURE_SETS
        }

    folds = make_cv_folds(sample_ids=sample_ids, labels=train_labels, num_folds=num_folds, random_seed=random_seed)
    per_method: dict[str, list[dict[str, Any]]] = {method_name: [] for method_name in FEATURE_SETS}
    train_by_id = {str(row["id"]): row for row in train_rows}
    test_by_id = {str(row["id"]): row for row in test_rows}
    for fold in folds:
        train_subset = [train_by_id[sample_id] for sample_id in fold.train_ids]
        test_subset = [test_by_id[sample_id] for sample_id in fold.test_ids]
        fold_metrics = train_transfer_models(
            train_subset=train_subset,
            test_subset=test_subset,
            top_k_values=top_k_values,
            random_seed=random_seed,
            split_mode="cross_validation",
            overlap_queries=len(sample_ids),
            aggregate_folds=False,
        )
        for method_name, payload in fold_metrics.items():
            per_method[method_name].append(payload)

    summary: dict[str, dict[str, Any]] = {}
    for method_name, fold_metrics in per_method.items():
        summary[method_name] = {
            "overlap_queries": len(sample_ids),
            "split_mode": "cross_validation",
            **summarize_router_folds(fold_metrics, top_k_values),
        }
    return summary


def train_transfer_models(
    *,
    train_subset: list[dict[str, Any]],
    test_subset: list[dict[str, Any]],
    top_k_values: list[int],
    random_seed: int,
    split_mode: str,
    overlap_queries: int,
    aggregate_folds: bool = True,
) -> dict[str, dict[str, Any]]:
    y_train = [int(row["label"]) for row in train_subset]
    y_test = [int(row["label"]) for row in test_subset]
    outputs = {}
    for method_name, feature_names in FEATURE_SETS.items():
        output = train_and_evaluate(
            matrix_from_rows(train_subset, feature_names),
            y_train,
            matrix_from_rows(test_subset, feature_names),
            y_test,
            random_seed,
        )
        metrics = summarize_router_predictions(
            method_name=method_name,
            test_rows=test_subset,
            predictions=output.predictions,
            probabilities=output.probabilities,
            top_k_values=top_k_values,
            auc=output.auc,
            accuracy=output.accuracy,
        )
        if aggregate_folds:
            outputs[method_name] = {
                "overlap_queries": overlap_queries,
                "train_samples": len(train_subset),
                "test_samples": len(test_subset),
                "split_mode": split_mode,
                **metrics,
            }
        else:
            outputs[method_name] = metrics
    return outputs


def analyze_probe_correlations(
    *,
    pair_specs: list[PairSpec],
    pair_rows: dict[str, list[dict[str, Any]]],
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, dataset_pairs in group_pairs_by_dataset(pair_specs).items():
        for left, right in itertools.combinations(dataset_pairs, 2):
            if mode == "same_graph_dense_swap":
                if left.graph_backend != right.graph_backend or left.dense_backend == right.dense_backend:
                    continue
            elif mode == "same_dense_graph_swap":
                if left.dense_backend != right.dense_backend or left.graph_backend == right.graph_backend:
                    continue
            elif mode != "all_pairs":
                raise ValueError(f"Unsupported feature_correlation_mode: {mode}")

            left_by_id = {str(row["id"]): row for row in pair_rows[left.key]}
            right_by_id = {str(row["id"]): row for row in pair_rows[right.key]}
            common_ids = sorted(set(left_by_id) & set(right_by_id))
            if len(common_ids) < 4:
                continue
            for feature_name in probe_feature_names():
                x = np.asarray([float(left_by_id[sample_id][feature_name]) for sample_id in common_ids], dtype=np.float64)
                y = np.asarray([float(right_by_id[sample_id][feature_name]) for sample_id in common_ids], dtype=np.float64)
                rows.append(
                    {
                        "dataset": dataset,
                        "left_pair": left.name,
                        "right_pair": right.name,
                        "left_dense_backend": left.dense_backend,
                        "left_graph_backend": left.graph_backend,
                        "right_dense_backend": right.dense_backend,
                        "right_graph_backend": right.graph_backend,
                        "feature": feature_name,
                        "overlap_queries": len(common_ids),
                        "spearman_rho": spearman_correlation(x, y),
                        "pearson_r": pearson_correlation(x, y),
                        "left_mean": float(np.mean(x)),
                        "right_mean": float(np.mean(y)),
                    }
                )
    return rows


def analyze_routing_consistency(
    *,
    pair_specs: list[PairSpec],
    pair_rows: dict[str, list[dict[str, Any]]],
    pair_metrics: dict[str, dict[str, Any] | None],
    top_k_values: list[int],
    evaluation_mode: str,
    train_ratio: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        metrics_payload = pair_metrics.get(pair.key)
        if not metrics_payload:
            continue
        retrieval_methods = metrics_payload.get("retrieval_methods", {})
        routing_methods = metrics_payload.get("routing_methods", {}).get("methods", {})
        combined = normalize_router_method_metrics(routing_methods.get("query_plus_probe", {}))
        probe_only = normalize_router_method_metrics(routing_methods.get("probe_only", {}))
        query_only = normalize_router_method_metrics(routing_methods.get("query_only", {}))
        row = {
            "dataset": pair.dataset,
            "pair": pair.name,
            "dense_backend": pair.dense_backend,
            "graph_backend": pair.graph_backend,
            "combined_graph_invocation_rate": combined.get("graph_invocation_rate"),
            "combined_auc": combined.get("auc"),
            "probe_only_graph_invocation_rate": probe_only.get("graph_invocation_rate"),
            "query_only_graph_invocation_rate": query_only.get("graph_invocation_rate"),
        }
        random_rows = random_baseline_rows(
            rows=pair_rows[pair.key],
            evaluation_mode=evaluation_mode,
            train_ratio=train_ratio,
            random_seed=random_seed,
        )
        for top_k in top_k_values:
            random_at_combined = evaluate_random_operating_point(
                random_rows,
                graph_rate=float(combined.get("graph_invocation_rate", 0.0)),
                top_k=top_k,
            )
            row.update(
                {
                    f"dense_recall@{top_k}": fetch_retrieval_metric(retrieval_methods, "dense_only", top_k),
                    f"graph_recall@{top_k}": fetch_retrieval_metric(retrieval_methods, "graph_only", top_k),
                    f"combined_recall@{top_k}": combined.get(f"recall@{top_k}"),
                    f"probe_only_recall@{top_k}": probe_only.get(f"recall@{top_k}"),
                    f"query_only_recall@{top_k}": query_only.get(f"recall@{top_k}"),
                    f"matched_random_recall@{top_k}": random_at_combined.get(f"recall@{top_k}"),
                    f"combined_minus_random_recall@{top_k}": (
                        float(combined.get(f"recall@{top_k}", 0.0))
                        - float(random_at_combined.get(f"recall@{top_k}", 0.0))
                    ),
                    f"graph_minus_dense_recall@{top_k}": (
                        float(fetch_retrieval_metric(retrieval_methods, "graph_only", top_k) or 0.0)
                        - float(fetch_retrieval_metric(retrieval_methods, "dense_only", top_k) or 0.0)
                    ),
                }
            )
        rows.append(row)
    return rows


def random_baseline_rows(
    *,
    rows: list[dict[str, Any]],
    evaluation_mode: str,
    train_ratio: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    if evaluation_mode == "cv":
        return rows
    labels = [int(row["label"]) for row in rows]
    sample_ids = [str(row["id"]) for row in rows]
    try:
        split = make_split(sample_ids=sample_ids, labels=labels, train_ratio=train_ratio, random_seed=random_seed)
    except ValueError:
        return rows
    test_id_set = set(split.test_ids)
    return [row for row in rows if str(row["id"]) in test_id_set]


def evaluate_random_operating_point(rows: list[dict[str, Any]], graph_rate: float, top_k: int) -> dict[str, float]:
    dense_latencies = [float(row["dense_latency_ms"]) for row in rows]
    graph_latencies = [float(row["graph_latency_ms"]) for row in rows]
    dense_recalls = [float(row[f"dense_recall@{top_k}"]) for row in rows]
    graph_recalls = [float(row[f"graph_recall@{top_k}"]) for row in rows]
    expected_latencies = [
        (1.0 - graph_rate) * dense_latency + graph_rate * graph_latency
        for dense_latency, graph_latency in zip(dense_latencies, graph_latencies)
    ]
    expected_recalls = [
        (1.0 - graph_rate) * dense_recall + graph_rate * graph_recall
        for dense_recall, graph_recall in zip(dense_recalls, graph_recalls)
    ]
    return {
        "threshold": float(graph_rate),
        "graph_invocation_rate": float(graph_rate),
        "avg_latency_ms": safe_mean(expected_latencies),
        "p95_latency_ms": percentile(expected_latencies, 95.0),
        f"recall@{top_k}": safe_mean(expected_recalls),
    }


def matrix_from_rows(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)


def summarize_router_predictions(
    method_name: str,
    test_rows: list[dict[str, Any]],
    predictions: list[int],
    probabilities: list[float],
    top_k_values: list[int],
    auc: float,
    accuracy: float,
) -> dict[str, Any]:
    graph_invocations = [int(prediction == 1) for prediction in predictions]
    latencies = [
        routed_latency_ms(method_name=method_name, row=row, choose_graph=prediction == 1)
        for row, prediction in zip(test_rows, predictions)
    ]
    llm_calls = [
        routed_llm_calls(method_name=method_name, row=row, choose_graph=prediction == 1)
        for row, prediction in zip(test_rows, predictions)
    ]
    token_costs = [
        routed_token_cost(method_name=method_name, row=row, choose_graph=prediction == 1)
        for row, prediction in zip(test_rows, predictions)
    ]
    payload = {
        "auc": float(auc),
        "routing_accuracy": float(accuracy),
        "graph_invocation_rate": safe_mean(graph_invocations),
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95.0),
        "avg_llm_calls_per_query": safe_mean(llm_calls),
        "avg_token_cost_per_query": safe_mean(token_costs),
        "avg_probability": safe_mean(probabilities),
    }
    for k in top_k_values:
        payload[f"recall@{k}"] = safe_mean(
            [
                float(row[f"graph_recall@{k}"] if prediction == 1 else row[f"dense_recall@{k}"])
                for row, prediction in zip(test_rows, predictions)
            ]
        )
    return payload


def summarize_router_folds(fold_metrics: list[dict[str, Any]], top_k_values: list[int]) -> dict[str, Any]:
    payload = {
        "auc_mean": safe_mean([float(metric["auc"]) for metric in fold_metrics]),
        "auc_std": safe_std([float(metric["auc"]) for metric in fold_metrics]),
        "routing_accuracy_mean": safe_mean([float(metric["routing_accuracy"]) for metric in fold_metrics]),
        "routing_accuracy_std": safe_std([float(metric["routing_accuracy"]) for metric in fold_metrics]),
        "graph_invocation_rate_mean": safe_mean([float(metric["graph_invocation_rate"]) for metric in fold_metrics]),
        "avg_latency_ms_mean": safe_mean([float(metric["avg_latency_ms"]) for metric in fold_metrics]),
        "p95_latency_ms_mean": safe_mean([float(metric["p95_latency_ms"]) for metric in fold_metrics]),
        "avg_llm_calls_per_query_mean": safe_mean([float(metric["avg_llm_calls_per_query"]) for metric in fold_metrics]),
        "avg_token_cost_per_query_mean": safe_mean([float(metric["avg_token_cost_per_query"]) for metric in fold_metrics]),
    }
    for k in top_k_values:
        payload[f"recall@{k}_mean"] = safe_mean([float(metric[f"recall@{k}"]) for metric in fold_metrics])
        payload[f"recall@{k}_std"] = safe_std([float(metric[f"recall@{k}"]) for metric in fold_metrics])
    return payload


def routed_latency_ms(method_name: str, row: dict[str, Any], choose_graph: bool) -> float:
    dense_latency = float(row["dense_latency_ms"])
    graph_latency = float(row["graph_latency_ms"])
    if method_name == "query_only":
        return graph_latency if choose_graph else dense_latency
    return dense_latency + (graph_latency if choose_graph else 0.0)


def routed_llm_calls(method_name: str, row: dict[str, Any], choose_graph: bool) -> float:
    dense_calls = float(row["dense_query_llm_calls"])
    graph_calls = float(row["graph_query_llm_calls"])
    if method_name == "query_only":
        return graph_calls if choose_graph else dense_calls
    return dense_calls + (graph_calls if choose_graph else 0.0)


def routed_token_cost(method_name: str, row: dict[str, Any], choose_graph: bool) -> float:
    dense_cost = float(row["dense_query_token_cost"])
    graph_cost = float(row["graph_query_token_cost"])
    if method_name == "query_only":
        return graph_cost if choose_graph else dense_cost
    return dense_cost + (graph_cost if choose_graph else 0.0)


def normalize_router_method_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    normalized = dict(payload)
    for key in list(payload):
        if key.endswith("_mean"):
            normalized[key[:-5]] = payload[key]
    return normalized


def fetch_retrieval_metric(methods: dict[str, Any], method_name: str, top_k: int) -> float | None:
    payload = methods.get(method_name, {})
    value = payload.get(f"recall@{top_k}")
    if value is None and f"recall@{top_k}_mean" in payload:
        value = payload.get(f"recall@{top_k}_mean")
    return float(value) if value is not None else None


def group_pairs_by_dataset(pair_specs: list[PairSpec]) -> dict[str, list[PairSpec]]:
    grouped: dict[str, list[PairSpec]] = {}
    for pair in pair_specs:
        grouped.setdefault(pair.dataset, []).append(pair)
    return grouped


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return pearson_correlation(rankdata_average_ties(x), rankdata_average_ties(y))


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


if __name__ == "__main__":
    main()
