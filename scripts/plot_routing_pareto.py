from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import probe_feature_names, query_feature_names
from src.model import make_cv_folds
from src.study_main import (
    matrix_from_rows,
    percentile,
    routed_latency_ms,
    safe_mean,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute out-of-fold routing probabilities, sweep routing thresholds, "
            "and render a Pareto curve over graph invocation rate or latency."
        )
    )
    parser.add_argument("--input", required=True, help="Path to routing_rows.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and operating-point tables.")
    parser.add_argument(
        "--policy",
        default="strict",
        choices=["strict", "moderate", "lenient", "threshold"],
        help="Dense sufficiency policy for correction labels.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Explicit threshold when --policy=threshold.")
    parser.add_argument("--label-k", type=int, default=5, help="Label on dense_recall@k vs graph_recall@k.")
    parser.add_argument("--top-k", type=int, default=5, help="Recall@k to plot on the y-axis.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--threshold-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated routing thresholds to sweep.",
    )
    parser.add_argument(
        "--x-axis",
        default="graph_invocation_rate",
        choices=["graph_invocation_rate", "avg_latency_ms"],
        help="Metric to use for the x-axis.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional retrieval metrics.json to annotate fixed dense/graph/rrf baselines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    threshold = resolve_threshold(args.policy, args.threshold)
    relabeled_rows = [
        relabel_row(row=row, label_k=args.label_k, policy=args.policy, threshold=threshold)
        for row in rows
    ]
    valid_rows = [row for row in relabeled_rows if row["label"] is not None]
    if len(valid_rows) < args.num_folds:
        raise ValueError(f"Not enough valid rows for {args.num_folds}-fold CV: {len(valid_rows)}")

    query_names = query_feature_names()
    probe_names = probe_feature_names()
    method_features = {
        "query_only": query_names,
        "probe_only": probe_names,
        "query_plus_probe": query_names + probe_names,
    }

    oof_predictions = compute_oof_predictions(
        rows=valid_rows,
        method_features=method_features,
        num_folds=args.num_folds,
        random_seed=args.random_seed,
    )
    thresholds = parse_threshold_grid(args.threshold_grid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    operating_points = {
        method_name: [
            evaluate_operating_point(
                method_name=method_name,
                rows=valid_rows,
                probabilities=oof_predictions[method_name],
                decision_threshold=decision_threshold,
                top_k=args.top_k,
            )
            for decision_threshold in thresholds
        ]
        for method_name in method_features
    }

    random_points = [
        evaluate_random_operating_point(valid_rows, graph_rate=graph_rate, top_k=args.top_k)
        for graph_rate in thresholds
    ]

    fixed_baselines = compute_fixed_baselines(valid_rows, top_k=args.top_k)
    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    retrieval_baselines = load_retrieval_baselines(metrics_path, top_k=args.top_k) if metrics_path else {}

    payload = {
        "input": args.input,
        "label_k": args.label_k,
        "top_k": args.top_k,
        "correction_label_policy": args.policy,
        "correction_threshold": threshold,
        "num_folds": args.num_folds,
        "x_axis": args.x_axis,
        "thresholds": thresholds,
        "label_stats": {
            "total_samples": len(relabeled_rows),
            "valid_samples": len(valid_rows),
            "tie_samples": len(relabeled_rows) - len(valid_rows),
            "label_0_count": sum(1 for row in valid_rows if int(row["label"]) == 0),
            "label_1_count": sum(1 for row in valid_rows if int(row["label"]) == 1),
        },
        "retrieval_baselines": retrieval_baselines,
        "fixed_valid_set_baselines": fixed_baselines,
        "operating_points": operating_points,
        "random_router": random_points,
    }

    with (output_dir / "pareto_operating_points.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_oof_predictions(output_dir / "oof_predictions.jsonl", valid_rows, oof_predictions)
    render_plot(
        output_path=output_dir / "pareto_curve.png",
        x_axis=args.x_axis,
        top_k=args.top_k,
        operating_points=operating_points,
        random_points=random_points,
        fixed_baselines=fixed_baselines,
        retrieval_baselines=retrieval_baselines,
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_threshold(policy: str, threshold: float | None) -> float:
    if policy == "strict":
        return 1.0
    if policy == "moderate":
        return 0.5
    if policy == "lenient":
        return np.nextafter(0.0, 1.0)
    if threshold is None:
        raise ValueError("--threshold is required when --policy=threshold")
    return float(threshold)


def relabel_row(row: dict[str, Any], label_k: int, policy: str, threshold: float) -> dict[str, Any]:
    dense_key = f"dense_recall@{label_k}"
    graph_key = f"graph_recall@{label_k}"
    dense_recall = float(row[dense_key])
    graph_recall = float(row[graph_key])
    if dense_recall >= threshold:
        label = 0
        label_reason = "dense_sufficient"
    elif graph_recall > dense_recall:
        label = 1
        label_reason = "graph_correction_needed"
    else:
        label = None
        label_reason = "discard_dense_insufficient_graph_not_better"

    payload = dict(row)
    payload["label"] = label
    payload["label_reason"] = label_reason
    payload["correction_label_policy"] = policy
    payload["correction_threshold"] = threshold
    payload["label_k"] = label_k
    return payload


def parse_threshold_grid(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Thresholds must be between 0 and 1. Got {threshold}.")
    return thresholds


def compute_oof_predictions(
    rows: list[dict[str, Any]],
    method_features: dict[str, list[str]],
    num_folds: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    sample_ids = [str(row["id"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    folds = make_cv_folds(sample_ids=sample_ids, labels=labels, num_folds=num_folds, random_seed=random_seed)

    row_by_id = {str(row["id"]): row for row in rows}
    predictions = {method_name: {} for method_name in method_features}

    for fold in folds:
        train_rows = [row_by_id[sample_id] for sample_id in fold.train_ids]
        test_rows = [row_by_id[sample_id] for sample_id in fold.test_ids]
        y_train = [int(row["label"]) for row in train_rows]

        for method_name, feature_names in method_features.items():
            classifier = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(random_state=random_seed, max_iter=1000)),
                ]
            )
            classifier.fit(matrix_from_rows(train_rows, feature_names), y_train)
            probabilities = classifier.predict_proba(matrix_from_rows(test_rows, feature_names))[:, 1].tolist()
            for row, probability in zip(test_rows, probabilities):
                predictions[method_name][str(row["id"])] = float(probability)

    return predictions


def evaluate_operating_point(
    method_name: str,
    rows: list[dict[str, Any]],
    probabilities: dict[str, float],
    decision_threshold: float,
    top_k: int,
) -> dict[str, float]:
    chooses_graph = [float(probabilities[str(row["id"])] >= decision_threshold) for row in rows]
    latencies = [
        routed_latency_ms(method_name=method_name, row=row, choose_graph=bool(choose_graph))
        for row, choose_graph in zip(rows, chooses_graph)
    ]
    recalls = [
        float(row[f"graph_recall@{top_k}"] if choose_graph else row[f"dense_recall@{top_k}"])
        for row, choose_graph in zip(rows, chooses_graph)
    ]
    return {
        "threshold": float(decision_threshold),
        "graph_invocation_rate": safe_mean(chooses_graph),
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95.0),
        f"recall@{top_k}": safe_mean(recalls),
    }


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


def compute_fixed_baselines(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float]]:
    dense_latencies = [float(row["dense_latency_ms"]) for row in rows]
    graph_latencies = [float(row["graph_latency_ms"]) for row in rows]
    fusion_latencies = [float(row["dense_latency_ms"]) + float(row["graph_latency_ms"]) for row in rows]

    return {
        "dense_only": {
            "graph_invocation_rate": 0.0,
            "avg_latency_ms": safe_mean(dense_latencies),
            "p95_latency_ms": percentile(dense_latencies, 95.0),
            f"recall@{top_k}": safe_mean([float(row[f"dense_recall@{top_k}"]) for row in rows]),
        },
        "graph_only": {
            "graph_invocation_rate": 1.0,
            "avg_latency_ms": safe_mean(graph_latencies),
            "p95_latency_ms": percentile(graph_latencies, 95.0),
            f"recall@{top_k}": safe_mean([float(row[f"graph_recall@{top_k}"]) for row in rows]),
        },
        "dense_graph_rrf": {
            "graph_invocation_rate": 1.0,
            "avg_latency_ms": safe_mean(fusion_latencies),
            "p95_latency_ms": percentile(fusion_latencies, 95.0),
            f"recall@{top_k}": safe_mean([float(row[f"fusion_recall@{top_k}"]) for row in rows]),
        },
    }


def load_retrieval_baselines(path: Path, top_k: int) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("retrieval_methods", {})
    baseline_names = {
        "dense_only": "dense_only",
        "graph_only": "graph_only",
        "dense_graph_rrf": "dense_graph_rrf",
    }
    baselines: dict[str, dict[str, float]] = {}
    for output_name, source_name in baseline_names.items():
        source = methods.get(source_name)
        if not source:
            continue
        baselines[output_name] = {
            "graph_invocation_rate": float(source["graph_invocation_rate"]),
            "avg_latency_ms": float(source["avg_latency_ms"]),
            "p95_latency_ms": float(source["p95_latency_ms"]),
            f"recall@{top_k}": float(source[f"recall@{top_k}"]),
        }
    return baselines


def write_oof_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, float]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row["id"])
            payload = {
                "id": sample_id,
                "label": int(row["label"]),
                "dense_recall@5": float(row.get("dense_recall@5", 0.0)),
                "graph_recall@5": float(row.get("graph_recall@5", 0.0)),
                "query_only_probability": float(predictions["query_only"][sample_id]),
                "probe_only_probability": float(predictions["probe_only"][sample_id]),
                "query_plus_probe_probability": float(predictions["query_plus_probe"][sample_id]),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_plot(
    output_path: Path,
    x_axis: str,
    top_k: int,
    operating_points: dict[str, list[dict[str, float]]],
    random_points: list[dict[str, float]],
    fixed_baselines: dict[str, dict[str, float]],
    retrieval_baselines: dict[str, dict[str, float]],
) -> None:
    plt.figure(figsize=(8.5, 6.5))

    curve_styles = {
        "query_only": {"label": "Query-only router", "color": "#1f77b4"},
        "probe_only": {"label": "Probe router", "color": "#ff7f0e"},
        "query_plus_probe": {"label": "Combined router", "color": "#2ca02c"},
    }
    for method_name, points in operating_points.items():
        points = sorted(points, key=lambda item: float(item[x_axis]))
        plt.plot(
            [float(point[x_axis]) for point in points],
            [float(point[f"recall@{top_k}"]) for point in points],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            **curve_styles[method_name],
        )

    random_points = sorted(random_points, key=lambda item: float(item[x_axis]))
    plt.plot(
        [float(point[x_axis]) for point in random_points],
        [float(point[f"recall@{top_k}"]) for point in random_points],
        linestyle="--",
        linewidth=1.4,
        color="#7f7f7f",
        label="Random router",
    )

    baseline_markers = {
        "dense_only": ("Dense-only", "D", "#9467bd"),
        "graph_only": ("Graph-only", "s", "#d62728"),
        "dense_graph_rrf": ("RRF", "^", "#8c564b"),
    }
    baseline_source = retrieval_baselines or fixed_baselines
    for baseline_name, (label, marker, color) in baseline_markers.items():
        point = baseline_source.get(baseline_name)
        if not point:
            continue
        plt.scatter(
            [float(point[x_axis])],
            [float(point[f"recall@{top_k}"])],
            s=70,
            marker=marker,
            color=color,
            label=label,
            zorder=5,
        )

    plt.xlabel("Graph invocation rate" if x_axis == "graph_invocation_rate" else "Average latency (ms)")
    plt.ylabel(f"Recall@{top_k}")
    plt.title(f"Routing Pareto Frontier (Recall@{top_k})")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
