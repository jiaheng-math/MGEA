from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.dataset import build_corpus, load_dataset, load_shared_corpus
from src.features import extract_probe_features, extract_query_features, probe_feature_names, query_feature_names
from src.labeling import build_oracle_label, summarize_labels
from src.model import make_cv_folds, make_split, train_and_evaluate
from src.retrieval_dense import build_dense_retriever
from src.retrieval_graph import build_graph_retriever
from src.utils import ensure_dir, load_spacy_model, load_yaml, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Pilot 0 routing experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_random_seed(int(config["random_seed"]))

    project_root = Path(args.config).resolve().parent.parent
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = ensure_dir(output_dir)
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path

    samples = load_dataset(
        dataset_path=str(dataset_path),
        subset_size=int(config["subset_size"]),
        random_seed=int(config["random_seed"]),
    )
    nlp = load_spacy_model()

    shared_corpus_path = config.get("shared_corpus_path")
    if shared_corpus_path:
        corpus_path = Path(str(shared_corpus_path))
        if not corpus_path.is_absolute():
            corpus_path = project_root / corpus_path
        corpus = load_shared_corpus(str(corpus_path))
    else:
        corpus = build_corpus(samples)
    dense_retriever = build_dense_retriever(corpus=corpus, config=config, project_root=project_root)
    graph_retriever = build_graph_retriever(corpus=corpus, nlp=nlp, config=config, project_root=project_root)

    query_names = query_feature_names()
    probe_names = probe_feature_names()
    rows: list[dict] = []
    tie_rows: list[dict] = []
    tie_count = 0
    top_k = int(config["top_k"])

    for sample in samples:
        dense_results = dense_retriever.retrieve(sample.question, top_k=top_k)
        graph_results = graph_retriever.retrieve(sample.question, top_k=top_k)
        label_result = build_oracle_label(sample, dense_results, graph_results)
        overlap_stats = summarize_retrieval_overlap(dense_results, graph_results, top_k=top_k)
        if label_result.label is None:
            tie_count += 1
            tie_rows.append(
                {
                    "id": sample.id,
                    "question": sample.question,
                    "dense_quality": float(label_result.dense_quality),
                    "graph_quality": float(label_result.graph_quality),
                    **overlap_stats,
                }
            )
            continue

        query_features, query_entities = extract_query_features(sample.question, nlp)
        probe_features = extract_probe_features(query_entities, dense_results)
        row = {
            "id": sample.id,
            "question": sample.question,
            "label": int(label_result.label),
            "dense_quality": float(label_result.dense_quality),
            "graph_quality": float(label_result.graph_quality),
            **query_features,
            **probe_features,
        }
        rows.append(row)

    label_summary = summarize_labels(rows)
    total_samples = len(samples)
    valid_samples = label_summary["valid_samples"]
    tie_analysis = summarize_tie_rows(tie_rows)
    partial_metrics = {
        "config": {
            "dataset_path": str(dataset_path),
            "subset_size": int(config["subset_size"]),
            "top_k": top_k,
            "random_seed": int(config["random_seed"]),
            "train_ratio": float(config["train_ratio"]),
            "output_dir": str(output_dir),
            "dense_backend": str(config.get("dense_backend", "sentence_transformers")),
            "dense_model_name": str(config.get("dense_model_name", "")) or None,
            "colbert_checkpoint": str(config.get("colbert_checkpoint", "")) or None,
            "graph_backend": str(config.get("graph_backend", "simplified")),
            "evaluation_mode": str(config.get("evaluation_mode", "train_test")).lower(),
            "num_folds": int(config.get("num_folds", 5)),
        },
        "label_stats": {
            "total_samples": total_samples,
            "valid_training_samples": valid_samples,
            "tie_samples": tie_count,
            "label_0_count": label_summary["label_0_count"],
            "label_1_count": label_summary["label_1_count"],
        },
        "tie_analysis": tie_analysis,
    }
    write_metrics(output_dir / "partial_metrics.json", partial_metrics)
    write_tie_analysis(output_dir / "tie_analysis.csv", tie_rows)
    if valid_samples < 4:
        raise ValueError(f"Not enough valid labeled samples after tie filtering: {valid_samples}")

    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("Need both label classes after tie filtering to compute AUC.")

    sample_ids = [str(row["id"]) for row in rows]
    evaluation_mode = str(config.get("evaluation_mode", "train_test")).lower()

    if evaluation_mode == "cv":
        evaluation = run_cross_validation(
            rows=rows,
            query_names=query_names,
            probe_names=probe_names,
            num_folds=int(config.get("num_folds", 5)),
            random_seed=int(config["random_seed"]),
        )
    else:
        evaluation = run_single_split(
            rows=rows,
            query_names=query_names,
            probe_names=probe_names,
            train_ratio=float(config["train_ratio"]),
            random_seed=int(config["random_seed"]),
        )

    metrics = {
        "config": {
            "dataset_path": str(dataset_path),
            "subset_size": int(config["subset_size"]),
            "top_k": top_k,
            "random_seed": int(config["random_seed"]),
            "train_ratio": float(config["train_ratio"]),
            "output_dir": str(output_dir),
            "dense_backend": str(config.get("dense_backend", "sentence_transformers")),
            "dense_model_name": str(config.get("dense_model_name", "")) or None,
            "colbert_checkpoint": str(config.get("colbert_checkpoint", "")) or None,
            "graph_backend": str(config.get("graph_backend", "simplified")),
            "evaluation_mode": evaluation_mode,
            "num_folds": int(config.get("num_folds", 5)),
        },
        "label_stats": {
            "total_samples": total_samples,
            "valid_training_samples": valid_samples,
            "tie_samples": tie_count,
            "label_0_count": label_summary["label_0_count"],
            "label_1_count": label_summary["label_1_count"],
            **evaluation["label_stats"],
        },
        "baselines": {
            **evaluation["baselines"],
        },
        "metrics": {
            **evaluation["metrics"],
        },
        "tie_analysis": tie_analysis,
    }
    if "folds" in evaluation:
        metrics["folds"] = evaluation["folds"]

    print("=== Label Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"Valid training samples: {valid_samples}")
    print(f"Tie samples: {tie_count}")
    print(f"Label 0 count (dense better): {label_summary['label_0_count']}")
    print(f"Label 1 count (graph better): {label_summary['label_1_count']}")
    print(
        "Tie overlap | "
        f"avg overlap@k: {tie_analysis['avg_overlap_rate']:.4f} | "
        f"avg jaccard: {tie_analysis['avg_jaccard']:.4f} | "
        f"high-overlap ties (>=0.6): {tie_analysis['high_overlap_tie_count']}/{tie_analysis['tie_count']}"
    )
    for key, value in evaluation["label_stats"].items():
        print(f"{format_label_key(key)}: {value}")
    print(
        f"Majority baseline | label: {evaluation['baselines']['majority_label']} | "
        f"AUC: {evaluation['baselines']['majority_auc']:.4f} | "
        f"Accuracy: {evaluation['baselines']['majority_accuracy']:.4f}"
    )
    print()
    print("=== Test Metrics ===")
    print_metric_line("Query-only", evaluation["metrics"]["query_only"])
    print_metric_line("Probe-only", evaluation["metrics"]["probe_only"])
    print_metric_line("Query + Probe", evaluation["metrics"]["query_plus_probe"])

    write_metrics(output_dir / "metrics.json", metrics)
    write_predictions(
        output_dir / "predictions.csv",
        evaluation["prediction_rows"],
    )


def matrix_from_rows(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = [[float(row[name]) for name in feature_names] for row in rows]
    return np.asarray(matrix, dtype=np.float64)


def run_single_split(
    rows: list[dict],
    query_names: list[str],
    probe_names: list[str],
    train_ratio: float,
    random_seed: int,
) -> dict:
    labels = [int(row["label"]) for row in rows]
    sample_ids = [str(row["id"]) for row in rows]
    split_result = make_split(
        sample_ids=sample_ids,
        labels=labels,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )
    train_id_set = set(split_result.train_ids)
    test_id_set = set(split_result.test_ids)
    train_rows = [row for row in rows if row["id"] in train_id_set]
    test_rows = [row for row in rows if row["id"] in test_id_set]

    query_output, probe_output, combined_output = train_three_models(
        train_rows=train_rows,
        test_rows=test_rows,
        query_names=query_names,
        probe_names=probe_names,
        random_seed=random_seed,
    )
    y_test = [int(row["label"]) for row in test_rows]
    majority_label = max(set(y_test), key=y_test.count)
    majority_accuracy = sum(1 for label in y_test if label == majority_label) / len(y_test)

    return {
        "label_stats": {
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "split_mode": split_result.split_mode,
        },
        "baselines": {
            "majority_label": int(majority_label),
            "majority_accuracy": float(majority_accuracy),
            "majority_auc": 0.5,
        },
        "metrics": {
            "query_only": {"auc": query_output.auc, "accuracy": query_output.accuracy},
            "probe_only": {"auc": probe_output.auc, "accuracy": probe_output.accuracy},
            "query_plus_probe": {"auc": combined_output.auc, "accuracy": combined_output.accuracy},
        },
        "prediction_rows": build_prediction_rows(
            rows=test_rows,
            prob_query=query_output.probabilities,
            prob_probe=probe_output.probabilities,
            prob_combined=combined_output.probabilities,
            pred_query=query_output.predictions,
            pred_probe=probe_output.predictions,
            pred_combined=combined_output.predictions,
        ),
    }


def run_cross_validation(
    rows: list[dict],
    query_names: list[str],
    probe_names: list[str],
    num_folds: int,
    random_seed: int,
) -> dict:
    labels = [int(row["label"]) for row in rows]
    sample_ids = [str(row["id"]) for row in rows]
    folds = make_cv_folds(
        sample_ids=sample_ids,
        labels=labels,
        num_folds=num_folds,
        random_seed=random_seed,
    )

    fold_records: list[dict] = []
    prediction_rows: list[dict] = []
    baseline_accuracies: list[float] = []
    baseline_labels: list[int] = []

    for fold in folds:
        train_id_set = set(fold.train_ids)
        test_id_set = set(fold.test_ids)
        train_rows = [row for row in rows if row["id"] in train_id_set]
        test_rows = [row for row in rows if row["id"] in test_id_set]
        query_output, probe_output, combined_output = train_three_models(
            train_rows=train_rows,
            test_rows=test_rows,
            query_names=query_names,
            probe_names=probe_names,
            random_seed=random_seed,
        )
        y_test = [int(row["label"]) for row in test_rows]
        majority_label = max(set(y_test), key=y_test.count)
        majority_accuracy = sum(1 for label in y_test if label == majority_label) / len(y_test)
        baseline_labels.append(int(majority_label))
        baseline_accuracies.append(float(majority_accuracy))

        fold_records.append(
            {
                "fold_index": fold.fold_index,
                "train_samples": len(train_rows),
                "test_samples": len(test_rows),
                "majority_label": int(majority_label),
                "majority_accuracy": float(majority_accuracy),
                "query_only": {"auc": query_output.auc, "accuracy": query_output.accuracy},
                "probe_only": {"auc": probe_output.auc, "accuracy": probe_output.accuracy},
                "query_plus_probe": {"auc": combined_output.auc, "accuracy": combined_output.accuracy},
            }
        )
        prediction_rows.extend(
            build_prediction_rows(
                rows=test_rows,
                prob_query=query_output.probabilities,
                prob_probe=probe_output.probabilities,
                prob_combined=combined_output.probabilities,
                pred_query=query_output.predictions,
                pred_probe=probe_output.predictions,
                pred_combined=combined_output.predictions,
                extra_fields={"fold": fold.fold_index},
            )
        )

    prediction_rows.sort(key=lambda row: row["id"])
    return {
        "label_stats": {
            "num_folds": num_folds,
            "train_samples_per_fold": [fold["train_samples"] for fold in fold_records],
            "test_samples_per_fold": [fold["test_samples"] for fold in fold_records],
            "split_mode": "cross_validation",
        },
        "baselines": {
            "majority_label": baseline_labels,
            "majority_accuracy": float(np.mean(baseline_accuracies)),
            "majority_auc": 0.5,
        },
        "metrics": {
            "query_only": summarize_fold_metric(fold_records, "query_only"),
            "probe_only": summarize_fold_metric(fold_records, "probe_only"),
            "query_plus_probe": summarize_fold_metric(fold_records, "query_plus_probe"),
        },
        "folds": fold_records,
        "prediction_rows": prediction_rows,
    }


def train_three_models(
    train_rows: list[dict],
    test_rows: list[dict],
    query_names: list[str],
    probe_names: list[str],
    random_seed: int,
):
    x_train_query = matrix_from_rows(train_rows, query_names)
    x_test_query = matrix_from_rows(test_rows, query_names)
    x_train_probe = matrix_from_rows(train_rows, probe_names)
    x_test_probe = matrix_from_rows(test_rows, probe_names)
    x_train_combined = matrix_from_rows(train_rows, query_names + probe_names)
    x_test_combined = matrix_from_rows(test_rows, query_names + probe_names)
    y_train = [int(row["label"]) for row in train_rows]
    y_test = [int(row["label"]) for row in test_rows]

    query_output = train_and_evaluate(x_train_query, y_train, x_test_query, y_test, random_seed)
    probe_output = train_and_evaluate(x_train_probe, y_train, x_test_probe, y_test, random_seed)
    combined_output = train_and_evaluate(x_train_combined, y_train, x_test_combined, y_test, random_seed)
    return query_output, probe_output, combined_output


def summarize_fold_metric(fold_records: list[dict], key: str) -> dict:
    aucs = [fold[key]["auc"] for fold in fold_records]
    accuracies = [fold[key]["accuracy"] for fold in fold_records]
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
    }


def build_prediction_rows(
    rows: list[dict],
    prob_query: list[float],
    prob_probe: list[float],
    prob_combined: list[float],
    pred_query: list[int],
    pred_probe: list[int],
    pred_combined: list[int],
    extra_fields: dict | None = None,
) -> list[dict]:
    output: list[dict] = []
    extra_fields = extra_fields or {}
    for idx, row in enumerate(rows):
        output.append(
            {
                "id": row["id"],
                "question": row["question"],
                "label": row["label"],
                "prob_query": prob_query[idx],
                "prob_probe": prob_probe[idx],
                "prob_combined": prob_combined[idx],
                "pred_query": pred_query[idx],
                "pred_probe": pred_probe[idx],
                "pred_combined": pred_combined[idx],
                **extra_fields,
            }
        )
    return output


def print_metric_line(label: str, metric_payload: dict) -> None:
    if "auc_mean" in metric_payload:
        print(
            f"{label:<15} | AUC: {metric_payload['auc_mean']:.4f} ± {metric_payload['auc_std']:.4f} | "
            f"Accuracy: {metric_payload['accuracy_mean']:.4f} ± {metric_payload['accuracy_std']:.4f}"
        )
    else:
        print(f"{label:<15} | AUC: {metric_payload['auc']:.4f} | Accuracy: {metric_payload['accuracy']:.4f}")


def format_label_key(key: str) -> str:
    return key.replace("_", " ").capitalize()


def write_metrics(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_predictions(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "question",
        "label",
        "fold",
        "prob_query",
        "prob_probe",
        "prob_combined",
        "pred_query",
        "pred_probe",
        "pred_combined",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "label": row["label"],
                    "fold": row.get("fold", ""),
                    "prob_query": row["prob_query"],
                    "prob_probe": row["prob_probe"],
                    "prob_combined": row["prob_combined"],
                    "pred_query": row["pred_query"],
                    "pred_probe": row["pred_probe"],
                    "pred_combined": row["pred_combined"],
                }
            )


def summarize_retrieval_overlap(dense_results: list, graph_results: list, top_k: int) -> dict:
    dense_ids = [passage.id for passage in dense_results[:top_k]]
    graph_ids = [passage.id for passage in graph_results[:top_k]]
    dense_set = set(dense_ids)
    graph_set = set(graph_ids)
    intersection = dense_set & graph_set
    union = dense_set | graph_set
    denominator = max(1, min(top_k, len(dense_ids), len(graph_ids)))
    return {
        "overlap_count": len(intersection),
        "overlap_rate": len(intersection) / denominator,
        "jaccard": len(intersection) / max(1, len(union)),
        "dense_ids": " || ".join(dense_ids),
        "graph_ids": " || ".join(graph_ids),
    }


def summarize_tie_rows(tie_rows: list[dict]) -> dict:
    if not tie_rows:
        return {
            "tie_count": 0,
            "avg_overlap_rate": 0.0,
            "avg_jaccard": 0.0,
            "high_overlap_tie_count": 0,
        }
    overlap_rates = [float(row["overlap_rate"]) for row in tie_rows]
    jaccards = [float(row["jaccard"]) for row in tie_rows]
    high_overlap = sum(1 for row in tie_rows if float(row["overlap_rate"]) >= 0.6)
    return {
        "tie_count": len(tie_rows),
        "avg_overlap_rate": float(np.mean(overlap_rates)),
        "avg_jaccard": float(np.mean(jaccards)),
        "high_overlap_tie_count": int(high_overlap),
    }


def write_tie_analysis(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "question",
        "dense_quality",
        "graph_quality",
        "overlap_count",
        "overlap_rate",
        "jaccard",
        "dense_ids",
        "graph_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


if __name__ == "__main__":
    main()
