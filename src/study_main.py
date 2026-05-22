from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.dataset import QASample, build_corpus, load_dataset, load_shared_corpus
from src.features import extract_probe_features, extract_query_features, probe_feature_names, query_feature_names
from src.model import make_cv_folds, make_split, train_and_evaluate
from src.retrieval_dense import build_dense_retriever
from src.retrieval_graph import build_graph_retriever
from src.utils import RetrievedPassage, ensure_dir, load_spacy_model, load_yaml, normalize_text, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runner for shared-corpus retrieval and routing studies.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_random_seed(int(config["random_seed"]))

    project_root = Path(args.config).resolve().parent.parent
    output_dir = resolve_path(project_root, str(config["output_dir"]))
    output_dir = ensure_dir(output_dir)
    dataset_path = resolve_path(project_root, str(config["dataset_path"]))

    samples = load_dataset(
        dataset_path=str(dataset_path),
        subset_size=int(config["subset_size"]),
        random_seed=int(config["random_seed"]),
    )
    shared_corpus_path = config.get("shared_corpus_path")
    if shared_corpus_path:
        corpus = load_shared_corpus(str(resolve_path(project_root, str(shared_corpus_path))))
    else:
        corpus = build_corpus(samples)

    nlp = load_spacy_model()
    dense_retriever = build_dense_retriever(corpus=corpus, config=config, project_root=project_root)
    graph_retriever = build_graph_retriever(corpus=corpus, nlp=nlp, config=config, project_root=project_root)

    top_k_values = sorted({int(value) for value in config.get("top_k_values", [3, 5])})
    label_top_k = int(config.get("label_top_k", max(top_k_values)))
    retrieval_depth = max(top_k_values + [label_top_k])

    rows = collect_rows(
        samples=samples,
        dense_retriever=dense_retriever,
        graph_retriever=graph_retriever,
        nlp=nlp,
        graph_backend=str(config.get("graph_backend", "bm25")),
        retrieval_depth=retrieval_depth,
        top_k_values=top_k_values,
        label_top_k=label_top_k,
        dense_query_llm_calls=float(config.get("dense_query_llm_calls", 0.0)),
        graph_query_llm_calls=float(config.get("graph_query_llm_calls", 0.0)),
        dense_query_token_cost=float(config.get("dense_query_token_cost", 0.0)),
        graph_query_token_cost=float(config.get("graph_query_token_cost", 0.0)),
        correction_label_policy=str(config.get("correction_label_policy", "strict")).lower(),
        correction_threshold=(
            float(config["correction_threshold"])
            if "correction_threshold" in config
            else None
        ),
    )

    retrieval_rows = [retrieval_row_from_full_row(row) for row in rows]
    routing_rows = [routing_row_from_full_row(row) for row in rows]

    write_jsonl(output_dir / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(output_dir / "routing_rows.jsonl", routing_rows)
    retrieval_summary = summarize_retrieval_methods(rows=routing_rows, top_k_values=top_k_values)
    partial_payload = {
        "config": {
            "dataset_path": str(dataset_path),
            "shared_corpus_path": str(resolve_path(project_root, str(shared_corpus_path))) if shared_corpus_path else None,
            "subset_size": int(config["subset_size"]),
            "top_k_values": top_k_values,
            "label_top_k": label_top_k,
            "random_seed": int(config["random_seed"]),
            "dense_backend": str(config.get("dense_backend", "sentence_transformers")),
            "dense_model_name": str(config.get("dense_model_name", "")) or None,
            "colbert_checkpoint": str(config.get("colbert_checkpoint", "")) or None,
            "graph_backend": str(config.get("graph_backend", "bm25")),
            "evaluation_mode": str(config.get("evaluation_mode", "train_test")).lower(),
            "num_folds": int(config.get("num_folds", 5)),
            "correction_label_policy": str(config.get("correction_label_policy", "strict")).lower(),
            "correction_threshold": (
                float(config["correction_threshold"])
                if "correction_threshold" in config
                else None
            ),
            "output_dir": str(output_dir),
        },
        "dataset_summary": summarize_dataset_rows(routing_rows),
        "retrieval_methods": retrieval_summary,
    }
    with (output_dir / "partial_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(partial_payload, handle, indent=2, ensure_ascii=False)

    routing_summary = summarize_routing_methods(
        rows=routing_rows,
        top_k_values=top_k_values,
        train_ratio=float(config.get("train_ratio", 0.8)),
        random_seed=int(config["random_seed"]),
        evaluation_mode=str(config.get("evaluation_mode", "train_test")).lower(),
        num_folds=int(config.get("num_folds", 5)),
    )

    payload = {
        "config": {
            "dataset_path": str(dataset_path),
            "shared_corpus_path": str(resolve_path(project_root, str(shared_corpus_path))) if shared_corpus_path else None,
            "subset_size": int(config["subset_size"]),
            "top_k_values": top_k_values,
            "label_top_k": label_top_k,
            "random_seed": int(config["random_seed"]),
            "dense_backend": str(config.get("dense_backend", "sentence_transformers")),
            "dense_model_name": str(config.get("dense_model_name", "")) or None,
            "colbert_checkpoint": str(config.get("colbert_checkpoint", "")) or None,
            "graph_backend": str(config.get("graph_backend", "bm25")),
            "evaluation_mode": str(config.get("evaluation_mode", "train_test")).lower(),
            "num_folds": int(config.get("num_folds", 5)),
            "correction_label_policy": str(config.get("correction_label_policy", "strict")).lower(),
            "correction_threshold": (
                float(config["correction_threshold"])
                if "correction_threshold" in config
                else None
            ),
            "output_dir": str(output_dir),
        },
        "dataset_summary": summarize_dataset_rows(routing_rows),
        "retrieval_methods": retrieval_summary,
        "routing_methods": routing_summary,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("=== Dataset Summary ===")
    print(json.dumps(payload["dataset_summary"], indent=2, ensure_ascii=False))
    print("=== Retrieval Methods ===")
    print(json.dumps(payload["retrieval_methods"], indent=2, ensure_ascii=False))
    print("=== Routing Methods ===")
    print(json.dumps(payload["routing_methods"], indent=2, ensure_ascii=False))


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def collect_rows(
    samples: list[QASample],
    dense_retriever,
    graph_retriever,
    nlp,
    graph_backend: str,
    retrieval_depth: int,
    top_k_values: list[int],
    label_top_k: int,
    dense_query_llm_calls: float,
    graph_query_llm_calls: float,
    dense_query_token_cost: float,
    graph_query_token_cost: float,
    correction_label_policy: str,
    correction_threshold: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        dense_start = time.perf_counter()
        dense_results = dense_retriever.retrieve(sample.question, top_k=retrieval_depth)
        dense_latency_ms = (time.perf_counter() - dense_start) * 1000.0

        graph_start = time.perf_counter()
        graph_results = graph_retriever.retrieve(sample.question, top_k=retrieval_depth)
        graph_latency_ms = (time.perf_counter() - graph_start) * 1000.0

        fusion_results = reciprocal_rank_fusion(dense_results, graph_results, top_k=retrieval_depth)
        query_features, query_entities = extract_query_features(sample.question, nlp)
        probe_features = extract_probe_features(query_entities, dense_results)

        dense_recalls = {f"recall@{k}": recall_at_k(sample, dense_results[:k]) for k in top_k_values}
        graph_recalls = {f"recall@{k}": recall_at_k(sample, graph_results[:k]) for k in top_k_values}
        fusion_recalls = {f"recall@{k}": recall_at_k(sample, fusion_results[:k]) for k in top_k_values}

        dense_quality = recall_at_k(sample, dense_results[:label_top_k])
        graph_quality = recall_at_k(sample, graph_results[:label_top_k])
        effective_threshold = resolve_correction_threshold(
            policy=correction_label_policy,
            threshold=correction_threshold,
        )
        label, label_reason = compute_correction_label(
            dense_recall=dense_quality,
            graph_recall=graph_quality,
            threshold=effective_threshold,
        )

        rows.append(
            {
                "id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "gold_answers": list(sample.answer_aliases),
                "dataset_name": sample.dataset_name,
                "workload": sample.workload,
                "question_type": sample.question_type,
                "graph_backend": graph_backend,
                "gold_titles": list(sample.gold_titles),
                "gold_passage_ids": list(sample.gold_passage_ids),
                "label": label,
                "label_reason": label_reason,
                "dense_quality": dense_quality,
                "graph_quality": graph_quality,
                "correction_label_policy": correction_label_policy,
                "correction_threshold": effective_threshold,
                "dense_latency_ms": dense_latency_ms,
                "graph_latency_ms": graph_latency_ms,
                "dense_query_llm_calls": dense_query_llm_calls,
                "graph_query_llm_calls": graph_query_llm_calls,
                "dense_query_token_cost": dense_query_token_cost,
                "graph_query_token_cost": graph_query_token_cost,
                "dense_ids": [passage.id for passage in dense_results],
                "graph_ids": [passage.id for passage in graph_results],
                "fusion_ids": [passage.id for passage in fusion_results],
                "dense_scores": [float(passage.score) for passage in dense_results],
                "graph_scores": [float(passage.score) for passage in graph_results],
                "dense_passages": serialize_passages(dense_results),
                "graph_passages": serialize_passages(graph_results),
                "fusion_passages": serialize_passages(fusion_results),
                **prefix_keys("dense_", dense_recalls),
                **prefix_keys("graph_", graph_recalls),
                **prefix_keys("fusion_", fusion_recalls),
                **query_features,
                **probe_features,
            }
        )
    return rows


def compute_correction_label(
    dense_recall: float,
    graph_recall: float,
    threshold: float,
) -> tuple[int | None, str]:
    if dense_recall >= threshold:
        return 0, "dense_sufficient"
    if graph_recall > dense_recall:
        return 1, "graph_correction_needed"
    return None, "discard_dense_insufficient_graph_not_better"


def resolve_correction_threshold(policy: str, threshold: float | None) -> float:
    if policy == "strict":
        return 1.0
    if policy == "moderate":
        return 0.5
    if policy == "lenient":
        return np.nextafter(0.0, 1.0)
    if policy == "threshold":
        if threshold is None:
            raise ValueError("correction_threshold is required when correction_label_policy=threshold")
        return float(threshold)
    raise ValueError(
        "correction_label_policy must be one of: strict, moderate, lenient, threshold"
    )


def serialize_passages(passages: list[RetrievedPassage]) -> list[dict[str, Any]]:
    return [
        {
            "id": passage.id,
            "title": passage.title,
            "source_doc_id": passage.source_doc_id,
            "score": float(passage.score),
            "text": passage.text,
        }
        for passage in passages
    ]


def retrieval_row_from_full_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "gold_answers": row.get("gold_answers", [row["answer"]]),
        "dataset_name": row["dataset_name"],
        "workload": row["workload"],
        "question_type": row["question_type"],
        "graph_backend": row["graph_backend"],
        "gold_titles": row["gold_titles"],
        "gold_passage_ids": row["gold_passage_ids"],
        "retrieval": {
            "dense": row["dense_passages"],
            "graph": row["graph_passages"],
            "fusion": row["fusion_passages"],
        },
    }


def routing_row_from_full_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("dense_passages", None)
    payload.pop("graph_passages", None)
    payload.pop("fusion_passages", None)
    return payload


def prefix_keys(prefix: str, payload: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in payload.items()}


def reciprocal_rank_fusion(
    dense_results: list[RetrievedPassage],
    graph_results: list[RetrievedPassage],
    top_k: int,
    k: int = 60,
) -> list[RetrievedPassage]:
    fused_scores: dict[str, float] = {}
    passage_lookup: dict[str, RetrievedPassage] = {}

    for ranked_results in (dense_results, graph_results):
        for rank, passage in enumerate(ranked_results, start=1):
            fused_scores[passage.id] = fused_scores.get(passage.id, 0.0) + 1.0 / (k + rank)
            passage_lookup[passage.id] = passage

    ranked_ids = sorted(fused_scores, key=lambda passage_id: (-fused_scores[passage_id], passage_id))[:top_k]
    return [
        RetrievedPassage(
            id=passage_lookup[passage_id].id,
            text=passage_lookup[passage_id].text,
            title=passage_lookup[passage_id].title,
            source_doc_id=passage_lookup[passage_id].source_doc_id,
            score=float(fused_scores[passage_id]),
        )
        for passage_id in ranked_ids
    ]


def recall_at_k(sample: QASample, retrieved: list[RetrievedPassage]) -> float:
    if sample.gold_passage_ids:
        gold_targets = set(sample.gold_passage_ids)
        retrieved_targets = {passage.id for passage in retrieved}
    elif sample.gold_titles:
        gold_targets = set(sample.gold_titles)
        retrieved_targets = {passage.title for passage in retrieved if passage.title}
    else:
        if not sample.answer.strip():
            return 0.0
        answer_norm = normalize_text(sample.answer)
        return 1.0 if any(answer_norm in normalize_text(passage.text) for passage in retrieved) else 0.0

    if not gold_targets:
        return 0.0
    return len(gold_targets & retrieved_targets) / len(gold_targets)


def summarize_retrieval_methods(rows: list[dict[str, Any]], top_k_values: list[int]) -> dict[str, Any]:
    return {
        "dense_only": summarize_fixed_method(rows, prefix="dense", top_k_values=top_k_values, graph_invocation_rate=0.0),
        "graph_only": summarize_fixed_method(rows, prefix="graph", top_k_values=top_k_values, graph_invocation_rate=1.0),
        "dense_graph_rrf": summarize_fusion_method(rows, top_k_values=top_k_values),
    }


def summarize_fixed_method(
    rows: list[dict[str, Any]],
    prefix: str,
    top_k_values: list[int],
    graph_invocation_rate: float,
) -> dict[str, Any]:
    latencies = [float(row[f"{prefix}_latency_ms"]) for row in rows]
    llm_calls = [float(row[f"{prefix}_query_llm_calls"]) for row in rows]
    token_costs = [float(row[f"{prefix}_query_token_cost"]) for row in rows]
    payload = {
        "num_queries": len(rows),
        "graph_invocation_rate": graph_invocation_rate,
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95.0),
        "avg_llm_calls_per_query": safe_mean(llm_calls),
        "avg_token_cost_per_query": safe_mean(token_costs),
    }
    for k in top_k_values:
        payload[f"recall@{k}"] = safe_mean([float(row[f"{prefix}_recall@{k}"]) for row in rows])
    return payload


def summarize_fusion_method(rows: list[dict[str, Any]], top_k_values: list[int]) -> dict[str, Any]:
    latencies = [float(row["dense_latency_ms"]) + float(row["graph_latency_ms"]) for row in rows]
    llm_calls = [float(row["dense_query_llm_calls"]) + float(row["graph_query_llm_calls"]) for row in rows]
    token_costs = [float(row["dense_query_token_cost"]) + float(row["graph_query_token_cost"]) for row in rows]
    payload = {
        "num_queries": len(rows),
        "graph_invocation_rate": 1.0,
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95.0),
        "avg_llm_calls_per_query": safe_mean(llm_calls),
        "avg_token_cost_per_query": safe_mean(token_costs),
    }
    for k in top_k_values:
        payload[f"recall@{k}"] = safe_mean([float(row[f"fusion_recall@{k}"]) for row in rows])
    return payload


def summarize_routing_methods(
    rows: list[dict[str, Any]],
    top_k_values: list[int],
    train_ratio: float,
    random_seed: int,
    evaluation_mode: str,
    num_folds: int,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["label"] is not None]
    summary: dict[str, Any] = {
        "label_stats": {
            "total_samples": len(rows),
            "valid_samples": len(valid_rows),
            "tie_samples": len(rows) - len(valid_rows),
            "label_0_count": sum(1 for row in valid_rows if int(row["label"]) == 0),
            "label_1_count": sum(1 for row in valid_rows if int(row["label"]) == 1),
        }
    }

    if len(valid_rows) < 4:
        summary["error"] = "Not enough valid labeled samples after tie filtering."
        return summary

    labels = [int(row["label"]) for row in valid_rows]
    if len(set(labels)) < 2:
        summary["error"] = "Need both label classes after tie filtering."
        return summary
    class_counts = {label: labels.count(label) for label in sorted(set(labels))}

    query_names = query_feature_names()
    probe_names = probe_feature_names()

    try:
        if evaluation_mode == "cv":
            if min(class_counts.values()) < num_folds:
                summary["error"] = (
                    f"Not enough samples in the minority class for {num_folds}-fold CV."
                )
                summary["class_counts"] = class_counts
                return summary
            summary.update(
                evaluate_router_cv(
                    rows=valid_rows,
                    top_k_values=top_k_values,
                    query_names=query_names,
                    probe_names=probe_names,
                    num_folds=num_folds,
                    random_seed=random_seed,
                )
            )
        else:
            summary.update(
                evaluate_router_split(
                    rows=valid_rows,
                    top_k_values=top_k_values,
                    query_names=query_names,
                    probe_names=probe_names,
                    train_ratio=train_ratio,
                    random_seed=random_seed,
                )
            )
    except ValueError as exc:
        summary["error"] = str(exc)
        summary["class_counts"] = class_counts
    return summary


def evaluate_router_split(
    rows: list[dict[str, Any]],
    top_k_values: list[int],
    query_names: list[str],
    probe_names: list[str],
    train_ratio: float,
    random_seed: int,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    sample_ids = [str(row["id"]) for row in rows]
    split = make_split(sample_ids=sample_ids, labels=labels, train_ratio=train_ratio, random_seed=random_seed)
    train_ids = set(split.train_ids)
    test_ids = set(split.test_ids)
    train_rows = [row for row in rows if row["id"] in train_ids]
    test_rows = [row for row in rows if row["id"] in test_ids]
    methods = train_router_models(
        train_rows=train_rows,
        test_rows=test_rows,
        top_k_values=top_k_values,
        query_names=query_names,
        probe_names=probe_names,
        random_seed=random_seed,
    )
    return {
        "split_mode": split.split_mode,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "warning": (
            "Train/test split fell back to resubstitution because the minority class had fewer than 2 samples."
            if split.split_mode == "resubstitution"
            else None
        ),
        "methods": methods,
    }


def evaluate_router_cv(
    rows: list[dict[str, Any]],
    top_k_values: list[int],
    query_names: list[str],
    probe_names: list[str],
    num_folds: int,
    random_seed: int,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    sample_ids = [str(row["id"]) for row in rows]
    folds = make_cv_folds(sample_ids=sample_ids, labels=labels, num_folds=num_folds, random_seed=random_seed)

    method_to_fold_metrics: dict[str, list[dict[str, Any]]] = {
        "query_only": [],
        "probe_only": [],
        "query_plus_probe": [],
    }

    for fold in folds:
        train_ids = set(fold.train_ids)
        test_ids = set(fold.test_ids)
        train_rows = [row for row in rows if row["id"] in train_ids]
        test_rows = [row for row in rows if row["id"] in test_ids]
        fold_methods = train_router_models(
            train_rows=train_rows,
            test_rows=test_rows,
            top_k_values=top_k_values,
            query_names=query_names,
            probe_names=probe_names,
            random_seed=random_seed,
        )
        for method_name, metrics in fold_methods.items():
            method_to_fold_metrics[method_name].append(metrics)

    return {
        "split_mode": "cross_validation",
        "num_folds": num_folds,
        "methods": {
            method_name: summarize_router_folds(fold_metrics, top_k_values)
            for method_name, fold_metrics in method_to_fold_metrics.items()
        },
    }


def train_router_models(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    top_k_values: list[int],
    query_names: list[str],
    probe_names: list[str],
    random_seed: int,
) -> dict[str, Any]:
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


def summarize_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, int] = {}
    by_workload: dict[str, int] = {}
    by_question_type: dict[str, int] = {}
    for row in rows:
        by_dataset[row["dataset_name"]] = by_dataset.get(row["dataset_name"], 0) + 1
        by_workload[row["workload"]] = by_workload.get(row["workload"], 0) + 1
        by_question_type[row["question_type"]] = by_question_type.get(row["question_type"], 0) + 1
    return {
        "num_queries": len(rows),
        "datasets": by_dataset,
        "workloads": by_workload,
        "question_types": by_question_type,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
