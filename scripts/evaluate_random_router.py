from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.study_main import percentile, safe_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a random router baseline at the same graph invocation rates as intelligent routers. "
            "For each query, random chooses dense vs graph with Bernoulli(p=target_graph_rate)."
        )
    )
    parser.add_argument("--routing-rows", required=True, help="Path to routing_rows.jsonl.")
    parser.add_argument(
        "--generations",
        default=None,
        help="Optional generations JSONL. If provided, also evaluates answer EM/F1.",
    )
    parser.add_argument(
        "--pareto-operating-points",
        default=None,
        help="pareto_operating_points.json from plot_routing_pareto.py. Uses router graph rates as targets.",
    )
    parser.add_argument(
        "--oof-predictions",
        default=None,
        help="Optional oof_predictions.jsonl. Restricts random evaluation to the same valid samples as routed QA.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--top-k", type=int, default=5, help="Recall@k field to evaluate.")
    parser.add_argument("--trials", type=int, default=100, help="Number of random trials.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--methods",
        default="query_only,probe_only,query_plus_probe",
        help="Comma-separated router methods whose rates are read from Pareto.",
    )
    parser.add_argument(
        "--target-rates",
        default="",
        help="Optional comma-separated graph rates when no Pareto file is supplied.",
    )
    parser.add_argument("--dense-method", default="dense", help="Generation method used when not invoking graph.")
    parser.add_argument("--graph-method", default="graph", help="Generation method used when invoking graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    routing_rows = {str(row["id"]): row for row in load_jsonl(Path(args.routing_rows))}
    generation_rows = (
        {str(row["id"]): row for row in load_jsonl(Path(args.generations))}
        if args.generations
        else None
    )
    qa_helpers = load_qa_helpers() if generation_rows is not None else None
    if args.oof_predictions:
        valid_ids = [str(row["id"]) for row in load_jsonl(Path(args.oof_predictions))]
    else:
        valid_ids = list(routing_rows)
    if generation_rows is not None:
        valid_ids = [sample_id for sample_id in valid_ids if sample_id in generation_rows]
    valid_ids = [sample_id for sample_id in valid_ids if sample_id in routing_rows]
    if not valid_ids:
        raise ValueError("No valid overlapping samples for random router evaluation.")

    targets = load_targets(
        pareto_path=Path(args.pareto_operating_points) if args.pareto_operating_points else None,
        methods=methods,
        target_rates=args.target_rates,
    )

    rng = random.Random(args.random_seed)
    payload: dict[str, Any] = {
        "routing_rows": args.routing_rows,
        "generations": args.generations,
        "pareto_operating_points": args.pareto_operating_points,
        "oof_predictions": args.oof_predictions,
        "num_samples": len(valid_ids),
        "top_k": args.top_k,
        "trials": args.trials,
        "random_seed": args.random_seed,
        "methods": {},
    }

    for method, points in targets.items():
        payload["methods"][method] = [
            evaluate_target_rate(
                sample_ids=valid_ids,
                routing_rows=routing_rows,
                generation_rows=generation_rows,
                qa_helpers=qa_helpers,
                target=target,
                top_k=args.top_k,
                trials=args.trials,
                rng=rng,
                dense_method=args.dense_method,
                graph_method=args.graph_method,
            )
            for target in points
        ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_targets(
    *,
    pareto_path: Path | None,
    methods: list[str],
    target_rates: str,
) -> dict[str, list[dict[str, float]]]:
    if pareto_path is not None:
        payload = json.loads(pareto_path.read_text(encoding="utf-8"))
        operating_points = payload.get("operating_points", {})
        targets: dict[str, list[dict[str, float]]] = {}
        for method in methods:
            points = operating_points.get(method, [])
            targets[method] = [
                {
                    "threshold": float(point.get("threshold", 0.0)),
                    "target_graph_invocation_rate": float(point["graph_invocation_rate"]),
                }
                for point in points
            ]
        return targets

    rates = [float(value.strip()) for value in target_rates.split(",") if value.strip()]
    if not rates:
        raise ValueError("Provide --pareto-operating-points or --target-rates.")
    return {
        "manual": [
            {
                "threshold": rate,
                "target_graph_invocation_rate": rate,
            }
            for rate in rates
        ]
    }


def evaluate_target_rate(
    *,
    sample_ids: list[str],
    routing_rows: dict[str, dict[str, Any]],
    generation_rows: dict[str, dict[str, Any]] | None,
    qa_helpers: dict[str, Any] | None,
    target: dict[str, float],
    top_k: int,
    trials: int,
    rng: random.Random,
    dense_method: str,
    graph_method: str,
) -> dict[str, Any]:
    target_rate = float(target["target_graph_invocation_rate"])
    trial_records = [
        evaluate_trial(
            sample_ids=sample_ids,
            routing_rows=routing_rows,
            generation_rows=generation_rows,
            qa_helpers=qa_helpers,
            graph_rate=target_rate,
            top_k=top_k,
            rng=rng,
            dense_method=dense_method,
            graph_method=graph_method,
        )
        for _ in range(trials)
    ]
    metric_names = [
        "graph_invocation_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        f"recall@{top_k}",
        "exact_match",
        "f1",
        "precision",
        "recall",
        "error_rate",
    ]
    summary: dict[str, Any] = {
        "threshold": float(target["threshold"]),
        "target_graph_invocation_rate": target_rate,
    }
    for metric_name in metric_names:
        values = [record[metric_name] for record in trial_records if record.get(metric_name) is not None]
        if not values:
            continue
        summary[f"{metric_name}_mean"] = safe_mean(values)
        summary[f"{metric_name}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return summary


def evaluate_trial(
    *,
    sample_ids: list[str],
    routing_rows: dict[str, dict[str, Any]],
    generation_rows: dict[str, dict[str, Any]] | None,
    qa_helpers: dict[str, Any] | None,
    graph_rate: float,
    top_k: int,
    rng: random.Random,
    dense_method: str,
    graph_method: str,
) -> dict[str, float | None]:
    chooses_graph = [rng.random() < graph_rate for _ in sample_ids]
    latencies = []
    recalls = []
    qa_records = []

    for sample_id, choose_graph in zip(sample_ids, chooses_graph):
        routing_row = routing_rows[sample_id]
        latencies.append(float(routing_row["graph_latency_ms"] if choose_graph else routing_row["dense_latency_ms"]))
        recalls.append(float(routing_row[f"graph_recall@{top_k}" if choose_graph else f"dense_recall@{top_k}"]))

        if generation_rows is not None and qa_helpers is not None:
            extract_gold_answers = qa_helpers["extract_gold_answers"]
            extract_prediction = qa_helpers["extract_prediction"]
            score_prediction_against_gold_answers = qa_helpers["score_prediction_against_gold_answers"]
            generation_row = generation_rows[sample_id]
            selected_method = graph_method if choose_graph else dense_method
            method_payload = generation_row.get("methods", {}).get(selected_method, {})
            prediction = extract_prediction(method_payload.get("answer", ""))
            scores = score_prediction_against_gold_answers(
                prediction=prediction,
                gold_answers=extract_gold_answers(generation_row),
            )
            qa_records.append(
                {
                    "exact_match": float(scores["exact_match"]),
                    "f1": float(scores["f1"]),
                    "precision": float(scores["precision"]),
                    "recall": float(scores["recall"]),
                    "has_error": float(bool(method_payload.get("error"))),
                }
            )

    payload: dict[str, float | None] = {
        "graph_invocation_rate": safe_mean(float(value) for value in chooses_graph),
        "avg_latency_ms": safe_mean(latencies),
        "p95_latency_ms": percentile(latencies, 95.0),
        f"recall@{top_k}": safe_mean(recalls),
        "exact_match": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "error_rate": None,
    }
    if qa_records:
        payload.update(
            {
                "exact_match": safe_mean(record["exact_match"] for record in qa_records),
                "f1": safe_mean(record["f1"] for record in qa_records),
                "precision": safe_mean(record["precision"] for record in qa_records),
                "recall": safe_mean(record["recall"] for record in qa_records),
                "error_rate": safe_mean(record["has_error"] for record in qa_records),
            }
        )
    return payload


def load_qa_helpers() -> dict[str, Any]:
    try:
        from evaluate_generations import (
            extract_gold_answers,
            extract_prediction,
            score_prediction_against_gold_answers,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QA generation helpers are required only when --generations is provided. "
            "Run from the project root or set PYTHONPATH=scripts:."
        ) from exc
    return {
        "extract_gold_answers": extract_gold_answers,
        "extract_prediction": extract_prediction,
        "score_prediction_against_gold_answers": score_prediction_against_gold_answers,
    }


if __name__ == "__main__":
    main()
