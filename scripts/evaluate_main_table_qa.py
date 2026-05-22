from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_generations import extract_gold_answers, extract_prediction, score_prediction_against_gold_answers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute main-table answer EM/F1 for fixed retrieval methods, intelligent routers, "
            "and a matched-budget random router."
        )
    )
    parser.add_argument("--generations", required=True, help="generations_top5.jsonl with dense/graph/fusion answers.")
    parser.add_argument("--routing-rows", required=True, help="routing_rows.jsonl for all queries.")
    parser.add_argument(
        "--oof-predictions",
        required=True,
        help="oof_predictions.jsonl from plot_routing_pareto.py. Missing ids, e.g. tie samples, use --tie-policy.",
    )
    parser.add_argument("--output", required=True, help="Output metrics JSON.")
    parser.add_argument(
        "--per-sample-output",
        default=None,
        help="Optional JSONL with per-sample selected path and EM/F1.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Router probability threshold.")
    parser.add_argument(
        "--router-methods",
        default="query_only,query_plus_probe",
        help="Comma-separated router methods to include.",
    )
    parser.add_argument(
        "--random-reference-method",
        default="query_plus_probe",
        help="Router method whose all-query graph invocation rate is used by random router.",
    )
    parser.add_argument("--random-trials", type=int, default=100, help="Random-router trials.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tie-policy", choices=["dense", "graph"], default="dense")
    parser.add_argument("--dense-method", default="dense")
    parser.add_argument("--graph-method", default="graph")
    parser.add_argument("--fusion-method", default="fusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_rows = {str(row["id"]): row for row in load_jsonl(Path(args.generations))}
    routing_rows = {str(row["id"]): row for row in load_jsonl(Path(args.routing_rows))}
    oof_rows = {str(row["id"]): row for row in load_jsonl(Path(args.oof_predictions))}
    sample_ids = [sample_id for sample_id in generation_rows if sample_id in routing_rows]
    if not sample_ids:
        raise ValueError("No overlapping ids between generations and routing rows.")

    router_methods = [method.strip() for method in args.router_methods.split(",") if method.strip()]
    per_sample_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "generations": args.generations,
        "routing_rows": args.routing_rows,
        "oof_predictions": args.oof_predictions,
        "num_samples": len(sample_ids),
        "threshold": args.threshold,
        "tie_policy": args.tie_policy,
        "random_trials": args.random_trials,
        "random_seed": args.random_seed,
        "methods": {},
    }

    fixed_methods = {
        "dense_only": args.dense_method,
        "graph_only": args.graph_method,
        "dense_graph_rrf": args.fusion_method,
    }
    for output_name, generation_method in fixed_methods.items():
        records = [
            score_selected_generation(
                sample_id=sample_id,
                generation_row=generation_rows[sample_id],
                selected_generation_method=generation_method,
                choose_graph=(generation_method == args.graph_method),
                decision_source=output_name,
            )
            for sample_id in sample_ids
        ]
        metrics["methods"][output_name] = summarize_records(records)
        per_sample_rows.extend(records)

    router_decisions: dict[str, list[dict[str, Any]]] = {}
    for method in router_methods:
        decisions = [
            make_router_decision(
                sample_id=sample_id,
                oof_row=oof_rows.get(sample_id),
                method=method,
                threshold=args.threshold,
                tie_policy=args.tie_policy,
            )
            for sample_id in sample_ids
        ]
        router_decisions[method] = decisions
        records = [
            score_selected_generation(
                sample_id=sample_id,
                generation_row=generation_rows[sample_id],
                selected_generation_method=args.graph_method if decision["choose_graph"] else args.dense_method,
                choose_graph=bool(decision["choose_graph"]),
                decision_source=f"{method}_router",
                probability=decision.get("probability"),
                missing_probability=bool(decision["missing_probability"]),
            )
            for sample_id, decision in zip(sample_ids, decisions)
        ]
        metrics["methods"][f"{method}_router"] = summarize_records(records)
        per_sample_rows.extend(records)

    if args.random_reference_method not in router_decisions:
        raise ValueError(
            f"--random-reference-method={args.random_reference_method} was not included in --router-methods."
        )
    target_rate = mean(float(decision["choose_graph"]) for decision in router_decisions[args.random_reference_method])
    random_summary, random_sample_rows = evaluate_random_router(
        sample_ids=sample_ids,
        generation_rows=generation_rows,
        graph_rate=target_rate,
        dense_method=args.dense_method,
        graph_method=args.graph_method,
        trials=args.random_trials,
        seed=args.random_seed,
    )
    random_summary["target_graph_invocation_rate"] = target_rate
    random_summary["reference_method"] = f"{args.random_reference_method}_router"
    metrics["methods"]["random_router"] = random_summary
    per_sample_rows.extend(random_sample_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_sample_output:
        write_jsonl(Path(args.per_sample_output), per_sample_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_router_decision(
    *,
    sample_id: str,
    oof_row: dict[str, Any] | None,
    method: str,
    threshold: float,
    tie_policy: str,
) -> dict[str, Any]:
    probability_key = f"{method}_probability"
    if oof_row is None or probability_key not in oof_row:
        return {
            "id": sample_id,
            "choose_graph": tie_policy == "graph",
            "probability": None,
            "missing_probability": True,
        }
    probability = float(oof_row[probability_key])
    return {
        "id": sample_id,
        "choose_graph": probability >= threshold,
        "probability": probability,
        "missing_probability": False,
    }


def score_selected_generation(
    *,
    sample_id: str,
    generation_row: dict[str, Any],
    selected_generation_method: str,
    choose_graph: bool,
    decision_source: str,
    probability: float | None = None,
    missing_probability: bool = False,
) -> dict[str, Any]:
    method_payload = generation_row.get("methods", {}).get(selected_generation_method, {})
    prediction = extract_prediction(method_payload.get("answer", ""))
    gold_answers = extract_gold_answers(generation_row)
    scores = score_prediction_against_gold_answers(prediction=prediction, gold_answers=gold_answers)
    return {
        "id": sample_id,
        "decision_source": decision_source,
        "selected_generation_method": selected_generation_method,
        "choose_graph": bool(choose_graph),
        "probability": probability,
        "missing_probability": bool(missing_probability),
        "prediction": prediction,
        "gold_answers": gold_answers,
        "matched_gold_answer": scores["matched_gold_answer"],
        "exact_match": float(scores["exact_match"]),
        "f1": float(scores["f1"]),
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "has_error": bool(method_payload.get("error")),
    }


def evaluate_random_router(
    *,
    sample_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    graph_rate: float,
    dense_method: str,
    graph_method: str,
    trials: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    trial_summaries = []
    per_sample_rows = []
    for trial in range(trials):
        records = []
        for sample_id in sample_ids:
            choose_graph = rng.random() < graph_rate
            selected_method = graph_method if choose_graph else dense_method
            record = score_selected_generation(
                sample_id=sample_id,
                generation_row=generation_rows[sample_id],
                selected_generation_method=selected_method,
                choose_graph=choose_graph,
                decision_source="random_router",
            )
            record["trial"] = trial
            records.append(record)
        trial_summaries.append(summarize_records(records))
        per_sample_rows.extend(records)

    metric_names = [
        "graph_invocation_rate",
        "exact_match",
        "f1",
        "precision",
        "recall",
        "error_rate",
    ]
    summary: dict[str, Any] = {"num_trials": trials}
    for metric_name in metric_names:
        values = [float(item[metric_name]) for item in trial_summaries]
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return summary, per_sample_rows


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(records),
        "graph_invocation_rate": mean(float(record["choose_graph"]) for record in records),
        "missing_probability_rate": mean(float(record.get("missing_probability", False)) for record in records),
        "exact_match": mean(record["exact_match"] for record in records),
        "f1": mean(record["f1"] for record in records),
        "precision": mean(record["precision"] for record in records),
        "recall": mean(record["recall"] for record in records),
        "error_rate": mean(float(record["has_error"]) for record in records),
    }


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
