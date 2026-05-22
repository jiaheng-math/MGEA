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

from evaluate_generations import (
    extract_gold_answers,
    extract_prediction,
    score_prediction_against_gold_answers,
    write_jsonl,
)


DEFAULT_ROUTER_METHODS = "query_only:query_only_router,probe_only:probe_only_router,query_plus_probe:ours_combined_router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer EM/F1 for routers at an exactly matched graph invocation budget. "
            "This reuses dense/graph generations and does not call an LLM."
        )
    )
    parser.add_argument("--generations", required=True, help="main_table_generations_top5_*.jsonl.")
    parser.add_argument("--oof-predictions", required=True, help="OOF router probabilities JSONL.")
    parser.add_argument("--output", required=True, help="Output metrics JSON.")
    parser.add_argument("--per-sample-output", default=None, help="Optional per-sample JSONL.")
    parser.add_argument(
        "--router-methods",
        default=DEFAULT_ROUTER_METHODS,
        help="Comma-separated source:output specs, e.g. query_only:query_only_router.",
    )
    parser.add_argument(
        "--reference-method",
        default="query_plus_probe",
        help="Source router method that defines the graph budget when --target-graph-rate/count are absent.",
    )
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to --reference-method to compute the default graph budget.",
    )
    parser.add_argument(
        "--budget-metrics",
        default=None,
        help=(
            "Optional routing metrics JSON. If supplied, the target graph count is read from "
            "routing_methods.methods[--budget-method].graph_invocation_rate[_mean] times valid_samples."
        ),
    )
    parser.add_argument(
        "--budget-method",
        default="query_plus_probe",
        help="Routing method whose stored graph invocation count defines the matched budget.",
    )
    parser.add_argument(
        "--target-graph-rate",
        type=float,
        default=None,
        help="Optional graph invocation rate to match exactly after rounding to a count.",
    )
    parser.add_argument(
        "--target-graph-count",
        type=int,
        default=None,
        help="Optional exact number of graph-routed queries. Overrides --target-graph-rate.",
    )
    parser.add_argument("--tie-policy", choices=["dense", "graph"], default="dense")
    parser.add_argument("--dense-method", default="dense_only")
    parser.add_argument("--graph-method", default="graph_only")
    parser.add_argument("--fusion-method", default="dense_graph_rrf")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_rows = {str(row["id"]): row for row in load_jsonl(Path(args.generations))}
    oof_rows = {str(row["id"]): row for row in load_jsonl(Path(args.oof_predictions))}
    sample_ids = list(generation_rows)
    if not sample_ids:
        raise ValueError("No generation rows found.")

    router_specs = parse_router_specs(args.router_methods)
    target_count, target_source = resolve_target_graph_count(
        sample_ids=sample_ids,
        oof_rows=oof_rows,
        reference_method=args.reference_method,
        reference_threshold=args.reference_threshold,
        budget_metrics=Path(args.budget_metrics) if args.budget_metrics else None,
        budget_method=args.budget_method,
        tie_policy=args.tie_policy,
        target_graph_rate=args.target_graph_rate,
        target_graph_count=args.target_graph_count,
    )

    per_sample_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "generations": args.generations,
        "oof_predictions": args.oof_predictions,
        "num_samples": len(sample_ids),
        "matching_strategy": "exact_top_probability_budget",
        "target_graph_count": target_count,
        "target_graph_invocation_rate": target_count / len(sample_ids),
        "target_source": target_source,
        "tie_policy": args.tie_policy,
        "dense_method": args.dense_method,
        "graph_method": args.graph_method,
        "fusion_method": args.fusion_method,
        "random_trials": args.random_trials,
        "random_seed": args.random_seed,
        "methods": {},
    }

    fixed_methods = {
        "dense_only": (args.dense_method, False),
        "graph_only": (args.graph_method, True),
        "dense_graph_rrf": (args.fusion_method, True),
    }
    for output_name, (generation_method, choose_graph) in fixed_methods.items():
        records = [
            score_selected_generation(
                sample_id=sample_id,
                generation_row=generation_rows[sample_id],
                selected_generation_method=generation_method,
                choose_graph=choose_graph,
                decision_source=output_name,
            )
            for sample_id in sample_ids
        ]
        metrics["methods"][output_name] = summarize_records(records)
        per_sample_rows.extend(records)

    for spec in router_specs:
        decisions = make_matched_budget_decisions(
            sample_ids=sample_ids,
            oof_rows=oof_rows,
            source_method=spec["source"],
            target_count=target_count,
            tie_policy=args.tie_policy,
        )
        records = [
            score_selected_generation(
                sample_id=sample_id,
                generation_row=generation_rows[sample_id],
                selected_generation_method=args.graph_method if decision["choose_graph"] else args.dense_method,
                choose_graph=bool(decision["choose_graph"]),
                decision_source=spec["output"],
                probability=decision["probability"],
                missing_probability=bool(decision["missing_probability"]),
            )
            for sample_id, decision in zip(sample_ids, decisions)
        ]
        summary = summarize_records(records)
        summary.update(summarize_decision_threshold(decisions))
        summary["source_method"] = spec["source"]
        metrics["methods"][spec["output"]] = summary
        per_sample_rows.extend(records)

    random_summary, random_rows = evaluate_random_router(
        sample_ids=sample_ids,
        generation_rows=generation_rows,
        target_count=target_count,
        dense_method=args.dense_method,
        graph_method=args.graph_method,
        trials=args.random_trials,
        seed=args.random_seed,
    )
    metrics["methods"]["random_router"] = random_summary
    per_sample_rows.extend(random_rows)

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


def parse_router_specs(value: str) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            source, output = item.split(":", 1)
        else:
            source = item
            output = f"{item}_router"
        specs.append({"source": source.strip(), "output": output.strip()})
    if not specs:
        raise ValueError("At least one router method is required.")
    return specs


def resolve_target_graph_count(
    *,
    sample_ids: list[str],
    oof_rows: dict[str, dict[str, Any]],
    reference_method: str,
    reference_threshold: float,
    budget_metrics: Path | None,
    budget_method: str,
    tie_policy: str,
    target_graph_rate: float | None,
    target_graph_count: int | None,
) -> tuple[int, dict[str, Any]]:
    if target_graph_count is not None:
        count = int(target_graph_count)
        source = {"type": "explicit_count", "target_graph_count": count}
    elif target_graph_rate is not None:
        count = round(len(sample_ids) * float(target_graph_rate))
        source = {"type": "explicit_rate", "target_graph_rate": float(target_graph_rate)}
    elif budget_metrics is not None:
        count, source = graph_count_from_metrics(budget_metrics, budget_method)
    else:
        decisions = [
            threshold_decision(
                sample_id=sample_id,
                oof_row=oof_rows.get(sample_id),
                source_method=reference_method,
                threshold=reference_threshold,
                tie_policy=tie_policy,
            )
            for sample_id in sample_ids
        ]
        count = sum(1 for decision in decisions if decision["choose_graph"])
        source = {
            "type": "reference_threshold",
            "reference_method": reference_method,
            "reference_threshold": reference_threshold,
            "reference_graph_count": count,
            "reference_missing_probability_count": sum(
                1 for decision in decisions if decision["missing_probability"]
            ),
        }
    if not 0 <= count <= len(sample_ids):
        raise ValueError(f"Target graph count must be in [0, {len(sample_ids)}]. Got {count}.")
    return count, source


def graph_count_from_metrics(path: Path, budget_method: str) -> tuple[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    routing_methods = payload.get("routing_methods", {})
    label_stats = routing_methods.get("label_stats", {})
    valid_samples = int(label_stats.get("valid_samples", 0))
    total_samples = int(label_stats.get("total_samples", valid_samples))
    method_payload = routing_methods.get("methods", {}).get(budget_method)
    if method_payload is None:
        raise ValueError(f"Budget method {budget_method!r} not found in {path}.")
    if "graph_invocation_rate_mean" in method_payload:
        graph_rate = float(method_payload["graph_invocation_rate_mean"])
        graph_rate_key = "graph_invocation_rate_mean"
    elif "graph_invocation_rate" in method_payload:
        graph_rate = float(method_payload["graph_invocation_rate"])
        graph_rate_key = "graph_invocation_rate"
    else:
        raise ValueError(f"No graph invocation rate found for {budget_method!r} in {path}.")
    if valid_samples <= 0:
        raise ValueError(f"Cannot infer graph count from {path}: valid_samples={valid_samples}.")
    count = round(graph_rate * valid_samples)
    return count, {
        "type": "stored_routing_metrics",
        "budget_metrics": str(path),
        "budget_method": budget_method,
        "graph_rate_key": graph_rate_key,
        "stored_graph_invocation_rate": graph_rate,
        "valid_samples": valid_samples,
        "total_samples": total_samples,
        "target_graph_count": count,
        "all_query_graph_invocation_rate": count / total_samples if total_samples else None,
    }


def threshold_decision(
    *,
    sample_id: str,
    oof_row: dict[str, Any] | None,
    source_method: str,
    threshold: float,
    tie_policy: str,
) -> dict[str, Any]:
    probability_key = f"{source_method}_probability"
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


def make_matched_budget_decisions(
    *,
    sample_ids: list[str],
    oof_rows: dict[str, dict[str, Any]],
    source_method: str,
    target_count: int,
    tie_policy: str,
) -> list[dict[str, Any]]:
    probability_key = f"{source_method}_probability"
    scored: list[tuple[float, str, float | None, bool]] = []
    for sample_id in sample_ids:
        oof_row = oof_rows.get(sample_id)
        if oof_row is None or probability_key not in oof_row:
            probability = None
            missing_probability = True
            score = float("inf") if tie_policy == "graph" else float("-inf")
        else:
            probability = float(oof_row[probability_key])
            missing_probability = False
            score = probability
        scored.append((score, sample_id, probability, missing_probability))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected_ids = {sample_id for _, sample_id, _, _ in ranked[:target_count]}
    lookup = {
        sample_id: {
            "id": sample_id,
            "choose_graph": sample_id in selected_ids,
            "probability": probability,
            "missing_probability": missing_probability,
            "source_method": source_method,
        }
        for _, sample_id, probability, missing_probability in scored
    }
    return [lookup[sample_id] for sample_id in sample_ids]


def summarize_decision_threshold(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_probabilities = [
        decision["probability"]
        for decision in decisions
        if decision["choose_graph"] and decision["probability"] is not None
    ]
    rejected_probabilities = [
        decision["probability"]
        for decision in decisions
        if not decision["choose_graph"] and decision["probability"] is not None
    ]
    return {
        "selected_min_probability": min(chosen_probabilities) if chosen_probabilities else None,
        "rejected_max_probability": max(rejected_probabilities) if rejected_probabilities else None,
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
    target_count: int,
    dense_method: str,
    graph_method: str,
    trials: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    trial_summaries = []
    per_sample_rows = []
    for trial in range(trials):
        selected_ids = set(rng.sample(sample_ids, target_count))
        records = []
        for sample_id in sample_ids:
            choose_graph = sample_id in selected_ids
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

    metric_names = ["graph_invocation_rate", "exact_match", "f1", "precision", "recall", "error_rate"]
    summary: dict[str, Any] = {"num_trials": trials, "target_graph_count": target_count}
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


if __name__ == "__main__":
    main()
