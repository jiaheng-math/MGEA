from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_generations import extract_gold_answers, extract_prediction, score_prediction_against_gold_answers, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate routed QA EM/F1 by combining dense and graph generations according to "
            "out-of-fold routing probabilities."
        )
    )
    parser.add_argument("--generations", required=True, help="Path to generations.jsonl.")
    parser.add_argument(
        "--oof-predictions",
        required=True,
        help="Path to oof_predictions.jsonl from plot_routing_pareto.py.",
    )
    parser.add_argument("--output", required=True, help="Path to write routed QA metrics JSON.")
    parser.add_argument(
        "--per-sample-output",
        default=None,
        help="Optional JSONL path for routed per-sample decisions and scores.",
    )
    parser.add_argument(
        "--methods",
        default="query_only,probe_only,query_plus_probe",
        help="Comma-separated router methods to evaluate.",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated routing thresholds.",
    )
    parser.add_argument("--dense-method", default="dense", help="Generation method used when not invoking graph.")
    parser.add_argument("--graph-method", default="graph", help="Generation method used when invoking graph.")
    parser.add_argument("--tie-policy", choices=["dense", "graph"], default="dense")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_rows = {str(row["id"]): row for row in load_jsonl(Path(args.generations))}
    prediction_rows = {str(row["id"]): row for row in load_jsonl(Path(args.oof_predictions))}
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    thresholds = sorted({float(value.strip()) for value in args.threshold_grid.split(",") if value.strip()})

    sample_ids = list(generation_rows)
    if not sample_ids:
        raise ValueError("No generation rows available for routed QA evaluation.")

    metrics: dict[str, Any] = {
        "generations": args.generations,
        "oof_predictions": args.oof_predictions,
        "num_samples": len(sample_ids),
        "dense_method": args.dense_method,
        "graph_method": args.graph_method,
        "tie_policy": args.tie_policy,
        "methods": {},
    }
    per_sample_rows: list[dict[str, Any]] = []

    for method in methods:
        probability_key = f"{method}_probability"
        method_points = []
        for threshold in thresholds:
            records = [
                evaluate_sample(
                    sample_id=sample_id,
                    generation_row=generation_rows[sample_id],
                    prediction_row=prediction_rows.get(sample_id),
                    probability_key=probability_key,
                    threshold=threshold,
                    dense_method=args.dense_method,
                    graph_method=args.graph_method,
                    tie_policy=args.tie_policy,
                )
                for sample_id in sample_ids
            ]
            method_points.append(summarize_threshold_records(threshold, records))
            for record in records:
                record["router_method"] = method
                per_sample_rows.append(record)
        metrics["methods"][method] = method_points

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


def evaluate_sample(
    *,
    sample_id: str,
    generation_row: dict[str, Any],
    prediction_row: dict[str, Any] | None,
    probability_key: str,
    threshold: float,
    dense_method: str,
    graph_method: str,
    tie_policy: str,
) -> dict[str, Any]:
    if prediction_row is None or probability_key not in prediction_row:
        probability = None
        choose_graph = tie_policy == "graph"
        missing_probability = True
    else:
        probability = float(prediction_row[probability_key])
        choose_graph = probability >= threshold
        missing_probability = False
    selected_method = graph_method if choose_graph else dense_method
    method_payload = generation_row.get("methods", {}).get(selected_method, {})
    prediction = extract_prediction(method_payload.get("answer", ""))
    gold_answers = extract_gold_answers(generation_row)
    gold_answer = gold_answers[0] if gold_answers else ""
    scores = score_prediction_against_gold_answers(prediction=prediction, gold_answers=gold_answers)

    return {
        "id": sample_id,
        "threshold": threshold,
        "probability": probability,
        "missing_probability": missing_probability,
        "choose_graph": choose_graph,
        "selected_generation_method": selected_method,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "gold_answers": gold_answers,
        "matched_gold_answer": scores["matched_gold_answer"],
        "exact_match": scores["exact_match"],
        "f1": scores["f1"],
        "precision": scores["precision"],
        "recall": scores["recall"],
        "has_error": bool(method_payload.get("error")),
    }


def summarize_threshold_records(threshold: float, records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "threshold": float(threshold),
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
