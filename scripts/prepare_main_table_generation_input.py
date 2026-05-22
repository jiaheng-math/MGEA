from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_ROUTER_METHODS = (
    "query_only:query_only_router,"
    "probe_only:probe_only_router,"
    "query_plus_probe:ours_combined_router"
)
DEFAULT_OUTPUT_ORDER = [
    "dense_only",
    "graph_only",
    "dense_graph_rrf",
    "query_only_router",
    "probe_only_router",
    "ours_combined_router",
    "random_router",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one generation input JSONL covering every main-table QA method. "
            "Each output row keeps the original query and stores top-k passages under "
            "dense_only, graph_only, dense_graph_rrf, query_only_router, "
            "probe_only_router, ours_combined_router, and random_router."
        )
    )
    parser.add_argument("--retrieval-results", required=True, help="Path to retrieval_results.jsonl.")
    parser.add_argument(
        "--oof-predictions",
        required=True,
        help="Path to oof_predictions.jsonl from plot_routing_pareto.py. Tie rows may be absent.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL for batch_generate_from_retrieval.py.")
    parser.add_argument("--top-k", type=int, default=5, help="Passages per method.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Router probability threshold.")
    parser.add_argument(
        "--tie-policy",
        choices=["dense", "graph"],
        default="dense",
        help="Path to use when a query has no OOF probability, e.g. tie/discarded routing rows.",
    )
    parser.add_argument(
        "--router-methods",
        default=DEFAULT_ROUTER_METHODS,
        help=(
            "Comma-separated source:output router specs. Source is the OOF prefix "
            "before '_probability'; output is the generation method name."
        ),
    )
    parser.add_argument(
        "--random-reference-method",
        default="query_plus_probe",
        help="Router source method whose all-query graph rate is matched by random_router.",
    )
    parser.add_argument(
        "--random-graph-rate",
        type=float,
        default=None,
        help="Optional explicit graph probability for random_router. Defaults to the reference router rate.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for random_router decisions.")
    parser.add_argument("--dense-source", default="dense", help="Retrieval key for dense passages.")
    parser.add_argument("--graph-source", default="graph", help="Retrieval key for graph passages.")
    parser.add_argument("--fusion-source", default="fusion", help="Retrieval key for RRF passages.")
    parser.add_argument(
        "--methods-output",
        default=",".join(DEFAULT_OUTPUT_ORDER),
        help="Optional comma-separated method order metadata for downstream --methods auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_rows = load_jsonl(Path(args.retrieval_results))
    oof_rows = {str(row["id"]): row for row in load_jsonl(Path(args.oof_predictions))}
    router_specs = parse_router_specs(args.router_methods)
    methods_output = [method.strip() for method in args.methods_output.split(",") if method.strip()]

    if args.random_graph_rate is None:
        reference_rate = compute_reference_graph_rate(
            rows=retrieval_rows,
            oof_rows=oof_rows,
            source_method=args.random_reference_method,
            threshold=args.threshold,
            tie_policy=args.tie_policy,
        )
    else:
        reference_rate = float(args.random_graph_rate)
    if not 0.0 <= reference_rate <= 1.0:
        raise ValueError(f"Random graph rate must be in [0, 1]. Got {reference_rate}.")

    rng = random.Random(args.random_seed)
    output_rows = [
        prepare_row(
            row=row,
            oof_row=oof_rows.get(str(row["id"])),
            router_specs=router_specs,
            dense_source=args.dense_source,
            graph_source=args.graph_source,
            fusion_source=args.fusion_source,
            top_k=args.top_k,
            threshold=args.threshold,
            tie_policy=args.tie_policy,
            random_graph_rate=reference_rate,
            rng=rng,
            methods_output=methods_output,
        )
        for row in retrieval_rows
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output_rows)

    summary = {
        "retrieval_results": args.retrieval_results,
        "oof_predictions": args.oof_predictions,
        "output": args.output,
        "num_queries": len(output_rows),
        "top_k": args.top_k,
        "threshold": args.threshold,
        "tie_policy": args.tie_policy,
        "router_methods": [
            {"source": spec["source"], "output": spec["output"]}
            for spec in router_specs
        ],
        "random_reference_method": args.random_reference_method,
        "random_graph_rate": reference_rate,
        "random_seed": args.random_seed,
        "methods": methods_output,
        "decision_summary": summarize_decisions(output_rows),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        source = source.strip()
        output = output.strip()
        if not source or not output:
            raise ValueError(f"Invalid router method spec: {item!r}")
        specs.append({"source": source, "output": output})
    if not specs:
        raise ValueError("At least one router method is required.")
    return specs


def compute_reference_graph_rate(
    *,
    rows: list[dict[str, Any]],
    oof_rows: dict[str, dict[str, Any]],
    source_method: str,
    threshold: float,
    tie_policy: str,
) -> float:
    decisions = [
        make_router_decision(
            sample_id=str(row["id"]),
            oof_row=oof_rows.get(str(row["id"])),
            source_method=source_method,
            threshold=threshold,
            tie_policy=tie_policy,
        )["choose_graph"]
        for row in rows
    ]
    return mean(float(value) for value in decisions)


def prepare_row(
    *,
    row: dict[str, Any],
    oof_row: dict[str, Any] | None,
    router_specs: list[dict[str, str]],
    dense_source: str,
    graph_source: str,
    fusion_source: str,
    top_k: int,
    threshold: float,
    tie_policy: str,
    random_graph_rate: float,
    rng: random.Random,
    methods_output: list[str],
) -> dict[str, Any]:
    retrieval = row.get("retrieval", {})
    dense_passages = top_passages(retrieval.get(dense_source, []), top_k)
    graph_passages = top_passages(retrieval.get(graph_source, []), top_k)
    fusion_passages = top_passages(retrieval.get(fusion_source, []), top_k)

    output_retrieval: dict[str, list[dict[str, Any]]] = {
        "dense_only": dense_passages,
        "graph_only": graph_passages,
        "dense_graph_rrf": fusion_passages,
    }
    decisions: dict[str, Any] = {
        "dense_only": {"selected_path": dense_source, "choose_graph": False},
        "graph_only": {"selected_path": graph_source, "choose_graph": True},
        "dense_graph_rrf": {"selected_path": fusion_source, "choose_graph": True},
    }

    for spec in router_specs:
        decision = make_router_decision(
            sample_id=str(row["id"]),
            oof_row=oof_row,
            source_method=spec["source"],
            threshold=threshold,
            tie_policy=tie_policy,
        )
        selected_path = graph_source if decision["choose_graph"] else dense_source
        output_retrieval[spec["output"]] = graph_passages if decision["choose_graph"] else dense_passages
        decisions[spec["output"]] = {
            "selected_path": selected_path,
            "choose_graph": bool(decision["choose_graph"]),
            "router_source_method": spec["source"],
            "probability": decision["probability"],
            "missing_probability": bool(decision["missing_probability"]),
        }

    random_choose_graph = rng.random() < random_graph_rate
    output_retrieval["random_router"] = graph_passages if random_choose_graph else dense_passages
    decisions["random_router"] = {
        "selected_path": graph_source if random_choose_graph else dense_source,
        "choose_graph": bool(random_choose_graph),
        "random_graph_rate": random_graph_rate,
    }

    ordered_retrieval = {
        method: output_retrieval[method]
        for method in methods_output
        if method in output_retrieval
    }
    for method, passages in output_retrieval.items():
        if method not in ordered_retrieval:
            ordered_retrieval[method] = passages

    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row.get("answer"),
        "gold_answer": row.get("gold_answer", row.get("answer")),
        "gold_answers": row.get("gold_answers", [row.get("answer")]),
        "dataset_name": row.get("dataset_name"),
        "workload": row.get("workload"),
        "question_type": row.get("question_type"),
        "retrieval": ordered_retrieval,
        "main_table_decisions": decisions,
        "main_table_methods": list(ordered_retrieval),
    }


def top_passages(value: Any, top_k: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value[:top_k] if isinstance(item, dict)]


def make_router_decision(
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


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    method_names: set[str] = set()
    for row in rows:
        method_names.update(row.get("main_table_decisions", {}))
    summary: dict[str, dict[str, float]] = {}
    for method in sorted(method_names):
        decisions = [
            row["main_table_decisions"][method]
            for row in rows
            if method in row.get("main_table_decisions", {})
        ]
        summary[method] = {
            "num_queries": len(decisions),
            "graph_invocation_rate": mean(float(item.get("choose_graph", False)) for item in decisions),
            "missing_probability_rate": mean(
                float(item.get("missing_probability", False)) for item in decisions
            ),
        }
    return summary


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
