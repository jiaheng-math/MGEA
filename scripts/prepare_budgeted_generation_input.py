"""Prepare QA generation input for OOF-router-gated budgeted correction.

This materializes retrieval methods that can be consumed by
scripts/batch_generate_from_retrieval.py:

  - dense_only
  - graph_only
  - oof_router_fixed_B5
  - oof_router_value_p<penalty>

The gate is the saved OOF probe-aware router decision. For value-budget methods,
graph-routed queries choose B by fitting out-of-fold action-value models:

    argmax_B predicted_recall(B) - slot_penalty * B

Rows not routed by the OOF gate use dense top-k.

Usage:
  python scripts/prepare_budgeted_generation_input.py \
    --result-dir results/study_hotpot_hipporag_colbert_500 \
    --output results/study_hotpot_hipporag_colbert_500/budgeted_generation_input.jsonl \
    --slot-penalties 0,0.002,0.005
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.features import probe_feature_names, query_feature_names
from eval_budgeted_correction import (
    attach_graph_features,
    cross_validated_fixed_gate_value_policy,
    graph_feature_names,
    load_jsonl,
    load_oof_gate,
    overlap_budget_action,
    parse_float_list,
    ranking_for_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare budgeted QA generation input.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-prob-key", default="query_plus_probe_probability")
    parser.add_argument("--actions", default="0,1,2,3,4,5")
    parser.add_argument("--slot-penalties", default="0,0.002,0.005")
    parser.add_argument(
        "--overlap-thresholds",
        default="0.2,0.6",
        help=(
            "Low,high dense-graph overlap thresholds. Router-gated heuristic uses "
            "B=1 for overlap>=high, B=3 for overlap>=low, else B=5."
        ),
    )
    parser.add_argument(
        "--fixed-budgets",
        default="1,3,5",
        help="Comma-separated fixed B baselines under the same OOF router gate.",
    )
    parser.add_argument(
        "--no-overlap-heuristic",
        action="store_true",
        help="Do not include the dense-graph overlap heuristic method.",
    )
    parser.add_argument(
        "--include-standard",
        action="store_true",
        help="Also include dense_graph_rrf if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = resolve_project_path(args.result_dir)
    output_path = resolve_project_path(args.output)
    actions = sorted({int(item.strip()) for item in args.actions.split(",") if item.strip()})
    fixed_budgets = sorted({int(item.strip()) for item in args.fixed_budgets.split(",") if item.strip()})
    penalties = parse_float_list(args.slot_penalties)
    overlap_thresholds = parse_float_list(args.overlap_thresholds)
    if len(overlap_thresholds) != 2 or overlap_thresholds[0] > overlap_thresholds[1]:
        raise ValueError("--overlap-thresholds must be two ascending floats, e.g. 0.2,0.6")

    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_rows = load_jsonl(result_dir / "retrieval_results.jsonl")
    retrieval_by_id = {str(row["id"]): row for row in retrieval_rows}
    rows = [row for row in routing if row.get("gold_passage_ids")]
    row_ids = [str(row["id"]) for row in rows]

    oof_gate = load_oof_gate(
        result_dir=result_dir,
        rows=rows,
        prob_key=args.router_prob_key,
        threshold=args.router_threshold,
    )
    if oof_gate is None:
        raise FileNotFoundError(
            f"Missing {result_dir / 'pareto_strict_cv' / 'oof_predictions.jsonl'}"
        )

    feature_rows = [attach_graph_features(row, args.top_k) for row in rows]
    feature_names = query_feature_names() + probe_feature_names() + graph_feature_names(args.top_k)

    method_actions: dict[str, list[int]] = {}
    for budget in fixed_budgets:
        method_actions[f"oof_router_fixed_B{budget}"] = [
            budget if choose_graph else 0 for choose_graph in oof_gate
        ]
    if not args.no_overlap_heuristic:
        method_actions["oof_router_overlap_heuristic"] = [
            overlap_budget_action(row, choose_graph, args.top_k, tuple(overlap_thresholds))
            for row, choose_graph in zip(rows, oof_gate)
        ]

    for penalty in penalties:
        report = cross_validated_fixed_gate_value_policy(
            rows=feature_rows,
            gate_decisions=oof_gate,
            feature_names=feature_names,
            actions=actions,
            top_k=args.top_k,
            num_folds=args.num_folds,
            random_seed=args.random_seed,
            slot_penalty=penalty,
            retrieval={},
        )
        method_actions[f"oof_router_value_p{penalty_slug(penalty)}"] = report["predicted_actions"]

    output_rows = []
    for idx, row in enumerate(rows):
        retrieval_row = retrieval_by_id.get(row_ids[idx])
        if retrieval_row is None:
            continue
        retrieval = retrieval_row.get("retrieval", {})
        dense_passages = top_passages(retrieval.get("dense", []), args.top_k)
        graph_passages = top_passages(retrieval.get("graph", []), args.top_k)

        output_retrieval: dict[str, list[dict[str, Any]]] = {
            "dense_only": dense_passages,
            "graph_only": graph_passages,
        }
        if args.include_standard and retrieval.get("fusion"):
            output_retrieval["dense_graph_rrf"] = top_passages(retrieval.get("fusion", []), args.top_k)

        decisions: dict[str, Any] = {
            "dense_only": {"choose_graph": False, "budget": 0},
            "graph_only": {"choose_graph": True, "budget": args.top_k},
        }

        lookup = passage_lookup(retrieval)
        for method, predicted_actions in method_actions.items():
            budget = int(predicted_actions[idx])
            selected_ids = ranking_for_action(row, budget, args.top_k)
            output_retrieval[method] = [lookup[pid] for pid in selected_ids if pid in lookup][: args.top_k]
            decisions[method] = {
                "choose_graph": budget > 0,
                "budget": budget,
                "router_gate": bool(oof_gate[idx]),
            }

        output_rows.append(
            {
                "id": retrieval_row["id"],
                "question": retrieval_row["question"],
                "answer": retrieval_row.get("answer"),
                "gold_answer": retrieval_row.get("gold_answer", retrieval_row.get("answer")),
                "gold_answers": retrieval_row.get("gold_answers", [retrieval_row.get("answer")]),
                "dataset_name": retrieval_row.get("dataset_name"),
                "workload": retrieval_row.get("workload"),
                "question_type": retrieval_row.get("question_type"),
                "retrieval": output_retrieval,
                "main_table_decisions": decisions,
                "main_table_methods": list(output_retrieval),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output_rows)

    summary = {
        "result_dir": str(result_dir),
        "output": str(output_path),
        "num_queries": len(output_rows),
        "top_k": args.top_k,
        "router_threshold": args.router_threshold,
        "router_prob_key": args.router_prob_key,
        "router_gate_rate": sum(oof_gate) / len(oof_gate),
        "methods": list(output_rows[0]["retrieval"]) if output_rows else [],
        "action_distribution": {
            method: dict(Counter(values))
            for method, values in method_actions.items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def top_passages(value: Any, top_k: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value[:top_k] if isinstance(item, dict)]


def passage_lookup(retrieval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for passages in retrieval.values():
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if isinstance(passage, dict) and passage.get("id") not in lookup:
                lookup[str(passage["id"])] = dict(passage)
    return lookup


def penalty_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
