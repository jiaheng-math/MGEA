from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.study_main import summarize_routing_methods
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel saved routing_rows.jsonl and rerun routing evaluation without rerunning retrieval."
    )
    parser.add_argument("--input", required=True, help="Path to saved routing_rows.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for relabeled rows and rerun metrics.")
    parser.add_argument(
        "--policy",
        default="strict",
        choices=["strict", "moderate", "lenient", "threshold"],
        help="Dense sufficiency policy for correction labels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit threshold when --policy=threshold.",
    )
    parser.add_argument(
        "--label-k",
        type=int,
        default=5,
        help="Which recall@k field to use for labeling, e.g. 5 uses dense_recall@5 and graph_recall@5.",
    )
    parser.add_argument(
        "--top-k-values",
        default="3,5",
        help="Comma-separated top-k values for routing summary output.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio for train_test mode.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--evaluation-mode",
        default="train_test",
        choices=["train_test", "cv"],
        help="Routing evaluation mode.",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds when evaluation-mode=cv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    output_dir = ensure_dir(args.output_dir)
    threshold = resolve_threshold(args.policy, args.threshold)
    relabeled_rows = [
        relabel_row(row=row, label_k=args.label_k, policy=args.policy, threshold=threshold)
        for row in rows
    ]

    with (Path(output_dir) / "routing_rows_relabeled.jsonl").open("w", encoding="utf-8") as handle:
        for row in relabeled_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_k_values = [int(value.strip()) for value in args.top_k_values.split(",") if value.strip()]
    routing_summary = summarize_routing_methods(
        rows=relabeled_rows,
        top_k_values=top_k_values,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        evaluation_mode=args.evaluation_mode,
        num_folds=args.num_folds,
    )

    payload = {
        "input": args.input,
        "label_k": args.label_k,
        "correction_label_policy": args.policy,
        "correction_threshold": threshold,
        "evaluation_mode": args.evaluation_mode,
        "num_folds": args.num_folds,
        "train_ratio": args.train_ratio,
        "top_k_values": top_k_values,
        "routing_methods": routing_summary,
    }
    with (Path(output_dir) / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

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
        return 1e-12
    if threshold is None:
        raise ValueError("--threshold is required when --policy=threshold")
    return threshold


def relabel_row(row: dict[str, Any], label_k: int, policy: str, threshold: float) -> dict[str, Any]:
    dense_key = f"dense_recall@{label_k}"
    graph_key = f"graph_recall@{label_k}"
    if dense_key not in row or graph_key not in row:
        raise KeyError(f"Missing required keys: {dense_key}, {graph_key}")

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


if __name__ == "__main__":
    main()
