from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable


DEFAULT_METHODS = [
    "dense_only",
    "graph_only",
    "oof_router_fixed_B1",
    "oof_router_fixed_B3",
    "oof_router_fixed_B5",
    "oof_router_overlap_heuristic",
    "oof_router_value_p0",
]

METHOD_LABELS = {
    "dense_only": "Dense-only",
    "graph_only": "Graph-only",
    "oof_router_fixed_B1": "Router + fixed B=1",
    "oof_router_fixed_B3": "Router + fixed B=3",
    "oof_router_fixed_B5": "Router + fixed B=5",
    "oof_router_overlap_heuristic": "Router + overlap heuristic",
    "oof_router_value_p0": "Router + value budget",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize dense-vs-graph evidence exposure for budgeted generation inputs. "
            "Graph evidence is counted by the method decision budget: for top-k evidence, "
            "the first k-B passages are dense-retained and the final B passages are graph-injected."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "JSONL"),
        required=True,
        help="Dataset label and budgeted_generation_input*.jsonl path. Can be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated retrieval methods to summarize.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["auto", "regex"],
        default="auto",
        help="Use tiktoken cl100k_base when available, otherwise regex tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    count_tokens, tokenizer_name = build_token_counter(args.tokenizer)

    rows: list[dict[str, Any]] = []
    for dataset_name, input_path in args.dataset:
        input_rows = load_jsonl(Path(input_path))
        for method in methods:
            rows.append(
                summarize_method(
                    dataset_name=dataset_name,
                    rows=input_rows,
                    method=method,
                    top_k=args.top_k,
                    count_tokens=count_tokens,
                    tokenizer_name=tokenizer_name,
                )
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output_csv": str(output_csv), "num_rows": len(rows), "tokenizer": tokenizer_name}, indent=2))


def build_token_counter(tokenizer: str) -> tuple[Callable[[str], int], str]:
    if tokenizer == "auto":
        try:
            import tiktoken  # type: ignore

            encoding = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(encoding.encode(text or "")), "tiktoken:cl100k_base"
        except Exception:
            pass
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    return lambda text: len(pattern.findall(text or "")), "regex"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_method(
    dataset_name: str,
    rows: list[dict[str, Any]],
    method: str,
    top_k: int,
    count_tokens: Callable[[str], int],
    tokenizer_name: str,
) -> dict[str, Any]:
    dense_slots: list[int] = []
    graph_slots: list[int] = []
    dense_tokens: list[int] = []
    graph_tokens: list[int] = []
    total_tokens: list[int] = []

    for row in rows:
        retrieval = row.get("retrieval", {})
        passages = retrieval.get(method, [])
        if not isinstance(passages, list):
            passages = []
        passages = passages[:top_k]

        decision = row.get("main_table_decisions", {}).get(method, {})
        budget = int(decision.get("budget", top_k if method == "graph_only" else 0))
        budget = max(0, min(top_k, budget))
        dense_count = max(0, min(len(passages), top_k - budget))
        graph_count = max(0, len(passages) - dense_count)

        d_tokens = sum(passage_tokens(passage, count_tokens) for passage in passages[:dense_count])
        g_tokens = sum(passage_tokens(passage, count_tokens) for passage in passages[dense_count:])

        dense_slots.append(dense_count)
        graph_slots.append(graph_count)
        dense_tokens.append(d_tokens)
        graph_tokens.append(g_tokens)
        total_tokens.append(d_tokens + g_tokens)

    avg_dense_tokens = safe_mean(dense_tokens)
    avg_graph_tokens = safe_mean(graph_tokens)
    avg_total_tokens = safe_mean(total_tokens)
    return {
        "dataset": dataset_name,
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "num_samples": len(rows),
        "avg_dense_slots": safe_mean(dense_slots),
        "avg_graph_slots": safe_mean(graph_slots),
        "avg_dense_evidence_tokens": avg_dense_tokens,
        "avg_graph_evidence_tokens": avg_graph_tokens,
        "avg_total_evidence_tokens": avg_total_tokens,
        "graph_evidence_token_share": avg_graph_tokens / avg_total_tokens if avg_total_tokens else 0.0,
        "tokenizer": tokenizer_name,
    }


def passage_tokens(passage: Any, count_tokens: Callable[[str], int]) -> int:
    if not isinstance(passage, dict):
        return 0
    title = str(passage.get("title") or passage.get("id") or "")
    text = str(passage.get("text") or "")
    return count_tokens(f"{title}\n{text}".strip())


def safe_mean(values: list[int]) -> float:
    return float(mean(values)) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "label",
        "num_samples",
        "avg_dense_slots",
        "avg_graph_slots",
        "avg_dense_evidence_tokens",
        "avg_graph_evidence_tokens",
        "avg_total_evidence_tokens",
        "graph_evidence_token_share",
        "tokenizer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":
    main()
