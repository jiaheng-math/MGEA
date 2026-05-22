"""Summarize how many marginal positives each slot selector captures.

This is the selector-level diagnostic for the reviewer question:

  Selector           Selected positives     Missed positives
  graph-score-slot   80/150                 70/150
  MonoT5-slot        ?/150                  ?/150
  MGEA               99/150                 51/150

The script uses deep graph retrieval to define the population:

  tail candidates: graph ranks base_k+1 .. slot_max_k, usually 6..20
  marginal positive: gold evidence not already present in G_base(q)

It then reads one or more generation-input JSONL files and checks which tail
passages each selector method actually put into the method context.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_MAP = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki": "results/study_2wiki_hipporag_colbert_500",
    "nq": "results/study_nq_hipporag_colbert_500",
}


DEFAULT_SELECTORS = [
    ("graph-score-slot", "score_slot_graph_5_to_20_cap5_avg7"),
    ("text-only slot", "slot_text_rerank_graph_5_to_20_cap5_avg7"),
    ("MonoT5-slot", "monot5_slot_graph_5_to_20_cap5_avg7"),
    ("MGEA", "slot_graph_5_to_20_cap5_avg7"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count selected/missed marginal positives for slot selectors."
    )
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument(
        "--deep-retrieval-files",
        nargs="+",
        required=True,
        help="One retrieval_results-style deep JSONL per dataset, same order.",
    )
    parser.add_argument(
        "--routing-row-files",
        nargs="*",
        default=[],
        help="Optional routing_rows.jsonl per dataset. Defaults to DATASET_MAP.",
    )
    parser.add_argument(
        "--generation-input",
        nargs="+",
        required=True,
        help="One or more generation-input JSONL files containing retrieval[method].",
    )
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--slot-max-k", type=int, default=20)
    parser.add_argument(
        "--selectors",
        nargs="*",
        default=[],
        metavar="LABEL=METHOD",
        help="Selector label and generation-input method name. Defaults to graph-score/text-only/MonoT5/MGEA.",
    )
    parser.add_argument(
        "--allow-missing-selectors",
        action="store_true",
        help="Keep selectors with zero matching rows instead of failing fast.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--examples-output",
        default="",
        help="Optional JSONL with missed marginal-positive examples per selector.",
    )
    parser.add_argument("--max-examples-per-selector", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    selectors = parse_selectors(args.selectors)
    routing_files = resolve_routing_files(args)
    generation_rows = load_generation_rows([resolve_path(path) for path in args.generation_input])
    validate_selector_methods(selectors, generation_rows, args.allow_missing_selectors)

    populations: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for dataset, deep_path_s, routing_path in zip(args.datasets, args.deep_retrieval_files, routing_files):
        deep_rows = load_jsonl(resolve_path(deep_path_s))
        if args.max_queries:
            deep_rows = deep_rows[: args.max_queries]
        routing_rows = load_jsonl(routing_path) if routing_path.exists() else []
        routing_by_id = {str(row.get("id")): row for row in routing_rows if row.get("id") is not None}
        ds_population, ds_warnings = build_population(dataset, deep_rows, routing_by_id, args)
        populations.update(ds_population)
        warnings.extend(ds_warnings)

    rows = []
    example_rows = []
    for label, method in selectors:
        summary, examples = summarize_selector(
            label=label,
            method=method,
            population=populations,
            generation_rows=generation_rows,
            max_examples=args.max_examples_per_selector,
        )
        rows.append(summary)
        example_rows.extend(examples)

    report = {
        "config": {
            "datasets": args.datasets,
            "deep_retrieval_files": args.deep_retrieval_files,
            "generation_input": args.generation_input,
            "base_k": args.base_k,
            "slot_max_k": args.slot_max_k,
            "selectors": [{"label": label, "method": method} for label, method in selectors],
        },
        "warnings": warnings,
        "population": population_summary(populations),
        "selectors": rows,
    }

    print_report(report)

    if args.output_json:
        output_json = resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {output_json}")
    if args.output_csv:
        output_csv = resolve_path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_csv, rows)
        print(f"CSV:  {output_csv}")
    if args.examples_output:
        examples_output = resolve_path(args.examples_output)
        examples_output.parent.mkdir(parents=True, exist_ok=True)
        with examples_output.open("w", encoding="utf-8") as handle:
            for row in example_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Examples: {examples_output}")


def validate_args(args: argparse.Namespace) -> None:
    if len(args.deep_retrieval_files) != len(args.datasets):
        raise SystemExit("--deep-retrieval-files must match --datasets in length/order.")
    if args.routing_row_files and len(args.routing_row_files) != len(args.datasets):
        raise SystemExit("--routing-row-files must match --datasets in length/order.")
    if args.slot_max_k <= args.base_k:
        raise SystemExit("--slot-max-k must be greater than --base-k.")


def parse_selectors(items: list[str]) -> list[tuple[str, str]]:
    if not items:
        return list(DEFAULT_SELECTORS)
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--selectors entries must be LABEL=METHOD, got: {item}")
        label, method = item.split("=", 1)
        label = label.strip()
        method = method.strip()
        if not label or not method:
            raise SystemExit(f"--selectors entries must be LABEL=METHOD, got: {item}")
        out.append((label, method))
    return out


def resolve_routing_files(args: argparse.Namespace) -> list[Path]:
    if args.routing_row_files:
        return [resolve_path(path) for path in args.routing_row_files]
    out = []
    for dataset in args.datasets:
        if dataset not in DATASET_MAP:
            raise SystemExit(f"Unknown dataset {dataset!r}; pass --routing-row-files explicitly.")
        out.append(resolve_path(DATASET_MAP[dataset]) / "routing_rows.jsonl")
    return out


def build_population(
    dataset: str,
    deep_rows: list[dict[str, Any]],
    routing_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    population: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    short_n = 0
    missing_graph_n = 0
    for row in deep_rows:
        qid = str(row.get("id"))
        routing = routing_by_id.get(qid, {})
        graph_passages = list((row.get("retrieval", {}) or {}).get("graph") or [])
        if not graph_passages:
            missing_graph_n += 1
            continue
        if len(graph_passages) < args.slot_max_k:
            short_n += 1
        base = dedupe_passage_ids(graph_passages[: args.base_k])
        base_set = set(base)
        gold = set(
            str(pid)
            for pid in (
                row.get("gold_passage_ids")
                or row.get("gold_titles")
                or routing.get("gold_passage_ids")
                or routing.get("gold_titles")
                or []
            )
        )
        missed_gold = gold - base_set
        seen = set(base_set)
        for rank, passage in enumerate(graph_passages[args.base_k : args.slot_max_k], start=args.base_k + 1):
            pid = passage_id(passage)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            key = record_key(qid, pid)
            population[key] = {
                "dataset": dataset,
                "qid": qid,
                "pid": pid,
                "rank": rank,
                "marginal_positive": pid in missed_gold,
                "question": row.get("question") or routing.get("question"),
                "title": passage.get("title") or passage.get("id"),
                "text": passage.get("text") or "",
            }
    if short_n:
        warnings.append(
            f"{dataset}: {short_n}/{len(deep_rows)} rows have fewer than slot_max_k={args.slot_max_k} graph candidates."
        )
    if missing_graph_n:
        warnings.append(f"{dataset}: {missing_graph_n}/{len(deep_rows)} rows have no retrieval.graph list.")
    return population, warnings


def load_generation_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing generation input: {path}")
        for row in load_jsonl(path):
            qid = row.get("id")
            if qid is None:
                continue
            qid_s = str(qid)
            if qid_s not in rows_by_id:
                rows_by_id[qid_s] = row
                continue
            rows_by_id[qid_s] = merge_generation_row(rows_by_id[qid_s], row)
    return rows_by_id


def validate_selector_methods(
    selectors: list[tuple[str, str]],
    generation_rows: dict[str, dict[str, Any]],
    allow_missing: bool,
) -> None:
    available = available_methods(generation_rows)
    missing = [(label, method) for label, method in selectors if method not in available]
    if not missing or allow_missing:
        return
    lines = [
        "Some selector methods are absent from the provided --generation-input files.",
        "This means their selected-positive counts would be 0 only because the method was not joined in.",
        "",
        "Missing selectors:",
    ]
    lines.extend(f"  - {label}: {method}" for label, method in missing)
    lines.append("")
    lines.append("Available methods:")
    lines.extend(f"  - {method}" for method in sorted(available))
    lines.append("")
    lines.append("Pass the generation input that contains the missing method, correct --selectors,")
    lines.append("or use --allow-missing-selectors only for debugging.")
    raise SystemExit("\n".join(lines))


def available_methods(generation_rows: dict[str, dict[str, Any]]) -> set[str]:
    methods: set[str] = set()
    for row in generation_rows.values():
        retrieval = row.get("retrieval") or {}
        if isinstance(retrieval, dict):
            methods.update(str(method) for method in retrieval)
        main_methods = row.get("main_table_methods") or []
        if isinstance(main_methods, list):
            methods.update(str(method) for method in main_methods)
    return methods


def merge_generation_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    retrieval = dict(left.get("retrieval") or {})
    retrieval.update(dict(right.get("retrieval") or {}))
    methods = list(left.get("main_table_methods") or retrieval.keys())
    for method in right.get("main_table_methods") or list((right.get("retrieval") or {}).keys()):
        if method not in methods:
            methods.append(method)
    merged["retrieval"] = retrieval
    merged["main_table_methods"] = methods
    return merged


def summarize_selector(
    label: str,
    method: str,
    population: dict[str, dict[str, Any]],
    generation_rows: dict[str, dict[str, Any]],
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    marginal_keys = {key for key, record in population.items() if record["marginal_positive"]}
    selected_keys: set[str] = set()
    selected_tail_keys: set[str] = set()
    method_present_rows = 0

    for qid, row in generation_rows.items():
        passages = (row.get("retrieval", {}) or {}).get(method)
        if not isinstance(passages, list):
            continue
        method_present_rows += 1
        for passage in passages:
            pid = passage_id(passage)
            if not pid:
                continue
            key = record_key(qid, pid)
            if key in population:
                selected_tail_keys.add(key)
                selected_keys.add(key)

    selected_positive = selected_keys & marginal_keys
    missed_positive = marginal_keys - selected_positive
    total_positive = len(marginal_keys)

    by_dataset = {}
    for dataset in sorted({record["dataset"] for record in population.values()}):
        ds_marginal = {key for key, record in population.items() if record["dataset"] == dataset and record["marginal_positive"]}
        ds_selected_positive = selected_keys & ds_marginal
        by_dataset[dataset] = selector_counts(len(ds_selected_positive), len(ds_marginal), len(ds_marginal - ds_selected_positive))

    examples = []
    for key in sorted(missed_positive, key=lambda item: (population[item]["dataset"], population[item]["qid"], population[item]["rank"], population[item]["pid"]))[:max_examples]:
        record = population[key]
        examples.append(
            {
                "selector": label,
                "method": method,
                "dataset": record["dataset"],
                "qid": record["qid"],
                "pid": record["pid"],
                "rank": record["rank"],
                "question": record.get("question"),
                "title": record.get("title"),
                "text_preview": str(record.get("text") or "")[:240],
            }
        )

    summary = {
        "selector": label,
        "method": method,
        "method_present_rows": method_present_rows,
        "selected_tail_slots": len(selected_tail_keys),
        **selector_counts(len(selected_positive), total_positive, len(missed_positive)),
        "by_dataset": by_dataset,
    }
    return summary, examples


def selector_counts(selected_positive: int, total_positive: int, missed_positive: int) -> dict[str, Any]:
    return {
        "selected_positive_n": selected_positive,
        "missed_positive_n": missed_positive,
        "population_positive_n": total_positive,
        "selected_positive": f"{selected_positive}/{total_positive}",
        "missed_positive": f"{missed_positive}/{total_positive}",
        "selected_positive_rate": round(selected_positive / total_positive, 6) if total_positive else 0.0,
        "missed_positive_rate": round(missed_positive / total_positive, 6) if total_positive else 0.0,
    }


def population_summary(population: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_dataset = defaultdict(lambda: {"tail_candidates": 0, "marginal_positive_n": 0})
    for record in population.values():
        item = by_dataset[record["dataset"]]
        item["tail_candidates"] += 1
        item["marginal_positive_n"] += int(record["marginal_positive"])
    return {
        "tail_candidates": len(population),
        "marginal_positive_n": sum(int(record["marginal_positive"]) for record in population.values()),
        "by_dataset": dict(by_dataset),
    }


def print_report(report: dict[str, Any]) -> None:
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
        print()
    pop = report["population"]
    print(f"Population: tail={pop['tail_candidates']} marginal_positive={pop['marginal_positive_n']}")
    print("| Selector | Selected positives | Missed positives | Selected tail slots | Rows with method |")
    print("|---|---:|---:|---:|---:|")
    for row in report["selectors"]:
        print(
            f"| {row['selector']} | {row['selected_positive']} "
            f"({100 * row['selected_positive_rate']:.1f}%) | "
            f"{row['missed_positive']} ({100 * row['missed_positive_rate']:.1f}%) | "
            f"{row['selected_tail_slots']} | {row['method_present_rows']} |"
        )
    print("\nBy dataset:")
    for row in report["selectors"]:
        print(f"\n{row['selector']}")
        print("| Dataset | Selected positives | Missed positives |")
        print("|---|---:|---:|")
        for dataset, counts in row["by_dataset"].items():
            print(
                f"| {dataset} | {counts['selected_positive']} "
                f"({100 * counts['selected_positive_rate']:.1f}%) | "
                f"{counts['missed_positive']} ({100 * counts['missed_positive_rate']:.1f}%) |"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "selector",
        "method",
        "method_present_rows",
        "selected_tail_slots",
        "selected_positive_n",
        "population_positive_n",
        "missed_positive_n",
        "selected_positive_rate",
        "missed_positive_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def passage_id(passage: Any) -> str:
    if not isinstance(passage, dict):
        return ""
    value = passage.get("id") or passage.get("source_doc_id") or passage.get("title")
    return str(value) if value is not None else ""


def dedupe_passage_ids(passages: list[Any]) -> list[str]:
    seen = set()
    out = []
    for passage in passages:
        pid = passage_id(passage)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def record_key(qid: str, pid: str) -> str:
    return f"{qid}\t{pid}"


if __name__ == "__main__":
    main()
