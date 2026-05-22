"""Summarize self-contained relevance vs. conditional marginal slot value.

This script is intentionally offline: it does not load or run any relevance
model, train anything, or rebuild retrieval. Feed it deep graph retrieval
rows plus your existing self-contained relevance labels/scores.

The target statistic is over graph ranks base_k+1 .. slot_max_k, usually
6..20:

  r_rel(q, s)      self-contained relevance from a score threshold
  r_marg(q, G5, s) conditional marginal value under the fixed graph top-5

By default r_marg is the same label used by eval_adaptive_context_budget.py:
the passage is a gold evidence passage not already present in G_base(q).

Example:

  python scripts/summarize_marginal_relevance_mismatch.py \\
    --datasets hotpot 2wiki \\
    --deep-retrieval-files \\
      results/study_hotpot_hipporag_colbert_500/retrieval_results_deep20.jsonl \\
      results/study_2wiki_hipporag_colbert_500/retrieval_results_deep20.jsonl \\
    --self-contained-labels rel=results/self_contained_relevance.jsonl \\
    --output-json results/marginal_relevance_mismatch/summary.json \\
    --output-csv results/marginal_relevance_mismatch/summary.csv

The self-contained relevance file can contain either {qid, pid, label} /
{qid, pid, is_relevant} or {qid, pid, score}. For score files, set
--relevance-threshold NAME=FLOAT if the default threshold 0.0 is not right.

For a retrieval-only sanity check, use --gold-as-relevance. That mode is not
a substitute for a self-contained reranker; it treats only annotated gold
passage ids/titles as relevant.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count where self-contained relevance differs from conditional "
            "marginal slot value over graph ranks base_k+1..slot_max_k."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument(
        "--deep-retrieval-files",
        nargs="+",
        required=True,
        help="One retrieval_results-style JSONL per dataset, same order. "
        "Each row must contain retrieval.graph to at least --slot-max-k.",
    )
    parser.add_argument(
        "--routing-row-files",
        nargs="*",
        default=[],
        help="Optional routing_rows.jsonl per dataset. Defaults to DATASET_MAP.",
    )
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--slot-max-k", type=int, default=20)
    parser.add_argument("--slot-target-avg-k", type=float, default=7.0)
    parser.add_argument("--slot-per-query-cap", type=int, default=5)
    parser.add_argument(
        "--budget-scope",
        choices=["combined", "dataset"],
        default="combined",
        help=(
            "Scope used to select graph-score slots. combined applies one global "
            "AvgK budget across all input datasets; dataset applies the same "
            "AvgK budget independently per dataset. The printed overall and "
            "by-dataset tables always use the same selected flags."
        ),
    )
    parser.add_argument(
        "--self-contained-labels",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Existing self-contained relevance labels/scores. JSONL rows may have "
        "qid/id, pid/passage_id, and label/is_relevant/relevant or score fields.",
    )
    parser.add_argument(
        "--relevance-scores",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Alias for --self-contained-labels, kept for compatibility.",
    )
    parser.add_argument(
        "--relevance-threshold",
        nargs="*",
        default=[],
        metavar="NAME=FLOAT",
        help="Threshold per relevance source. Default is 0.0.",
    )
    parser.add_argument(
        "--relevance-model",
        nargs="*",
        default=[],
        metavar="NAME=MODEL",
        help="Optional model filter per source for score caches containing a model field.",
    )
    parser.add_argument(
        "--gold-as-relevance",
        action="store_true",
        help="Add a gold-label relevance source for debugging/sanity checks.",
    )
    parser.add_argument(
        "--rate-denominator",
        choices=["tail", "relevant", "marginal", "selected"],
        default="tail",
        help="Denominator for the printed one-column paper table.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--examples-output",
        default="",
        help="Optional JSONL with a few example candidates per category.",
    )
    parser.add_argument("--max-examples-per-category", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    routing_files = resolve_routing_files(args)
    score_sources = load_score_sources(args)
    if args.gold_as_relevance:
        score_sources["gold"] = {}

    all_records: list[dict[str, Any]] = []
    dataset_records: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    for dataset, deep_path_s, routing_path in zip(args.datasets, args.deep_retrieval_files, routing_files):
        deep_path = resolve_path(deep_path_s)
        routing_rows = load_jsonl(routing_path) if routing_path.exists() else []
        routing_by_id = {str(row.get("id")): row for row in routing_rows if row.get("id") is not None}
        deep_rows = load_jsonl(deep_path)
        if args.max_queries:
            deep_rows = deep_rows[: args.max_queries]

        records, dataset_warnings = build_candidate_records(dataset, deep_rows, routing_by_id, args)
        dataset_records[dataset] = records
        all_records.extend(records)
        warnings.extend(dataset_warnings)

    report = {
        "config": {
            "datasets": args.datasets,
            "base_k": args.base_k,
            "slot_max_k": args.slot_max_k,
            "slot_target_avg_k": args.slot_target_avg_k,
            "slot_per_query_cap": args.slot_per_query_cap,
            "budget_scope": args.budget_scope,
            "rate_denominator": args.rate_denominator,
            "relevance_sources": sorted(score_sources),
        },
        "warnings": warnings,
        "overall": {},
        "by_dataset": {},
    }

    csv_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for source_name, scores in score_sources.items():
        mark_graph_score_slots_for_scope(all_records, dataset_records, args)
        source_summary, source_csv, source_examples = summarize_source(
            records=all_records,
            source_name=source_name,
            scores=scores,
            threshold=float(source_thresholds(args).get(source_name, 0.0)),
            denominator=args.rate_denominator,
            max_examples=args.max_examples_per_category,
        )
        report["overall"][source_name] = source_summary
        csv_rows.extend(source_csv)
        example_rows.extend(source_examples)

        for dataset, records in dataset_records.items():
            ds_summary, ds_csv, ds_examples = summarize_source(
                records=records,
                source_name=source_name,
                scores=scores,
                threshold=float(source_thresholds(args).get(source_name, 0.0)),
                denominator=args.rate_denominator,
                max_examples=args.max_examples_per_category,
                dataset=dataset,
            )
            report["by_dataset"].setdefault(dataset, {})[source_name] = ds_summary
            csv_rows.extend(ds_csv)
            example_rows.extend(ds_examples)

    print_report(report, args.rate_denominator)

    if args.output_json:
        output_json = resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {output_json}")
    if args.output_csv:
        output_csv = resolve_path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_csv, csv_rows)
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
    if not args.self_contained_labels and not args.relevance_scores and not args.gold_as_relevance:
        raise SystemExit("Pass --self-contained-labels NAME=PATH or --gold-as-relevance.")


def resolve_routing_files(args: argparse.Namespace) -> list[Path]:
    if args.routing_row_files:
        return [resolve_path(path) for path in args.routing_row_files]
    out: list[Path] = []
    for dataset in args.datasets:
        if dataset not in DATASET_MAP:
            raise SystemExit(f"Unknown dataset {dataset!r}; pass --routing-row-files explicitly.")
        out.append(resolve_path(DATASET_MAP[dataset]) / "routing_rows.jsonl")
    return out


def load_score_sources(args: argparse.Namespace) -> dict[str, dict[tuple[str, str], float]]:
    thresholds = source_thresholds(args)
    model_filters = source_models(args)
    sources: dict[str, dict[tuple[str, str], float]] = {}
    inputs = list(args.self_contained_labels) + list(args.relevance_scores)
    for item in inputs:
        name, path_s = parse_name_value(item, "--self-contained-labels")
        path = resolve_path(path_s)
        if not path.exists():
            raise SystemExit(f"Missing relevance score file for {name}: {path}")
        sources[name] = load_score_file(path, model_filters.get(name))
        thresholds.setdefault(name, 0.0)
    return sources


def load_score_file(path: Path, model_filter: str | None) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in load_jsonl(path):
        if model_filter is not None and str(row.get("model")) != model_filter:
            continue
        qid = first_present(row, ["qid", "query_id", "id"])
        pid = first_present(row, ["pid", "passage_id", "doc_id"])
        if qid is None or pid is None:
            continue
        if row.get("score") is not None:
            scores[(str(qid), str(pid))] = float(row["score"])
            continue
        label = first_present(row, ["label", "is_relevant", "relevant", "self_contained_relevant"])
        if label is None:
            continue
        scores[(str(qid), str(pid))] = 1.0 if boolish(label) else -1.0
    return scores


def build_candidate_records(
    dataset: str,
    deep_rows: list[dict[str, Any]],
    routing_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    short_n = 0
    missing_graph_n = 0

    for row in deep_rows:
        qid = str(row.get("id"))
        routing = routing_by_id.get(qid, {})
        retrieval = row.get("retrieval", {}) or {}
        graph_passages = list(retrieval.get("graph") or [])
        if not graph_passages:
            missing_graph_n += 1
            continue
        if len(graph_passages) < args.slot_max_k:
            short_n += 1

        base_ids = dedupe_passage_ids(graph_passages[: args.base_k])
        base_set = set(base_ids)
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

        for idx, passage in enumerate(graph_passages[args.base_k : args.slot_max_k], start=args.base_k + 1):
            pid_value = passage_id(passage)
            if not pid_value or pid_value in seen:
                continue
            seen.add(pid_value)
            records.append(
                {
                    "dataset": dataset,
                    "qid": qid,
                    "pid": pid_value,
                    "rank": idx,
                    "graph_score": float(passage.get("score", 0.0) or 0.0),
                    "marginal_positive": pid_value in missed_gold,
                    "gold_relevant": pid_value in gold,
                    "question": row.get("question") or routing.get("question"),
                    "title": passage.get("title") or passage.get("id"),
                    "text": passage.get("text") or "",
                    "graph_score_slot_selected": False,
                }
            )

    if short_n:
        warnings.append(
            f"{dataset}: {short_n}/{len(deep_rows)} rows have fewer than slot_max_k={args.slot_max_k} graph candidates."
        )
    if missing_graph_n:
        warnings.append(f"{dataset}: {missing_graph_n}/{len(deep_rows)} rows have no retrieval.graph list.")
    if not records:
        warnings.append(f"{dataset}: no tail candidates were collected.")
    return records, warnings


def mark_graph_score_slots(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not records:
        return
    qids = {record["qid"] for record in records}
    total_slots = int(round(len(qids) * (args.slot_target_avg_k - args.base_k)))
    total_slots = max(0, min(total_slots, len(records), args.slot_per_query_cap * len(qids)))
    counts: Counter[str] = Counter()
    selected = 0
    ordered = sorted(records, key=lambda r: (-float(r["graph_score"]), str(r["qid"]), int(r["rank"]), str(r["pid"])))
    for record in ordered:
        if selected >= total_slots:
            break
        if counts[record["qid"]] >= args.slot_per_query_cap:
            continue
        record["graph_score_slot_selected"] = True
        counts[record["qid"]] += 1
        selected += 1


def mark_graph_score_slots_for_scope(
    all_records: list[dict[str, Any]],
    dataset_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    for record in all_records:
        record["graph_score_slot_selected"] = False
    if args.budget_scope == "combined":
        mark_graph_score_slots(all_records, args)
        return
    for records in dataset_records.values():
        mark_graph_score_slots(records, args)


def summarize_source(
    records: list[dict[str, Any]],
    source_name: str,
    scores: dict[tuple[str, str], float],
    threshold: float,
    denominator: str,
    max_examples: int,
    dataset: str = "ALL",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stats = make_stats()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        relevant, score = is_relevant(record, source_name, scores, threshold)
        marginal = bool(record["marginal_positive"])
        selected = bool(record["graph_score_slot_selected"])

        stats["tail_candidates"] += 1
        stats["covered_by_relevance_scores"] += 1 if score is not None or source_name == "gold" else 0
        stats["self_relevant_n"] += int(relevant)
        stats["marginal_positive_n"] += int(marginal)
        stats["graph_score_selected_n"] += int(selected)
        add_category(stats, examples, "relevant_and_marginal_positive", relevant and marginal, record, score, max_examples)
        add_category(stats, examples, "relevant_but_marginal_redundant", relevant and not marginal, record, score, max_examples)
        add_category(stats, examples, "marginal_positive_but_not_top_graph_score", marginal and not selected, record, score, max_examples)
        add_category(stats, examples, "rel_marg_disagree", relevant != marginal, record, score, max_examples)
        add_category(stats, examples, "marginal_positive_but_not_self_relevant", marginal and not relevant, record, score, max_examples)

    summary = finalize_stats(stats, denominator)
    summary["threshold"] = threshold
    summary["dataset"] = dataset
    summary["source"] = source_name

    csv_rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        csv_rows.append(
            {
                "dataset": dataset,
                "source": source_name,
                "threshold": threshold,
                "category": category,
                "n": summary["categories"][category]["n"],
                "rate": summary["categories"][category]["rate"],
                "rate_denominator": denominator,
                "tail_candidates": summary["tail_candidates"],
                "self_relevant_n": summary["self_relevant_n"],
                "marginal_positive_n": summary["marginal_positive_n"],
                "graph_score_selected_n": summary["graph_score_selected_n"],
                "covered_by_relevance_scores": summary["covered_by_relevance_scores"],
            }
        )

    example_rows = []
    for category, rows in examples.items():
        for row in rows:
            example_rows.append({"dataset": dataset, "source": source_name, "category": category, **row})
    return summary, csv_rows, example_rows


CATEGORY_ORDER = [
    "relevant_and_marginal_positive",
    "relevant_but_marginal_redundant",
    "marginal_positive_but_not_top_graph_score",
    "rel_marg_disagree",
    "marginal_positive_but_not_self_relevant",
]


def make_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "tail_candidates": 0,
        "covered_by_relevance_scores": 0,
        "self_relevant_n": 0,
        "marginal_positive_n": 0,
        "graph_score_selected_n": 0,
        "categories": {category: 0 for category in CATEGORY_ORDER},
    }
    return stats


def add_category(
    stats: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    condition: bool,
    record: dict[str, Any],
    score: float | None,
    max_examples: int,
) -> None:
    if not condition:
        return
    stats["categories"][category] += 1
    if len(examples[category]) >= max_examples:
        return
    text = str(record.get("text") or "")
    examples[category].append(
        {
            "qid": record["qid"],
            "pid": record["pid"],
            "rank": record["rank"],
            "graph_score": record["graph_score"],
            "relevance_score": score,
            "marginal_positive": record["marginal_positive"],
            "graph_score_slot_selected": record["graph_score_slot_selected"],
            "question": record.get("question"),
            "title": record.get("title"),
            "text_preview": text[:240],
        }
    )


def finalize_stats(stats: dict[str, Any], denominator: str) -> dict[str, Any]:
    denom = denominator_value(stats, denominator)
    categories = {}
    for category, n in stats["categories"].items():
        categories[category] = {
            "n": int(n),
            "rate": round(n / denom, 6) if denom else 0.0,
            "percent": round(100.0 * n / denom, 2) if denom else 0.0,
        }
    return {
        "tail_candidates": int(stats["tail_candidates"]),
        "covered_by_relevance_scores": int(stats["covered_by_relevance_scores"]),
        "self_relevant_n": int(stats["self_relevant_n"]),
        "marginal_positive_n": int(stats["marginal_positive_n"]),
        "graph_score_selected_n": int(stats["graph_score_selected_n"]),
        "denominator": denominator,
        "denominator_n": int(denom),
        "categories": categories,
    }


def denominator_value(stats: dict[str, Any], denominator: str) -> int:
    if denominator == "tail":
        return int(stats["tail_candidates"])
    if denominator == "relevant":
        return int(stats["self_relevant_n"])
    if denominator == "marginal":
        return int(stats["marginal_positive_n"])
    if denominator == "selected":
        return int(stats["graph_score_selected_n"])
    raise ValueError(denominator)


def is_relevant(
    record: dict[str, Any],
    source_name: str,
    scores: dict[tuple[str, str], float],
    threshold: float,
) -> tuple[bool, float | None]:
    if source_name == "gold":
        return bool(record["gold_relevant"]), 1.0 if record["gold_relevant"] else 0.0
    score = scores.get((str(record["qid"]), str(record["pid"])))
    if score is None:
        return False, None
    return score >= threshold, score


def print_report(report: dict[str, Any], denominator: str) -> None:
    print(f"Rate denominator: {denominator}")
    if report.get("warnings"):
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    for source, summary in report["overall"].items():
        print(f"\nOverall: {source} (threshold={summary['threshold']})")
        print_compact_table(summary)

    if report["by_dataset"]:
        print("\nBy dataset:")
        for dataset, by_source in report["by_dataset"].items():
            for source, summary in by_source.items():
                print(f"\n{dataset}: {source} (threshold={summary['threshold']})")
                print_compact_table(summary)


def print_compact_table(summary: dict[str, Any]) -> None:
    print(
        "tail={tail_candidates} relevant={self_relevant_n} marginal={marginal_positive_n} "
        "graph_score_selected={graph_score_selected_n} denom={denominator_n}".format(**summary)
    )
    print("| Category | Rate | Count |")
    print("|---|---:|---:|")
    labels = {
        "relevant_and_marginal_positive": "relevant and marginal positive",
        "relevant_but_marginal_redundant": "relevant but marginal redundant",
        "marginal_positive_but_not_top_graph_score": "marginal positive but not top graph-score",
        "marginal_positive_but_not_self_relevant": "marginal positive but not self-contained relevant",
    }
    for key, label in labels.items():
        item = summary["categories"][key]
        print(f"| {label} | {item['percent']:.2f}% | {item['n']} |")
    disagree = summary["categories"]["rel_marg_disagree"]
    print(f"r_rel != r_marg: {disagree['n']} ({disagree['percent']:.2f}%)")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "source",
        "threshold",
        "category",
        "n",
        "rate",
        "rate_denominator",
        "tail_candidates",
        "self_relevant_n",
        "marginal_positive_n",
        "graph_score_selected_n",
        "covered_by_relevance_scores",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def source_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {name: float(value) for name, value in (parse_name_value(item, "--relevance-threshold") for item in args.relevance_threshold)}


def source_models(args: argparse.Namespace) -> dict[str, str]:
    return {name: value for name, value in (parse_name_value(item, "--relevance-model") for item in args.relevance_model)}


def parse_name_value(item: str, flag: str) -> tuple[str, str]:
    if "=" not in item:
        raise SystemExit(f"{flag} entries must be NAME=VALUE, got: {item}")
    name, value = item.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise SystemExit(f"{flag} entries must be NAME=VALUE, got: {item}")
    return name, value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "relevant", "positive", "pos"}:
        return True
    if text in {"0", "false", "no", "n", "irrelevant", "negative", "neg", ""}:
        return False
    return bool(text)


def passage_id(passage: Any) -> str:
    if not isinstance(passage, dict):
        return ""
    value = passage.get("id") or passage.get("source_doc_id") or passage.get("title")
    return str(value) if value is not None else ""


def dedupe_passage_ids(passages: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for passage in passages:
        pid = passage_id(passage)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


if __name__ == "__main__":
    main()
