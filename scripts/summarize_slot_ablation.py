"""Summarize slot feature ablations.

Reads the allocation report from eval_adaptive_context_budget.py and, optionally,
one QA metrics JSON from evaluate_generations.py. Prints compact Markdown tables
for the paper-facing ablation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANT_METHODS = {
    "Full": ("full", "slot_graph_5_to_20_cap5_avg7"),
    "passage-rerank": ("passage_rerank", "slot_passage_rerank_graph_5_to_20_cap5_avg7"),
    "text-rerank": ("text_rerank", "slot_text_rerank_graph_5_to_20_cap5_avg7"),
    "w/o novelty": ("no_novelty", "slot_no_novelty_graph_5_to_20_cap5_avg7"),
    "w/o dense-support": ("no_dense_support", "slot_no_dense_support_graph_5_to_20_cap5_avg7"),
    "No-Probe": ("no_probe", "slot_no_probe_graph_5_to_20_cap5_avg7"),
    "Probe-Only": ("probe_only", "slot_probe_only_graph_5_to_20_cap5_avg7"),
    "slot-local only": ("slot_only", "slot_slot_only_graph_5_to_20_cap5_avg7"),
    "graph rank/score only": ("graph_only", "slot_graph_only_graph_5_to_20_cap5_avg7"),
}

DEFAULT_VARIANTS = ["Full", "No-Probe"]
PAPER_FEATURE_VARIANTS = [
    "Full",
    "text-rerank",
    "passage-rerank",
    "w/o novelty",
    "w/o dense-support",
    "No-Probe",
    "slot-local only",
    "graph rank/score only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize slot feature ablation outputs.")
    parser.add_argument("--allocation-report", required=True, help="marginal_slot_allocation_ablation_eval.json")
    parser.add_argument("--qa-metrics", default=None, help="Optional qa_metrics_<reader>.json")
    parser.add_argument("--include-probe-only", action="store_true")
    parser.add_argument(
        "--paper-feature-ablation",
        action="store_true",
        help="Print the broader feature ablation table for paper diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_json(Path(args.allocation_report))
    qa_metrics = load_json(Path(args.qa_metrics)) if args.qa_metrics else None
    variants = PAPER_FEATURE_VARIANTS if args.paper_feature_ablation else list(DEFAULT_VARIANTS)
    if args.include_probe_only:
        variants.append("Probe-Only")

    print("## Slot Model Ablation\n")
    print("| Dataset | Variant | AUC | AP | Selected positive slots | AvgK |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset_report in report.get("reports", []):
        dataset = dataset_report.get("dataset")
        model_metrics = dataset_report.get("slot_model_metrics", {})
        plan_metrics = dataset_report.get("slot_plan_metrics", {})
        for label in variants:
            variant_key, method = VARIANT_METHODS[label]
            mm = model_metrics.get(variant_key, {})
            pm = plan_metrics.get(method, {})
            selected = format_selected_positive(pm)
            print(
                f"| {dataset} | {label} | {fmt(mm.get('candidate_auc'))} | "
                f"{fmt(mm.get('candidate_ap'))} | {selected} | {fmt(pm.get('avg_k'))} |"
            )

    if qa_metrics:
        print("\n## QA Ablation\n")
        print("| Variant | Method | EM | F1 | Avg Prompt Tok |")
        print("|---|---|---:|---:|---:|")
        method_metrics = qa_metrics.get("methods", {})
        for label in variants:
            _, method = VARIANT_METHODS[label]
            metrics = method_metrics.get(method, {})
            print(
                f"| {label} | `{method}` | {fmt(metrics.get('exact_match'))} | "
                f"{fmt(metrics.get('f1'))} | {fmt(metrics.get('avg_prompt_tokens'))} |"
            )

        for method in [
            "graph_top5",
            "random_slot_graph_5_to_20_cap5_avg7",
            "score_slot_graph_5_to_20_cap5_avg7",
            "graph_top8",
            "graph_top10",
        ]:
            metrics = method_metrics.get(method)
            if not metrics:
                continue
            print(
                f"| Baseline | `{method}` | {fmt(metrics.get('exact_match'))} | "
                f"{fmt(metrics.get('f1'))} | {fmt(metrics.get('avg_prompt_tokens'))} |"
            )


def format_selected_positive(metrics: dict[str, Any]) -> str:
    selected = metrics.get("selected_positive_n")
    population = metrics.get("population_positive_n")
    if selected is None:
        return ""
    if population is None:
        return str(selected)
    return f"{selected} / {population}"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
