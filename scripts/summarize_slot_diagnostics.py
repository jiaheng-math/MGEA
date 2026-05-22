from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROWS = [
    ("graph-score-slot", "graph_score", "score_slot_graph_5_to_20_cap5_avg7"),
    ("text-rerank", "text_rerank", "slot_text_rerank_graph_5_to_20_cap5_avg7"),
    ("w/o G5-relative features", "passage_rerank", "slot_passage_rerank_graph_5_to_20_cap5_avg7"),
    ("full learned-slot", "full", "slot_graph_5_to_20_cap5_avg7"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize compact slot-selection diagnostics.")
    parser.add_argument(
        "--allocation-reports",
        nargs="+",
        required=True,
        help="One or more eval_adaptive_context_budget.py JSON reports.",
    )
    parser.add_argument(
        "--qa-metrics",
        nargs="*",
        default=[],
        help="Optional one or more evaluate_generations.py metrics JSON files.",
    )
    parser.add_argument("--latex", action="store_true", help="Print a LaTeX table instead of Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [load_json(Path(path)) for path in args.allocation_reports]
    qa_metrics = [load_json(Path(path)) for path in args.qa_metrics]
    rows = [summarize_row(label, variant, method, reports, qa_metrics) for label, variant, method in ROWS]
    if args.latex:
        print_latex(rows)
    else:
        print_markdown(rows)


def summarize_row(
    label: str,
    variant: str | None,
    method: str,
    reports: list[dict[str, Any]],
    qa_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    auc_values: list[float] = []
    selected_positive = 0
    population_positive = 0
    found_plan = False
    seen_auc_keys: set[tuple[str, str]] = set()
    seen_plan_keys: set[tuple[str, str]] = set()

    for report in reports:
        for dataset_report in report.get("reports", []):
            dataset = str(dataset_report.get("dataset") or "")
            model_metrics = dataset_report.get("slot_model_metrics", {})
            auc_key = (dataset, variant or "")
            if variant and variant in model_metrics and auc_key not in seen_auc_keys:
                seen_auc_keys.add(auc_key)
                auc = model_metrics[variant].get("candidate_auc")
                if auc is not None:
                    auc_values.append(float(auc))

            plan_metrics = dataset_report.get("slot_plan_metrics", {})
            plan_key = (dataset, method)
            if method in plan_metrics and plan_key not in seen_plan_keys:
                seen_plan_keys.add(plan_key)
                found_plan = True
                metrics = plan_metrics[method]
                selected_positive += int(metrics.get("selected_positive_n") or 0)
                population_positive += int(metrics.get("population_positive_n") or 0)

    avg_tokens = None
    for metrics_file in qa_metrics:
        method_metrics = metrics_file.get("methods", {}).get(method)
        if method_metrics and method_metrics.get("avg_prompt_tokens") is not None:
            avg_tokens = float(method_metrics["avg_prompt_tokens"])
            break

    return {
        "label": label,
        "method": method,
        "auc": sum(auc_values) / len(auc_values) if auc_values else None,
        "selected_positive": selected_positive if found_plan else None,
        "population_positive": population_positive if found_plan else None,
        "avg_tokens": avg_tokens,
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Method | Slot AUC | Pos. slots selected | Avg Tokens |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | {fmt(row['auc'])} | "
            f"{fmt_selected(row)} | {fmt(row['avg_tokens'], digits=1)} |"
        )


def print_latex(rows: list[dict[str, Any]]) -> None:
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Slot-selection diagnostics under the same AvgK=7 budget.}")
    print(r"\label{tab:slot_selection_diag}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Method & Slot AUC & Pos. slots selected & Avg Tokens \\")
    print(r"\midrule")
    for row in rows:
        label = row["label"]
        if label == "full learned-slot":
            label = r"\textbf{full learned-slot}"
        print(
            f"{label} & {fmt(row['auc'])} & {fmt_selected(row)} & "
            f"{fmt(row['avg_tokens'], digits=1)} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def fmt_selected(row: dict[str, Any]) -> str:
    selected = row.get("selected_positive")
    population = row.get("population_positive")
    if selected is None:
        return "--"
    if population is None:
        return str(selected)
    return f"{selected}/{population}"


def fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
