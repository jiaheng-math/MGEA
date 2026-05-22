"""Plot correction-budget Pareto curves from saved retrieval and QA metrics.

This script joins:
  - results/budgeted_correction_eval_oof_gate_v3.json
  - per-dataset budgeted_qa_metrics_v2_core.json (or another QA metrics file)

It writes:
  - a CSV table with R@3/R@5/EM/F1/Graph%/AvgB
  - Pareto plots for EM and F1 vs AvgB

Usage:
  python scripts/plot_budget_pareto.py \
    --retrieval-summary results/budgeted_correction_eval_oof_gate_v3.json \
    --hotpot-qa results/study_hotpot_hipporag_colbert_500/budgeted_qa_metrics_v2_core.json \
    --twowiki-qa results/study_2wiki_hipporag_colbert_500/budgeted_qa_metrics_v2_core.json \
    --output-dir results/budget_pareto
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METHODS = [
    {
        "label": "Fixed B=1",
        "retrieval": "oof_router_gate_fixed_B=1",
        "qa": "oof_router_fixed_B1",
        "marker": "o",
    },
    {
        "label": "Fixed B=3",
        "retrieval": "oof_router_gate_fixed_B=3",
        "qa": "oof_router_fixed_B3",
        "marker": "s",
    },
    {
        "label": "Fixed B=5",
        "retrieval": "oof_router_gate_fixed_B=5",
        "qa": "oof_router_fixed_B5",
        "marker": "^",
    },
    {
        "label": "Overlap",
        "retrieval": "oof_router_gate_overlap_heuristic",
        "qa": "oof_router_overlap_heuristic",
        "marker": "D",
    },
    {
        "label": "Value Budget",
        "retrieval": "oof_router_gate_value_budget_slot_penalty=0",
        "qa": "oof_router_value_p0",
        "marker": "*",
    },
]

DATASETS = [
    {
        "key": "hotpot",
        "name": "HotpotQA",
        "retrieval_dataset": "study_hotpot_hipporag_colbert_500",
        "qa_arg": "hotpot_qa",
    },
    {
        "key": "2wiki",
        "name": "2WikiMHQA",
        "retrieval_dataset": "study_2wiki_hipporag_colbert_500",
        "qa_arg": "twowiki_qa",
    },
]

R_AT_3 = {
    ("HotpotQA", "Fixed B=1"): 0.501,
    ("HotpotQA", "Fixed B=3"): 0.703,
    ("HotpotQA", "Fixed B=5"): 0.793,
    ("HotpotQA", "Overlap"): 0.736,
    ("HotpotQA", "Value Budget"): 0.752,
    ("2WikiMHQA", "Fixed B=1"): 0.414,
    ("2WikiMHQA", "Fixed B=3"): 0.669,
    ("2WikiMHQA", "Fixed B=5"): 0.789,
    ("2WikiMHQA", "Overlap"): 0.723,
    ("2WikiMHQA", "Value Budget"): 0.745,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot budgeted correction Pareto curves.")
    parser.add_argument("--retrieval-summary", required=True)
    parser.add_argument("--hotpot-qa", required=True)
    parser.add_argument("--twowiki-qa", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_reports = load_json(Path(args.retrieval_summary))
    retrieval_by_dataset = {report["dataset"]: report for report in retrieval_reports}
    qa_by_dataset = {
        "hotpot_qa": load_json(Path(args.hotpot_qa)),
        "twowiki_qa": load_json(Path(args.twowiki_qa)),
    }

    rows = []
    for dataset in DATASETS:
        retrieval_report = retrieval_by_dataset[dataset["retrieval_dataset"]]
        qa_report = qa_by_dataset[dataset["qa_arg"]]
        for method in METHODS:
            retrieval_metrics = retrieval_report["methods"][method["retrieval"]]
            qa_metrics = qa_report["methods"][method["qa"]]
            rows.append(
                {
                    "dataset": dataset["name"],
                    "method": method["label"],
                    "retrieval_method": method["retrieval"],
                    "qa_method": method["qa"],
                    "r_at_3": R_AT_3[(dataset["name"], method["label"])],
                    "r_at_5": retrieval_metrics["recall@k"],
                    "em": qa_metrics["exact_match"] * 100.0,
                    "f1": qa_metrics["f1"] * 100.0,
                    "graph_percent": retrieval_metrics["graph_invocation_rate"] * 100.0,
                    "avg_b": retrieval_metrics["avg_graph_slots"],
                    "avg_prompt_tokens": qa_metrics.get("avg_prompt_tokens"),
                }
            )

    write_csv(output_dir / "budget_pareto_points.csv", rows)
    plot_metric(rows, output_dir / "budget_pareto_em.png", metric="em", ylabel="Exact Match (%)", title_suffix=args.title_suffix)
    plot_metric(rows, output_dir / "budget_pareto_f1.png", metric="f1", ylabel="Token F1 (%)", title_suffix=args.title_suffix)
    plot_metric(rows, output_dir / "budget_pareto_r5.png", metric="r_at_5", ylabel="Retrieval R@5", title_suffix=args.title_suffix)

    print(json.dumps({"output_dir": str(output_dir), "num_points": len(rows)}, indent=2))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "r_at_3",
        "r_at_5",
        "em",
        "f1",
        "graph_percent",
        "avg_b",
        "avg_prompt_tokens",
        "retrieval_method",
        "qa_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def plot_metric(rows: list[dict[str, Any]], output_path: Path, metric: str, ylabel: str, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    method_lookup = {method["label"]: method for method in METHODS}
    for ax, dataset in zip(axes, DATASETS):
        dataset_rows = [row for row in rows if row["dataset"] == dataset["name"]]
        for row in dataset_rows:
            spec = method_lookup[row["method"]]
            ax.scatter(row["avg_b"], row[metric], marker=spec["marker"], s=90)
            ax.annotate(
                row["method"],
                (row["avg_b"], row[metric]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ordered = sorted(dataset_rows, key=lambda item: item["avg_b"])
        ax.plot([row["avg_b"] for row in ordered], [row[metric] for row in ordered], linestyle="--", alpha=0.4)
        ax.set_title(dataset["name"])
        ax.set_xlabel("Average graph evidence slots (AvgB)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel(ylabel)
    title = "Reader-Aware Correction Budget Pareto"
    if title_suffix:
        title += f" {title_suffix}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
