from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_yaml


PRIORITY_FEATURES = [
    "dense_top1_top2_gap",
    "dense_entity_coverage_ratio",
    "dense_topk_score_std",
]

PAIR_BY_SCENARIO = {
    "dense_swap": ("colbert_hipporag", "bge_large_hipporag"),
    "graph_swap": ("colbert_hipporag", "colbert_lightrag"),
    "full_swap": ("colbert_hipporag", "bge_large_lightrag"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize backend robustness outputs into paper-ready tables.")
    parser.add_argument("--config", required=True, help="Path to backend robustness YAML config.")
    parser.add_argument(
        "--analysis-dir",
        default="",
        help="Optional explicit analyze_backend_robustness output directory. Defaults to config output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    project_root = Path(args.config).resolve().parent.parent
    analysis_dir = resolve_path(project_root, args.analysis_dir) if args.analysis_dir else resolve_path(project_root, str(config["output_dir"]))
    output_dir = ensure_dir(analysis_dir / "summary")

    transfer_rows = read_csv(analysis_dir / "backend_transfer_matrix.csv")
    correlation_rows = read_csv(analysis_dir / "probe_feature_correlation.csv")
    consistency_rows = read_csv(analysis_dir / "routing_consistency_summary.csv")
    validation_payload = try_load_json(analysis_dir / "validation" / "artifact_validation.json")

    datasets = [str(dataset_cfg["name"]) for dataset_cfg in config.get("datasets", [])]
    transfer_table: list[dict[str, Any]] = []
    correlation_table: list[dict[str, Any]] = []
    consistency_table: list[dict[str, Any]] = []
    markdown_lines: list[str] = [
        "# Backend Robustness Summary",
        "",
        "This file is generated from validated backend-pair routing artifacts.",
        "",
    ]

    if validation_payload:
        markdown_lines.extend(
            [
                "## Validation",
                "",
                f"- Validation errors: {int(validation_payload.get('num_errors', 0))}",
                f"- Exact shared query ids required: {bool(validation_payload.get('require_exact_shared', False))}",
                "",
            ]
        )

    for dataset in datasets:
        dataset_transfer_rows = [row for row in transfer_rows if row.get("dataset") == dataset and row.get("method") == "query_plus_probe"]
        dataset_correlation_rows = [row for row in correlation_rows if row.get("dataset") == dataset]
        dataset_consistency_rows = [row for row in consistency_rows if row.get("dataset") == dataset]

        scenario_rows = build_transfer_scenarios(dataset, dataset_transfer_rows)
        transfer_table.extend(scenario_rows)

        correlation_summary_rows = summarize_probe_correlations(dataset, dataset_correlation_rows)
        correlation_table.extend(correlation_summary_rows)

        consistency_summary_rows = summarize_consistency(dataset, dataset_consistency_rows)
        consistency_table.extend(consistency_summary_rows)

        markdown_lines.extend(render_dataset_markdown(dataset, scenario_rows, correlation_summary_rows, consistency_summary_rows))

    write_csv(output_dir / "transfer_scenarios.csv", transfer_table)
    write_csv(output_dir / "probe_correlation_summary.csv", correlation_table)
    write_csv(output_dir / "routing_consistency_paper.csv", consistency_table)
    (output_dir / "backend_robustness_summary.md").write_text("\n".join(markdown_lines).rstrip() + "\n", encoding="utf-8")

    payload = {
        "analysis_dir": str(analysis_dir),
        "summary_dir": str(output_dir),
        "datasets": datasets,
        "num_transfer_rows": len(transfer_table),
        "num_correlation_rows": len(correlation_table),
        "num_consistency_rows": len(consistency_table),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_transfer_scenarios(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["train_pair"], row["test_pair"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for scenario, (train_pair, test_pair) in PAIR_BY_SCENARIO.items():
        transfer = by_key.get((train_pair, test_pair))
        test_self = by_key.get((test_pair, test_pair))
        train_self = by_key.get((train_pair, train_pair))
        if not transfer:
            output.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "train_pair": train_pair,
                    "test_pair": test_pair,
                    "available": False,
                }
            )
            continue
        row = {
            "dataset": dataset,
            "scenario": scenario,
            "train_pair": train_pair,
            "test_pair": test_pair,
            "available": True,
            "overlap_queries": maybe_float(transfer.get("overlap_queries")),
            "transfer_auc": maybe_float(transfer.get("auc")),
            "transfer_recall@3": maybe_float(transfer.get("recall@3")),
            "transfer_recall@5": maybe_float(transfer.get("recall@5")),
            "transfer_graph_invocation_rate": maybe_float(transfer.get("graph_invocation_rate")),
            "test_self_recall@3": maybe_float(test_self.get("recall@3")) if test_self else None,
            "test_self_recall@5": maybe_float(test_self.get("recall@5")) if test_self else None,
            "test_self_auc": maybe_float(test_self.get("auc")) if test_self else None,
            "train_self_recall@5": maybe_float(train_self.get("recall@5")) if train_self else None,
        }
        if test_self:
            row["delta_to_test_self_recall@3"] = safe_delta(row["transfer_recall@3"], row["test_self_recall@3"])
            row["delta_to_test_self_recall@5"] = safe_delta(row["transfer_recall@5"], row["test_self_recall@5"])
            row["delta_to_test_self_auc"] = safe_delta(row["transfer_auc"], row["test_self_auc"])
        output.append(row)
    return output


def summarize_probe_correlations(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in PRIORITY_FEATURES:
        matched = [row for row in rows if row.get("feature") == feature]
        if not matched:
            output.append({"dataset": dataset, "feature": feature, "available": False})
            continue
        spearman_values = [float(row["spearman_rho"]) for row in matched if is_number(row.get("spearman_rho"))]
        output.append(
            {
                "dataset": dataset,
                "feature": feature,
                "available": bool(spearman_values),
                "num_pairs": len(matched),
                "mean_spearman_rho": safe_mean(spearman_values),
                "min_spearman_rho": min(spearman_values) if spearman_values else None,
                "max_spearman_rho": max(spearman_values) if spearman_values else None,
            }
        )
    return output


def summarize_consistency(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        pair = str(row["pair"])
        entry = {
            "dataset": dataset,
            "pair": pair,
            "dense_backend": row.get("dense_backend"),
            "graph_backend": row.get("graph_backend"),
            "combined_graph_invocation_rate": maybe_float(row.get("combined_graph_invocation_rate")),
            "combined_auc": maybe_float(row.get("combined_auc")),
            "combined_recall@3": maybe_float(row.get("combined_recall@3")),
            "combined_recall@5": maybe_float(row.get("combined_recall@5")),
            "matched_random_recall@3": maybe_float(row.get("matched_random_recall@3")),
            "matched_random_recall@5": maybe_float(row.get("matched_random_recall@5")),
            "combined_minus_random_recall@3": maybe_float(row.get("combined_minus_random_recall@3")),
            "combined_minus_random_recall@5": maybe_float(row.get("combined_minus_random_recall@5")),
        }
        output.append(entry)
    return output


def render_dataset_markdown(
    dataset: str,
    transfer_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> list[str]:
    lines = [f"## {dataset}", ""]
    if dataset == "nq":
        lines.append("NQ is summarized as a light-weight easy-workload sanity check.")
        lines.append("")

    lines.extend(["### Transfer", ""])
    for row in transfer_rows:
        if not row.get("available"):
            lines.append(f"- {row['scenario']}: missing")
            continue
        delta_r5 = row.get("delta_to_test_self_recall@5")
        delta_auc = row.get("delta_to_test_self_auc")
        lines.append(
            "- "
            f"{row['scenario']}: transfer R@5={fmt(row.get('transfer_recall@5'))}, "
            f"test-self R@5={fmt(row.get('test_self_recall@5'))}, "
            f"delta={fmt_signed(delta_r5)}; "
            f"AUC delta={fmt_signed(delta_auc)}"
        )
    lines.append("")

    if dataset != "nq":
        lines.extend(["### Probe Correlation", ""])
        for row in correlation_rows:
            if not row.get("available"):
                lines.append(f"- {row['feature']}: missing")
                continue
            lines.append(
                f"- {row['feature']}: mean Spearman={fmt(row.get('mean_spearman_rho'))} "
                f"(min={fmt(row.get('min_spearman_rho'))}, max={fmt(row.get('max_spearman_rho'))})"
            )
        lines.append("")

    lines.extend(["### Consistency", ""])
    for row in consistency_rows:
        lines.append(
            "- "
            f"{row['pair']}: graph%={fmt_pct(row.get('combined_graph_invocation_rate'))}, "
            f"R@5={fmt(row.get('combined_recall@5'))}, "
            f"vs matched random={fmt_signed(row.get('combined_minus_random_recall@5'))}"
        )
    lines.append("")
    return lines


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def try_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if value_f != value_f:
        return None
    return value_f


def is_number(value: Any) -> bool:
    return maybe_float(value) is not None


def safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def fmt(value: Any) -> str:
    value_f = maybe_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f:.4f}"


def fmt_signed(value: Any) -> str:
    value_f = maybe_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f:+.4f}"


def fmt_pct(value: Any) -> str:
    value_f = maybe_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f * 100.0:.1f}%"


if __name__ == "__main__":
    main()
