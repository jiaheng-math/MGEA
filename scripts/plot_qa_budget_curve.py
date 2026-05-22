from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_generations import extract_gold_answers, extract_prediction, score_prediction_against_gold_answers


DEFAULT_ROUTER_METHODS = "query_only:Query-only,probe_only:Probe-only,query_plus_probe:Combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot answer-level EM/F1 budget curves by reusing dense/graph generations. "
            "No LLM calls are made."
        )
    )
    parser.add_argument("--generations", required=True, help="main_table_generations_top5_*.jsonl.")
    parser.add_argument("--oof-predictions", required=True, help="OOF router probabilities JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and PNG outputs.")
    parser.add_argument("--dataset-name", default="", help="Name used in plot titles.")
    parser.add_argument(
        "--router-methods",
        default=DEFAULT_ROUTER_METHODS,
        help="Comma-separated source:label specs, e.g. query_only:Query-only.",
    )
    parser.add_argument(
        "--budget-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="Comma-separated graph invocation rates to evaluate.",
    )
    parser.add_argument("--tie-policy", choices=["dense", "graph"], default="dense")
    parser.add_argument("--dense-method", default="dense_only")
    parser.add_argument("--graph-method", default="graph_only")
    parser.add_argument("--fusion-method", default="dense_graph_rrf")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_rows = {str(row["id"]): row for row in load_jsonl(Path(args.generations))}
    oof_rows = {str(row["id"]): row for row in load_jsonl(Path(args.oof_predictions))}
    sample_ids = list(generation_rows)
    if not sample_ids:
        raise ValueError("No generation rows found.")

    router_specs = parse_router_specs(args.router_methods)
    budgets = parse_budget_grid(args.budget_grid)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = evaluate_fixed_methods(
        sample_ids=sample_ids,
        generation_rows=generation_rows,
        dense_method=args.dense_method,
        graph_method=args.graph_method,
        fusion_method=args.fusion_method,
    )
    router_curves = {
        spec["label"]: [
            evaluate_router_budget(
                sample_ids=sample_ids,
                generation_rows=generation_rows,
                oof_rows=oof_rows,
                source_method=spec["source"],
                graph_rate=budget,
                tie_policy=args.tie_policy,
                dense_method=args.dense_method,
                graph_method=args.graph_method,
            )
            for budget in budgets
        ]
        for spec in router_specs
    }
    random_curve = [
        evaluate_random_budget(
            sample_ids=sample_ids,
            generation_rows=generation_rows,
            graph_rate=budget,
            dense_method=args.dense_method,
            graph_method=args.graph_method,
            trials=args.random_trials,
            seed=args.random_seed,
        )
        for budget in budgets
    ]

    payload = {
        "generations": args.generations,
        "oof_predictions": args.oof_predictions,
        "num_samples": len(sample_ids),
        "tie_policy": args.tie_policy,
        "budget_grid": budgets,
        "fixed_methods": fixed,
        "router_curves": router_curves,
        "random_router": random_curve,
        "random_trials": args.random_trials,
        "random_seed": args.random_seed,
    }
    json_path = output_dir / "qa_budget_curve.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    render_metric_plot(
        output_path=output_dir / "qa_budget_curve_em.png",
        metric_name="exact_match",
        ylabel="Exact Match",
        dataset_name=args.dataset_name,
        fixed=fixed,
        router_curves=router_curves,
        random_curve=random_curve,
    )
    render_metric_plot(
        output_path=output_dir / "qa_budget_curve_f1.png",
        metric_name="f1",
        ylabel="F1",
        dataset_name=args.dataset_name,
        fixed=fixed,
        router_curves=router_curves,
        random_curve=random_curve,
    )
    print(json.dumps({"output_dir": str(output_dir), "json": str(json_path)}, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_router_specs(value: str) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            source, label = item.split(":", 1)
        else:
            source = item
            label = item
        specs.append({"source": source.strip(), "label": label.strip()})
    if not specs:
        raise ValueError("At least one router method is required.")
    return specs


def parse_budget_grid(value: str) -> list[float]:
    budgets = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    for budget in budgets:
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"Budget must be in [0, 1]. Got {budget}.")
    return budgets


def evaluate_fixed_methods(
    *,
    sample_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    dense_method: str,
    graph_method: str,
    fusion_method: str,
) -> dict[str, dict[str, Any]]:
    return {
        "Dense-only": summarize_scores(
            score_method(sample_ids, generation_rows, dense_method),
            graph_invocation_rate=0.0,
        ),
        "Graph-only": summarize_scores(
            score_method(sample_ids, generation_rows, graph_method),
            graph_invocation_rate=1.0,
        ),
        "RRF": summarize_scores(
            score_method(sample_ids, generation_rows, fusion_method),
            graph_invocation_rate=1.0,
        ),
    }


def score_method(
    sample_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    method: str,
) -> list[dict[str, float]]:
    return [
        score_answer(generation_row=generation_rows[sample_id], method=method)
        for sample_id in sample_ids
    ]


def evaluate_router_budget(
    *,
    sample_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    oof_rows: dict[str, dict[str, Any]],
    source_method: str,
    graph_rate: float,
    tie_policy: str,
    dense_method: str,
    graph_method: str,
) -> dict[str, Any]:
    target_count = round(len(sample_ids) * graph_rate)
    selected_ids = select_top_probability_ids(
        sample_ids=sample_ids,
        oof_rows=oof_rows,
        source_method=source_method,
        target_count=target_count,
        tie_policy=tie_policy,
    )
    scores = [
        score_answer(
            generation_row=generation_rows[sample_id],
            method=graph_method if sample_id in selected_ids else dense_method,
        )
        for sample_id in sample_ids
    ]
    return summarize_scores(scores, graph_invocation_rate=target_count / len(sample_ids))


def select_top_probability_ids(
    *,
    sample_ids: list[str],
    oof_rows: dict[str, dict[str, Any]],
    source_method: str,
    target_count: int,
    tie_policy: str,
) -> set[str]:
    probability_key = f"{source_method}_probability"
    scored: list[tuple[float, str]] = []
    for sample_id in sample_ids:
        oof_row = oof_rows.get(sample_id)
        if oof_row is None or probability_key not in oof_row:
            score = float("inf") if tie_policy == "graph" else float("-inf")
        else:
            score = float(oof_row[probability_key])
        scored.append((score, sample_id))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    return {sample_id for _, sample_id in ranked[:target_count]}


def evaluate_random_budget(
    *,
    sample_ids: list[str],
    generation_rows: dict[str, dict[str, Any]],
    graph_rate: float,
    dense_method: str,
    graph_method: str,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    target_count = round(len(sample_ids) * graph_rate)
    rng = random.Random(seed)
    summaries = []
    for _ in range(trials):
        selected_ids = set(rng.sample(sample_ids, target_count))
        scores = [
            score_answer(
                generation_row=generation_rows[sample_id],
                method=graph_method if sample_id in selected_ids else dense_method,
            )
            for sample_id in sample_ids
        ]
        summaries.append(summarize_scores(scores, graph_invocation_rate=target_count / len(sample_ids)))
    output: dict[str, Any] = {
        "graph_invocation_rate": target_count / len(sample_ids),
        "target_graph_count": target_count,
        "num_trials": trials,
    }
    for metric_name in ["exact_match", "f1", "precision", "recall", "error_rate"]:
        values = [float(summary[metric_name]) for summary in summaries]
        output[metric_name] = mean(values)
        output[f"{metric_name}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return output


def score_answer(generation_row: dict[str, Any], method: str) -> dict[str, float]:
    payload = generation_row.get("methods", {}).get(method, {})
    prediction = extract_prediction(payload.get("answer", ""))
    scores = score_prediction_against_gold_answers(
        prediction=prediction,
        gold_answers=extract_gold_answers(generation_row),
    )
    return {
        "exact_match": float(scores["exact_match"]),
        "f1": float(scores["f1"]),
        "precision": float(scores["precision"]),
        "recall": float(scores["recall"]),
        "error_rate": float(bool(payload.get("error"))),
    }


def summarize_scores(scores: list[dict[str, float]], graph_invocation_rate: float) -> dict[str, Any]:
    return {
        "graph_invocation_rate": float(graph_invocation_rate),
        "target_graph_count": round(len(scores) * graph_invocation_rate),
        "exact_match": mean(score["exact_match"] for score in scores),
        "f1": mean(score["f1"] for score in scores),
        "precision": mean(score["precision"] for score in scores),
        "recall": mean(score["recall"] for score in scores),
        "error_rate": mean(score["error_rate"] for score in scores),
    }


def render_metric_plot(
    *,
    output_path: Path,
    metric_name: str,
    ylabel: str,
    dataset_name: str,
    fixed: dict[str, dict[str, Any]],
    router_curves: dict[str, list[dict[str, Any]]],
    random_curve: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.5, 6.2))
    styles = {
        "Query-only": {"color": "#1f77b4", "marker": "o"},
        "Probe-only": {"color": "#ff7f0e", "marker": "o"},
        "Combined": {"color": "#2ca02c", "marker": "o"},
    }
    for label, points in router_curves.items():
        points = sorted(points, key=lambda item: float(item["graph_invocation_rate"]))
        style = styles.get(label, {"marker": "o"})
        plt.plot(
            [point["graph_invocation_rate"] for point in points],
            [point[metric_name] for point in points],
            linewidth=1.8,
            markersize=3.5,
            label=label,
            **style,
        )
    random_curve = sorted(random_curve, key=lambda item: float(item["graph_invocation_rate"]))
    plt.plot(
        [point["graph_invocation_rate"] for point in random_curve],
        [point[metric_name] for point in random_curve],
        linestyle="--",
        linewidth=1.4,
        color="#7f7f7f",
        label="Random",
    )

    baseline_styles = {
        "Dense-only": ("D", "#9467bd"),
        "Graph-only": ("s", "#d62728"),
        "RRF": ("^", "#8c564b"),
    }
    for label, payload in fixed.items():
        marker, color = baseline_styles[label]
        plt.scatter(
            [payload["graph_invocation_rate"]],
            [payload[metric_name]],
            s=70,
            marker=marker,
            color=color,
            label=label,
            zorder=5,
        )

    title_prefix = f"{dataset_name} " if dataset_name else ""
    plt.xlabel("Graph invocation rate")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}QA Budget Curve ({ylabel})")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
