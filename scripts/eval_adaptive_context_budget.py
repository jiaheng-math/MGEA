"""Evaluate adaptive graph context budgeting over deep retrieval reservoirs.

The first contribution makes graph usage conditional. This script tests a
second, complementary idea: reallocate the saved evidence budget. Instead of
forcing every query to use graph top-5, learn which queries should receive a
larger graph context and which can be compressed to a smaller context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - cloud env dependency.
    raise SystemExit("Missing scikit-learn in this environment.") from exc


DATASET_MAP = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki": "results/study_2wiki_hipporag_colbert_500",
    "nq": "results/study_nq_hipporag_colbert_500",
}

CONTENT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were",
    "are", "his", "her", "its", "into", "than", "then", "who", "what",
    "when", "where", "which",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive graph context budgets.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--deep-retrieval-files", nargs="+", required=True)
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--small-k", type=int, default=3)
    parser.add_argument("--large-k", type=int, nargs="+", default=[8, 10, 15, 20])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument(
        "--target-avg-k",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional budget-calibrated policies. For each target average K, expand only the "
            "highest-risk queries needed to meet the target budget."
        ),
    )
    parser.add_argument(
        "--disable-random-budget-baseline",
        action="store_true",
        help="Do not add random query expansion baselines at the same target average K.",
    )
    parser.add_argument(
        "--slot-target-avg-k",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional marginal slot policies. For each target average K, select individual "
            "tail passages from graph ranks base_k+1..slot_max_k under a global slot budget."
        ),
    )
    parser.add_argument(
        "--slot-max-k",
        type=int,
        default=20,
        help="Maximum graph rank used as the marginal slot reservoir.",
    )
    parser.add_argument(
        "--slot-per-query-cap",
        type=int,
        default=5,
        help="Maximum number of learned/random/score-selected tail slots added to one query.",
    )
    parser.add_argument(
        "--slot-feature-variants",
        nargs="*",
        default=["full", "no_probe", "probe_only"],
        help=(
            "Feature groups for learned marginal slot models. Supported variants: "
            "full (current method), no_probe (drop shared dense probe features only), "
            "no_novelty (drop slot novelty/query-overlap features), "
            "probe_only (shared dense probe features without slot-local features), "
            "passage_rerank (query-candidate reranker without graph_top5 context features). "
            "text_rerank (clean supervised reranker using only question/candidate text). "
            "Additional diagnostic variants are available but not recommended for the "
            "paper-facing clean ablation."
        ),
    )
    parser.add_argument(
        "--disable-random-slot-baseline",
        action="store_true",
        help="Do not add random marginal slot baselines at the same target average K.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", default="results/adaptive_context_budget_eval.json")
    parser.add_argument("--per-sample-output", default="results/adaptive_context_budget_per_sample.jsonl")
    parser.add_argument(
        "--generation-input-output",
        default=None,
        help="Optional JSONL generation input for batch_generate_from_retrieval.py.",
    )
    parser.add_argument(
        "--generation-methods",
        nargs="*",
        default=[
            "graph_top5",
            "graph_top8",
            "graph_top10",
            "budget_graph_5_to_8_avg6",
            "budget_graph_5_to_10_avg7",
            "random_budget_graph_5_to_10_avg7",
            "slot_graph_5_to_20_cap5_avg7",
            "score_slot_graph_5_to_20_cap5_avg7",
            "random_slot_graph_5_to_20_cap5_avg7",
        ],
        help="Methods to include in generation input. Use 'auto' to include all methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dirs = resolve_dataset_dirs(args)
    if len(args.deep_retrieval_files) != len(dataset_dirs):
        raise ValueError("--deep-retrieval-files must have the same length/order as --datasets/--result-dirs.")

    reports = []
    per_sample_all: list[dict[str, Any]] = []
    generation_rows_all: list[dict[str, Any]] = []
    for (dataset, result_dir), deep_path in zip(dataset_dirs, args.deep_retrieval_files):
        report, per_sample, generation_rows = evaluate_dataset(dataset, result_dir, resolve_project_path(deep_path), args)
        reports.append(report)
        per_sample_all.extend(per_sample)
        generation_rows_all.extend(generation_rows)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"reports": reports}, indent=2, ensure_ascii=False), encoding="utf-8")

    per_sample_output = resolve_project_path(args.per_sample_output)
    per_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_output.open("w", encoding="utf-8") as handle:
        for row in per_sample_all:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.generation_input_output:
        generation_output = resolve_project_path(args.generation_input_output)
        generation_output.parent.mkdir(parents=True, exist_ok=True)
        with generation_output.open("w", encoding="utf-8") as handle:
            for row in generation_rows_all:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(reports)
    print(f"\nFull report: {output}")
    print(f"Per-sample report: {per_sample_output}")
    if args.generation_input_output:
        print(f"Generation input: {resolve_project_path(args.generation_input_output)}")


def evaluate_dataset(
    dataset: str,
    result_dir: Path,
    deep_retrieval_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    if args.max_queries:
        rows = rows[: args.max_queries]
    retrieval_by_id = {str(row["id"]): row for row in load_jsonl(deep_retrieval_path)}
    examples = [build_example(dataset, row, retrieval_by_id[str(row["id"])], args) for row in rows]

    expansion_probs: dict[tuple[int, int], dict[str, float]] = {}
    expansion_metrics = {}
    for large_k in args.large_k:
        for start_k in sorted({args.small_k, args.base_k}):
            if start_k >= large_k:
                continue
            probs, metrics = train_oof_expansion(examples, start_k, large_k, args)
            expansion_probs[(start_k, large_k)] = probs
            expansion_metrics[f"{start_k}_to_{large_k}"] = metrics
    budget_plans, budget_plan_metrics = build_budget_plans(examples, expansion_probs, args)
    slot_scores, slot_model_metrics = train_oof_slots(examples, args)
    slot_plans, slot_plan_metrics = build_slot_plans(examples, slot_scores, args)
    if args.slot_target_avg_k and args.slot_max_k > args.base_k:
        graph_score_records = slot_records(examples, {}, args)
        if graph_score_records:
            slot_model_metrics.setdefault("graph_score", score_slot_model_metrics(graph_score_records))

    aggs = make_aggs()
    per_sample: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for example in examples:
        rankings = {
            f"dense_top{args.base_k}": example["dense_ids"][: args.base_k],
            f"graph_top{args.small_k}": example["graph_ids"][: args.small_k],
            f"graph_top{args.base_k}": example["graph_ids"][: args.base_k],
        }
        for large_k in args.large_k:
            rankings[f"graph_top{large_k}"] = example["graph_ids"][:large_k]
            rankings[f"oracle_graph_{args.small_k}_to_{large_k}"] = choose_better_graph_budget(
                example, args.small_k, large_k
            )
            rankings[f"oracle_graph_{args.base_k}_to_{large_k}"] = choose_better_graph_budget(
                example, args.base_k, large_k
            )
            for threshold in args.thresholds:
                suffix = format_threshold(threshold)
                use_large = expansion_probs[(args.small_k, large_k)][example["id"]] >= threshold
                rankings[f"adaptive_graph_{args.small_k}_to_{large_k}_t{suffix}"] = (
                    example["graph_ids"][:large_k] if use_large else example["graph_ids"][: args.small_k]
                )
                use_large = expansion_probs[(args.base_k, large_k)][example["id"]] >= threshold
                rankings[f"adaptive_graph_{args.base_k}_to_{large_k}_t{suffix}"] = (
                    example["graph_ids"][:large_k] if use_large else example["graph_ids"][: args.base_k]
                )
            for target_avg_k in args.target_avg_k:
                for start_k in (args.small_k, args.base_k):
                    for method in budget_methods_for(start_k, large_k, target_avg_k, args):
                        if method not in budget_plans:
                            continue
                        rankings[method] = (
                            example["graph_ids"][:large_k]
                            if example["id"] in budget_plans[method]
                            else example["graph_ids"][:start_k]
                        )
        for method, selected_by_qid in slot_plans.items():
            selected = selected_by_qid.get(example["id"], [])
            rankings[method] = merge_graph_slots(example, selected, args)

        scores = {}
        for method, ids in rankings.items():
            score = score_ranking(ids, example["gold"], set(example["dense_ids"][: args.base_k]))
            scores[method] = score
            update_agg(aggs[method]["ALL"], score, len(ids))
            if example["gold"] - set(example["dense_ids"][: args.base_k]):
                update_agg(aggs[method]["dense_miss"], score, len(ids))

        per_sample.append(
            {
                "dataset": dataset,
                "id": example["id"],
                "dense_miss": bool(example["gold"] - set(example["dense_ids"][: args.base_k])),
                "expansion_probs": {
                    f"{start_k}_to_{large_k}": probs[example["id"]]
                    for (start_k, large_k), probs in expansion_probs.items()
                },
                "scores": scores,
            }
        )
        if args.generation_input_output:
            generation_rows.append(build_generation_row(example, rankings, args))

    return {
        "dataset": dataset,
        "n": len(examples),
        "base_k": args.base_k,
        "small_k": args.small_k,
        "large_k": args.large_k,
        "target_avg_k": args.target_avg_k,
        "candidate_source": str(deep_retrieval_path),
        "expansion_metrics": expansion_metrics,
        "budget_plan_metrics": budget_plan_metrics,
        "slot_model_metrics": slot_model_metrics,
        "slot_plan_metrics": slot_plan_metrics,
        "metrics": summarize_aggs(aggs),
    }, per_sample, generation_rows


def train_oof_expansion(
    examples: list[dict[str, Any]],
    start_k: int,
    large_k: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    labels = [1 if graph_budget_gain(example, start_k, large_k) > 0 else 0 for example in examples]
    n_splits = min(args.num_folds, min(Counter(labels).values())) if len(set(labels)) > 1 else args.num_folds
    n_splits = max(2, min(args.num_folds, n_splits))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
    probs_by_id: dict[str, float] = {}
    y_true_all: list[int] = []
    y_score_all: list[float] = []

    x_all = np.asarray([budget_features(example, start_k, large_k, args) for example in examples], dtype=np.float64)
    y_all = np.asarray(labels, dtype=np.int64)
    for train_idx, test_idx in splitter.split(x_all, y_all):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        if len(set(y_train.tolist())) < 2:
            clf = DummyClassifier(strategy="prior")
        else:
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=args.random_seed,
                        ),
                    ),
                ]
            )
        clf.fit(x_train, y_train)
        scores = positive_scores(clf, x_all[test_idx])
        for idx, score in zip(test_idx, scores.tolist()):
            probs_by_id[examples[idx]["id"]] = float(score)
            y_true_all.append(int(y_all[idx]))
            y_score_all.append(float(score))

    metrics = {
        "start_k": start_k,
        "large_k": large_k,
        "positive_rate": round(float(np.mean(labels)), 4),
        "positive_n": int(sum(labels)),
        "n_splits": n_splits,
    }
    if len(set(y_true_all)) > 1:
        metrics["auc"] = round(float(roc_auc_score(y_true_all, y_score_all)), 4)
        metrics["ap"] = round(float(average_precision_score(y_true_all, y_score_all)), 4)
    return probs_by_id, metrics


def build_budget_plans(
    examples: list[dict[str, Any]],
    expansion_probs: dict[tuple[int, int], dict[str, float]],
    args: argparse.Namespace,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    plans: dict[str, set[str]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for large_k in args.large_k:
        for target_avg_k in args.target_avg_k:
            for start_k in (args.small_k, args.base_k):
                if not (start_k <= target_avg_k <= large_k):
                    continue
                probs = expansion_probs.get((start_k, large_k))
                if probs is None:
                    continue
                method = budget_method_name(start_k, large_k, target_avg_k)
                expand_count = int(round(len(examples) * (target_avg_k - start_k) / max(1, large_k - start_k)))
                expand_count = max(0, min(len(examples), expand_count))
                ranked = sorted(
                    examples,
                    key=lambda example: (-probs[example["id"]], example["id"]),
                )
                selected = ranked[:expand_count]
                selected_ids = {example["id"] for example in selected}
                plans[method] = selected_ids
                all_gains = [graph_budget_gain(example, start_k, large_k) for example in examples]
                metrics[method] = budget_plan_stats(
                    examples=examples,
                    selected=selected,
                    all_gains=all_gains,
                    start_k=start_k,
                    large_k=large_k,
                    target_avg_k=target_avg_k,
                    expand_count=expand_count,
                    selection="learned",
                )
                if not args.disable_random_budget_baseline:
                    random_method = random_budget_method_name(start_k, large_k, target_avg_k)
                    rng = np.random.default_rng(stable_seed(args.random_seed, random_method))
                    random_indices = rng.permutation(len(examples))[:expand_count].tolist()
                    random_selected = [examples[idx] for idx in random_indices]
                    plans[random_method] = {example["id"] for example in random_selected}
                    metrics[random_method] = budget_plan_stats(
                        examples=examples,
                        selected=random_selected,
                        all_gains=all_gains,
                        start_k=start_k,
                        large_k=large_k,
                        target_avg_k=target_avg_k,
                        expand_count=expand_count,
                        selection="random",
                    )
    return plans, metrics


def train_oof_slots(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    if not args.slot_target_avg_k or args.slot_max_k <= args.base_k:
        return {}, {}

    candidate_counts = [len(slot_candidate_ids(example, args)) for example in examples]
    eval_query_labels = [
        1 if any(slot_label(example, pid, args) for pid in slot_candidate_ids(example, args)) else 0
        for example in examples
    ]
    if not examples or sum(candidate_counts) == 0:
        return {}, {"enabled": False, "reason": "no tail candidates"}

    variants = normalize_slot_feature_variants(args.slot_feature_variants)
    scores_by_variant: dict[str, dict[str, dict[str, float]]] = {}
    metrics_by_variant: dict[str, Any] = {}

    for variant in variants:
        train_query_labels = [
            1 if any(
                slot_training_label(example, pid, args, variant)
                for pid in slot_training_candidate_ids(example, args, variant)
            ) else 0
            for example in examples
        ]
        if len(set(train_query_labels)) > 1 and min(Counter(train_query_labels).values()) >= 2:
            n_splits = min(args.num_folds, min(Counter(train_query_labels).values()))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
            split_iter = splitter.split(np.zeros(len(examples)), train_query_labels)
        else:
            n_splits = max(2, min(args.num_folds, len(examples)))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
            split_iter = splitter.split(np.zeros(len(examples)))
        folds = list(split_iter)

        scores_by_qid: dict[str, dict[str, float]] = {}
        y_true_all: list[int] = []
        y_score_all: list[float] = []

        for train_q_idx, test_q_idx in folds:
            x_train, y_train = flatten_slot_candidates([examples[i] for i in train_q_idx], args, variant)
            if len(y_train) == 0:
                continue
            if len(set(y_train.tolist())) < 2:
                clf = DummyClassifier(strategy="prior")
            else:
                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=args.random_seed,
                            ),
                        ),
                    ]
                )
            clf.fit(x_train, y_train)
            for idx in test_q_idx:
                example = examples[idx]
                candidates = slot_candidate_ids(example, args)
                if not candidates:
                    scores_by_qid[example["id"]] = {}
                    continue
                x_test = np.asarray(
                    [slot_features(example, pid, args, variant=variant) for pid in candidates],
                    dtype=np.float64,
                )
                scores = positive_scores(clf, x_test).tolist()
                scores_by_qid[example["id"]] = {pid: float(score) for pid, score in zip(candidates, scores)}
                for pid, score in zip(candidates, scores):
                    # Always evaluate slot AUC against the conditional marginal
                    # label, even when the baseline was trained as a standard
                    # self-contained passage reranker.
                    y_true_all.append(slot_label(example, pid, args))
                    y_score_all.append(float(score))

        metrics = {
            "feature_variant": variant,
            "n_splits": n_splits,
            "slot_max_k": args.slot_max_k,
            "slot_per_query_cap": args.slot_per_query_cap,
            "candidate_n": int(sum(candidate_counts)),
            "candidate_positive_n": int(sum(y_true_all)),
            "candidate_positive_rate": round(float(np.mean(y_true_all)), 4) if y_true_all else None,
            "query_positive_rate": round(float(np.mean(eval_query_labels)), 4) if eval_query_labels else None,
            "training_label": slot_training_label_name(variant),
            "eval_label": "conditional_marginal_slot",
            "training_query_positive_rate": round(float(np.mean(train_query_labels)), 4) if train_query_labels else None,
            "feature_dim": slot_feature_dim(examples, args, variant),
        }
        if len(set(y_true_all)) > 1:
            metrics["candidate_auc"] = round(float(roc_auc_score(y_true_all, y_score_all)), 4)
            metrics["candidate_ap"] = round(float(average_precision_score(y_true_all, y_score_all)), 4)
        scores_by_variant[variant] = scores_by_qid
        metrics_by_variant[variant] = metrics

    return scores_by_variant, metrics_by_variant


def flatten_slot_candidates(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for example in examples:
        for pid in slot_training_candidate_ids(example, args, variant):
            x_rows.append(slot_features(example, pid, args, variant=variant))
            y_rows.append(slot_training_label(example, pid, args, variant))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def build_slot_plans(
    examples: list[dict[str, Any]],
    slot_scores: dict[str, dict[str, dict[str, float]]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, Any]]]:
    if not args.slot_target_avg_k or args.slot_max_k <= args.base_k:
        return {}, {}

    variants = normalize_slot_feature_variants(args.slot_feature_variants)
    records_by_variant = {
        variant: slot_records(examples, slot_scores.get(variant, {}), args)
        for variant in variants
    }
    records = records_by_variant[variants[0]] if variants else []
    plans: dict[str, dict[str, list[str]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    max_slots = max(0, args.slot_per_query_cap) * len(examples)

    for target_avg_k in args.slot_target_avg_k:
        if target_avg_k < args.base_k:
            continue
        total_slots = int(round(len(examples) * (target_avg_k - args.base_k)))
        total_slots = max(0, min(total_slots, max_slots, len(records)))
        if total_slots == 0:
            continue

        for variant in variants:
            variant_records = records_by_variant[variant]
            learned_method = slot_method_name(
                args.base_k,
                args.slot_max_k,
                args.slot_per_query_cap,
                target_avg_k,
                variant=variant,
            )
            learned = select_slot_records(
                sorted(variant_records, key=lambda item: (-item["score"], item["qid"], item["rank"], item["pid"])),
                total_slots,
                args.slot_per_query_cap,
            )
            plans[learned_method] = slot_plan_by_qid(learned)
            metrics[learned_method] = slot_plan_stats(
                variant_records,
                learned,
                examples,
                args,
                target_avg_k,
                total_slots,
                selection=f"learned:{variant}",
            )

        score_method = score_slot_method_name(args.base_k, args.slot_max_k, args.slot_per_query_cap, target_avg_k)
        score_selected = select_slot_records(
            sorted(records, key=lambda item: (-item["graph_score"], item["qid"], item["rank"], item["pid"])),
            total_slots,
            args.slot_per_query_cap,
        )
        plans[score_method] = slot_plan_by_qid(score_selected)
        metrics[score_method] = slot_plan_stats(
            records, score_selected, examples, args, target_avg_k, total_slots, selection="graph_score"
        )

        oracle_method = oracle_slot_method_name(args.base_k, args.slot_max_k, args.slot_per_query_cap, target_avg_k)
        oracle_selected = select_slot_records(
            sorted(records, key=lambda item: (-item["gain"], item["rank"], item["qid"], item["pid"])),
            total_slots,
            args.slot_per_query_cap,
        )
        plans[oracle_method] = slot_plan_by_qid(oracle_selected)
        metrics[oracle_method] = slot_plan_stats(
            records, oracle_selected, examples, args, target_avg_k, total_slots, selection="oracle"
        )

        if not args.disable_random_slot_baseline:
            random_method = random_slot_method_name(args.base_k, args.slot_max_k, args.slot_per_query_cap, target_avg_k)
            rng = np.random.default_rng(stable_seed(args.random_seed, random_method))
            random_order = list(rng.permutation(len(records)))
            random_selected = select_slot_records([records[idx] for idx in random_order], total_slots, args.slot_per_query_cap)
            plans[random_method] = slot_plan_by_qid(random_selected)
            metrics[random_method] = slot_plan_stats(
                records, random_selected, examples, args, target_avg_k, total_slots, selection="random"
            )

    return plans, metrics


def score_slot_model_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(record["label"]) for record in records]
    y_score = [float(record["graph_score"]) for record in records]
    metrics: dict[str, Any] = {
        "feature_variant": "graph_score",
        "candidate_n": len(records),
        "candidate_positive_n": int(sum(y_true)),
        "candidate_positive_rate": round(float(np.mean(y_true)), 4) if y_true else None,
    }
    if len(set(y_true)) > 1:
        metrics["candidate_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        metrics["candidate_ap"] = round(float(average_precision_score(y_true, y_score)), 4)
    return metrics


def slot_records(
    examples: list[dict[str, Any]],
    slot_scores: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        graph_rank = {pid: idx + 1 for idx, pid in enumerate(example["graph_ids"])}
        for pid in slot_candidate_ids(example, args):
            records.append(
                {
                    "qid": example["id"],
                    "pid": pid,
                    "rank": graph_rank.get(pid, 999),
                    "score": float(slot_scores.get(example["id"], {}).get(pid, 0.0)),
                    "graph_score": score_for_pid(example["graph_ids"], example["graph_scores"], pid),
                    "label": slot_label(example, pid, args),
                    "gain": slot_gain(example, pid, args),
                }
            )
    return records


def slot_candidate_ids(example: dict[str, Any], args: argparse.Namespace) -> list[str]:
    base = set(example["graph_ids"][: args.base_k])
    seen = set(base)
    candidates = []
    for pid in example["graph_ids"][args.base_k : args.slot_max_k]:
        if pid in seen:
            continue
        seen.add(pid)
        candidates.append(pid)
    return candidates


def slot_training_candidate_ids(example: dict[str, Any], args: argparse.Namespace, variant: str) -> list[str]:
    variant = normalize_slot_feature_variant(variant)
    if variant != "text_rerank":
        return slot_candidate_ids(example, args)

    # A standard text-only passage reranker is trained on self-contained
    # relevance over the whole graph top-k candidate pool. At inference time we
    # still allocate only tail slots through slot_candidate_ids().
    seen: set[str] = set()
    candidates: list[str] = []
    for pid in example["graph_ids"][: args.slot_max_k]:
        if pid in seen:
            continue
        seen.add(pid)
        candidates.append(pid)
    return candidates


def slot_label(example: dict[str, Any], pid: str, args: argparse.Namespace) -> int:
    return 1 if pid in slot_target_gold(example, args) else 0


def slot_training_label(example: dict[str, Any], pid: str, args: argparse.Namespace, variant: str) -> int:
    variant = normalize_slot_feature_variant(variant)
    if variant == "text_rerank":
        return 1 if pid in set(example["gold"]) else 0
    return slot_label(example, pid, args)


def slot_training_label_name(variant: str) -> str:
    variant = normalize_slot_feature_variant(variant)
    if variant == "text_rerank":
        return "self_contained_relevance"
    return "conditional_marginal_slot"


def slot_gain(example: dict[str, Any], pid: str, args: argparse.Namespace) -> float:
    if pid not in slot_target_gold(example, args):
        return 0.0
    return 1.0 / max(1, len(example["gold"]))


def slot_target_gold(example: dict[str, Any], args: argparse.Namespace) -> set[str]:
    return set(example["gold"]) - set(example["graph_ids"][: args.base_k])


def slot_features(
    example: dict[str, Any],
    pid: str,
    args: argparse.Namespace,
    *,
    variant: str = "full",
) -> list[float]:
    variant = normalize_slot_feature_variant(variant)
    query_context = list(example["features"])
    shared_probe = shared_probe_features(query_context)
    non_probe_context = non_probe_query_context_features(query_context)
    local = slot_local_features(example, pid, args)
    if variant == "full":
        return query_context + local
    if variant == "no_probe":
        return non_probe_context + local
    if variant == "probe_only":
        return shared_probe
    if variant == "slot_only":
        return local
    if variant == "passage_rerank":
        return passage_rerank_features(non_probe_context, local)
    if variant == "text_rerank":
        return text_rerank_features(example, pid)
    if variant == "graph_only":
        return local[:9]
    if variant == "no_dense_support":
        return query_context + local[:9] + local[16:]
    if variant == "no_novelty":
        return query_context + local[:16]
    raise ValueError(f"Unsupported slot feature variant: {variant}")


def shared_probe_features(query_context: list[float]) -> list[float]:
    # Must stay aligned with query_features(): dense score confidence, dense
    # entity coverage, and dense candidate diversity. This is the shared signal
    # used by the router and tested in the slot ablation.
    probe_indices = [2, 3, 4, 8, 9]
    return [query_context[idx] for idx in probe_indices if idx < len(query_context)]


def non_probe_query_context_features(query_context: list[float]) -> list[float]:
    probe_indices = {2, 3, 4, 8, 9}
    return [value for idx, value in enumerate(query_context) if idx not in probe_indices]


def passage_rerank_features(non_probe_context: list[float], local: list[float]) -> list[float]:
    # Standard supervised passage-reranker baseline: query/candidate features
    # only, without conditioning on the already selected graph_top5 prefix.
    indices = [
        0,   # graph rank
        1,   # reciprocal graph rank
        2,   # distance from base_k
        3,   # graph score
        4,   # graph score z
        5,   # previous-local graph score slope
        6,   # next-local graph score slope
        9,   # appears in dense ranking
        11,  # appears in dense top slot_max_k
        12,  # reciprocal dense rank
        13,  # dense rank
        14,  # dense score
        15,  # dense score z
        16,  # candidate passage length
        19,  # query term coverage
        20,  # query-candidate jaccard
    ]
    return non_probe_context + [local[idx] for idx in indices if idx < len(local)]


def text_rerank_features(example: dict[str, Any], pid: str) -> list[float]:
    # Clean supervised passage-reranker baseline: only question and candidate
    # passage text. No graph/dense scores, ranks, or graph_top5 prefix features.
    passage = example.get("passage_lookup", {}).get(pid, {})
    question = str(example.get("row", {}).get("question") or "")
    title = str(passage.get("title") or pid)
    text = str(passage.get("text") or "")

    q_tokens = content_tokens(question)
    title_tokens = content_tokens(title)
    body_tokens = content_tokens(text)
    passage_tokens = title_tokens + body_tokens
    first_100_tokens = passage_tokens[:100]
    first_200_tokens = passage_tokens[:200]

    q_terms = set(q_tokens)
    title_terms = set(title_tokens)
    body_terms = set(body_tokens)
    passage_terms_set = set(passage_tokens)
    first_100_terms = set(first_100_tokens)
    first_200_terms = set(first_200_tokens)

    q_bigrams = ngrams(q_tokens, 2)
    passage_bigrams = ngrams(passage_tokens, 2)
    q_numbers = {token for token in re.findall(r"\d+", question)}
    passage_numbers = {token for token in re.findall(r"\d+", f"{title} {text}")}
    overlap = q_terms & passage_terms_set
    title_overlap = q_terms & title_terms
    body_overlap = q_terms & body_terms

    return [
        float(len(q_tokens)),
        float(len(q_terms)),
        len(passage_tokens) / 100.0,
        len(title_tokens) / 20.0,
        float(len(overlap)),
        len(overlap) / max(1, len(q_terms)),
        len(overlap) / max(1, len(passage_terms_set)),
        jaccard_terms(q_terms, passage_terms_set),
        len(title_overlap) / max(1, len(q_terms)),
        jaccard_terms(q_terms, title_terms),
        len(body_overlap) / max(1, len(q_terms)),
        len(q_bigrams & passage_bigrams) / max(1, len(q_bigrams)),
        len(q_numbers & passage_numbers) / max(1, len(q_numbers)) if q_numbers else 0.0,
        len(q_terms & first_100_terms) / max(1, len(q_terms)),
        len(q_terms & first_200_terms) / max(1, len(q_terms)),
        1.0 if q_terms and q_terms <= passage_terms_set else 0.0,
    ]


def slot_local_features(example: dict[str, Any], pid: str, args: argparse.Namespace) -> list[float]:
    dense_rank = {dense_pid: idx + 1 for idx, dense_pid in enumerate(example["dense_ids"])}
    graph_rank = {graph_pid: idx + 1 for idx, graph_pid in enumerate(example["graph_ids"])}
    rank = graph_rank.get(pid, 999)
    graph_idx = rank - 1 if rank != 999 else 999
    dense_idx = dense_rank[pid] - 1 if pid in dense_rank else 999
    candidate_terms_set = passage_terms(example, pid)
    prefix_terms = set()
    for prefix_pid in example["graph_ids"][: args.base_k]:
        prefix_terms.update(passage_terms(example, prefix_pid))
    query_terms_set = query_terms(example)
    graph_scores = example["graph_scores"]
    dense_scores = example["dense_scores"]
    graph_score = score_at(graph_scores, graph_idx)
    dense_score = score_at(dense_scores, dense_idx)
    graph_z = zscore_at(graph_scores, graph_idx)
    dense_z = zscore_at(dense_scores, dense_idx)
    return [
        float(rank),
        reciprocal_rank(rank),
        float(rank - args.base_k),
        graph_score,
        graph_z,
        score_at(graph_scores, graph_idx - 1) - graph_score if graph_idx > 0 else 0.0,
        graph_score - score_at(graph_scores, graph_idx + 1),
        graph_score - score_at(graph_scores, args.base_k - 1),
        graph_score - (float(np.mean(graph_scores[: args.base_k])) if graph_scores[: args.base_k] else 0.0),
        1.0 if pid in dense_rank else 0.0,
        1.0 if dense_rank.get(pid, 999) <= args.base_k else 0.0,
        1.0 if dense_rank.get(pid, 999) <= args.slot_max_k else 0.0,
        reciprocal_rank(dense_rank.get(pid)),
        float(dense_rank.get(pid, 999)),
        dense_score,
        dense_z,
        len(candidate_terms_set) / 100.0,
        len(candidate_terms_set - prefix_terms) / max(1, len(candidate_terms_set)),
        jaccard_terms(candidate_terms_set, prefix_terms),
        len(candidate_terms_set & query_terms_set) / max(1, len(query_terms_set)),
        jaccard_terms(candidate_terms_set, query_terms_set),
    ]


def slot_feature_dim(examples: list[dict[str, Any]], args: argparse.Namespace, variant: str) -> int | None:
    for example in examples:
        candidates = slot_candidate_ids(example, args)
        if candidates:
            return int(len(slot_features(example, candidates[0], args, variant=variant)))
    return None


def select_slot_records(records: list[dict[str, Any]], total_slots: int, per_query_cap: int) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for record in records:
        if len(selected) >= total_slots:
            break
        key = (record["qid"], record["pid"])
        if key in seen or counts[record["qid"]] >= per_query_cap:
            continue
        selected.append(record)
        seen.add(key)
        counts[record["qid"]] += 1
    return selected


def slot_plan_by_qid(selected: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_qid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for record in selected:
        by_qid[record["qid"]].append((int(record["rank"]), record["pid"]))
    return {
        qid: [pid for _, pid in sorted(values)]
        for qid, values in by_qid.items()
    }


def slot_plan_stats(
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    target_avg_k: float,
    total_slots: int,
    selection: str,
) -> dict[str, Any]:
    oracle_selected = select_slot_records(
        sorted(records, key=lambda item: (-item["gain"], item["rank"], item["qid"], item["pid"])),
        total_slots,
        args.slot_per_query_cap,
    )
    selected_positive_n = sum(int(record["label"]) for record in selected)
    population_positive_n = sum(int(record["label"]) for record in records)
    selected_gain = sum(float(record["gain"]) for record in selected)
    oracle_gain = sum(float(record["gain"]) for record in oracle_selected)
    return {
        "selection": selection,
        "base_k": args.base_k,
        "slot_max_k": args.slot_max_k,
        "slot_per_query_cap": args.slot_per_query_cap,
        "target_avg_k": target_avg_k,
        "selected_slots": len(selected),
        "selected_queries": len({record["qid"] for record in selected}),
        "avg_k": round(args.base_k + len(selected) / max(1, len(examples)), 4),
        "selected_positive_n": int(selected_positive_n),
        "population_positive_n": int(population_positive_n),
        "selected_positive_rate": round(selected_positive_n / max(1, len(selected)), 4),
        "population_positive_rate": round(population_positive_n / max(1, len(records)), 4),
        "selected_recall_gain": round(selected_gain / max(1, len(examples)), 4),
        "oracle_recall_gain_same_budget": round(oracle_gain / max(1, len(examples)), 4),
    }


def merge_graph_slots(example: dict[str, Any], selected_slots: list[str], args: argparse.Namespace) -> list[str]:
    base = dedupe(example["graph_ids"][: args.base_k])
    base_set = set(base)
    selected_set = set(selected_slots)
    tail = [
        pid
        for pid in example["graph_ids"][args.base_k : args.slot_max_k]
        if pid in selected_set and pid not in base_set
    ]
    return base + dedupe(tail)


def budget_plan_stats(
    examples: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    all_gains: list[float],
    start_k: int,
    large_k: int,
    target_avg_k: float,
    expand_count: int,
    selection: str,
) -> dict[str, Any]:
    selected_gains = [graph_budget_gain(example, start_k, large_k) for example in selected]
    return {
        "selection": selection,
        "start_k": start_k,
        "large_k": large_k,
        "target_avg_k": target_avg_k,
        "expanded_n": expand_count,
        "expanded_rate": round(expand_count / max(1, len(examples)), 4),
        "selected_positive_rate": round(
            sum(1 for gain in selected_gains if gain > 0) / max(1, len(selected_gains)),
            4,
        ),
        "population_positive_rate": round(
            sum(1 for gain in all_gains if gain > 0) / max(1, len(all_gains)),
            4,
        ),
        "selected_avg_gain": round(float(np.mean(selected_gains)), 4) if selected_gains else 0.0,
        "oracle_avg_gain": round(
            float(np.mean(sorted((gain for gain in all_gains if gain > 0), reverse=True)[:expand_count])),
            4,
        )
        if expand_count
        else 0.0,
    }


def build_example(
    dataset: str,
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    retrieval = retrieval_row.get("retrieval", {})
    dense_passages = list(retrieval.get("dense") or [])
    graph_passages = list(retrieval.get("graph") or [])
    dense_ids = [str(passage.get("id")) for passage in dense_passages if passage.get("id") is not None]
    graph_ids = [str(passage.get("id")) for passage in graph_passages if passage.get("id") is not None]
    dense_scores = [float(passage.get("score", 0.0) or 0.0) for passage in dense_passages]
    graph_scores = [float(passage.get("score", 0.0) or 0.0) for passage in graph_passages]
    gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
    return {
        "id": str(row["id"]),
        "dataset": dataset,
        "row": row,
        "gold": gold,
        "dense_ids": dense_ids,
        "graph_ids": graph_ids,
        "dense_scores": dense_scores,
        "graph_scores": graph_scores,
        "passage_lookup": passage_lookup(retrieval),
        "features": query_features(row, dense_ids, graph_ids, dense_scores, graph_scores, args),
    }


def build_generation_row(
    example: dict[str, Any],
    rankings: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    methods = selected_generation_methods(rankings, args)
    lookup = example["passage_lookup"]
    retrieval = {
        method: [lookup[pid] for pid in rankings[method] if pid in lookup]
        for method in methods
    }
    row = example["row"]
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "gold_answer": row.get("gold_answer", row.get("answer")),
        "gold_answers": row.get("gold_answers", [row.get("answer")]),
        "dataset_name": row.get("dataset_name") or example.get("dataset"),
        "workload": row.get("workload"),
        "question_type": row.get("question_type"),
        "retrieval": retrieval,
        "main_table_methods": methods,
        "main_table_decisions": {
            method: {"selected_path": method, "top_k": len(retrieval[method])}
            for method in methods
        },
    }


def selected_generation_methods(rankings: dict[str, list[str]], args: argparse.Namespace) -> list[str]:
    if args.generation_methods == ["auto"]:
        return sorted(rankings)
    if args.generation_methods == ["slot_ablation"]:
        return [method for method in slot_ablation_generation_methods(args) if method in rankings]
    return [method for method in args.generation_methods if method in rankings]


def slot_ablation_generation_methods(args: argparse.Namespace) -> list[str]:
    methods: list[str] = [
        f"graph_top{args.base_k}",
        "graph_top8",
        "graph_top10",
    ]
    for target_avg_k in args.slot_target_avg_k:
        methods.extend(
            [
                random_slot_method_name(args.base_k, args.slot_max_k, args.slot_per_query_cap, target_avg_k),
                score_slot_method_name(args.base_k, args.slot_max_k, args.slot_per_query_cap, target_avg_k),
            ]
        )
        for variant in normalize_slot_feature_variants(args.slot_feature_variants):
            methods.append(
                slot_method_name(
                    args.base_k,
                    args.slot_max_k,
                    args.slot_per_query_cap,
                    target_avg_k,
                    variant=variant,
                )
            )
    return dedupe(methods)


def passage_lookup(retrieval: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for passages in retrieval.values():
        if not isinstance(passages, list):
            continue
        for passage in passages:
            passage_id = passage.get("id")
            if passage_id is not None:
                lookup[str(passage_id)] = passage
    return lookup


def query_features(
    row: dict[str, Any],
    dense_ids: list[str],
    graph_ids: list[str],
    dense_scores: list[float],
    graph_scores: list[float],
    args: argparse.Namespace,
) -> list[float]:
    dense_top = dense_ids[: args.base_k]
    graph_top = graph_ids[: args.base_k]
    dense_set = set(dense_top)
    graph_set = set(graph_top)
    values = [
        len(dense_set & graph_set) / max(1, args.base_k),
        float(len([pid for pid in graph_top if pid not in dense_set])),
        score_at(dense_scores, 0),
        score_at(dense_scores, 0) - score_at(dense_scores, 1),
        float(np.std(dense_scores[: args.base_k])) if dense_scores else 0.0,
        score_at(graph_scores, 0),
        score_at(graph_scores, 0) - score_at(graph_scores, 1),
        float(np.std(graph_scores[: args.base_k])) if graph_scores else 0.0,
        float(row.get("dense_entity_coverage_ratio", 0.0) or 0.0),
        float(row.get("dense_unique_doc_count", 0.0) or 0.0),
        float(row.get("query_length_tokens", 0.0) or 0.0),
        float(row.get("query_entity_count", 0.0) or 0.0),
        float(row.get("conjunction_count", 0.0) or 0.0),
        float(row.get("has_comparison_cue", 0.0) or 0.0),
    ]
    question = str(row.get("question") or "")
    values.extend([
        float(len(re.findall(r"\w+", question))),
        float(question.count(",")),
        1.0 if re.search(r"\b(and|or|both|between|before|after)\b", question.lower()) else 0.0,
    ])
    return values


def budget_features(example: dict[str, Any], start_k: int, large_k: int, args: argparse.Namespace) -> list[float]:
    dense_ids = example["dense_ids"]
    graph_ids = example["graph_ids"]
    dense_scores = example["dense_scores"]
    graph_scores = example["graph_scores"]
    prefix = graph_ids[:start_k]
    tail = graph_ids[start_k:large_k]
    prefix_set = set(prefix)
    dense_base_set = set(dense_ids[: args.base_k])
    dense_deep_rank = {pid: idx + 1 for idx, pid in enumerate(dense_ids)}
    tail_scores = graph_scores[start_k:large_k]
    prefix_scores = graph_scores[:start_k]
    tail_dense_ranks = [dense_deep_rank[pid] for pid in tail if pid in dense_deep_rank]
    tail_terms = set()
    prefix_terms = set()
    for pid in tail:
        tail_terms.update(passage_terms(example, pid))
    for pid in prefix:
        prefix_terms.update(passage_terms(example, pid))
    tail_new_terms = tail_terms - prefix_terms
    features = list(example["features"])
    features.extend(
        [
            float(start_k),
            float(large_k),
            float(large_k - start_k),
            len(tail) / max(1, large_k - start_k),
            len([pid for pid in tail if pid not in prefix_set]) / max(1, len(tail)),
            len([pid for pid in tail if pid not in dense_base_set]) / max(1, len(tail)),
            len([pid for pid in tail if pid in dense_deep_rank]) / max(1, len(tail)),
            reciprocal_rank(min(tail_dense_ranks)) if tail_dense_ranks else 0.0,
            score_at(graph_scores, start_k),
            score_at(graph_scores, start_k - 1) - score_at(graph_scores, start_k),
            float(np.mean(tail_scores)) if tail_scores else 0.0,
            float(np.max(tail_scores)) if tail_scores else 0.0,
            float(np.std(tail_scores)) if tail_scores else 0.0,
            (float(np.mean(tail_scores)) - float(np.mean(prefix_scores))) if tail_scores and prefix_scores else 0.0,
            len(tail_new_terms) / max(1, len(tail_terms)),
            jaccard_terms(tail_terms, prefix_terms),
        ]
    )
    return features


def passage_terms(example: dict[str, Any], pid: str) -> set[str]:
    cache = example.setdefault("_term_cache", {})
    if pid in cache:
        return cache[pid]
    passage = example.get("passage_lookup", {}).get(pid, {})
    text = " ".join([str(passage.get("title") or pid), str(passage.get("text") or "")])
    terms = set(content_tokens(text))
    cache[pid] = terms
    return terms


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in CONTENT_STOPWORDS
    ]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)}


def reciprocal_rank(rank: int | float | None) -> float:
    return 0.0 if rank is None else 1.0 / float(rank)


def jaccard_terms(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def choose_better_graph_budget(example: dict[str, Any], small_k: int, large_k: int) -> list[str]:
    small = example["graph_ids"][:small_k]
    large = example["graph_ids"][:large_k]
    small_score = score_ranking(small, example["gold"], set())["recall"]
    large_score = score_ranking(large, example["gold"], set())["recall"]
    return large if large_score > small_score else small


def graph_budget_gain(example: dict[str, Any], small_k: int, large_k: int) -> float:
    small = score_ranking(example["graph_ids"][:small_k], example["gold"], set())["recall"]
    large = score_ranking(example["graph_ids"][:large_k], example["gold"], set())["recall"]
    return float(large - small)


def positive_scores(clf: Any, x_rows: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_rows)
        if probs.shape[1] == 1:
            return np.full(x_rows.shape[0], float(clf.classes_[0]))
        pos_col = list(clf.classes_).index(1)
        return probs[:, pos_col]
    return clf.predict(x_rows)


def score_ranking(ids: list[str], gold: set[str], dense_top: set[str]) -> dict[str, Any]:
    selected = set(ids)
    missed = gold - dense_top
    recall = len(selected & gold) / max(1, len(gold))
    return {
        "recall": float(recall),
        "missed_recovery": len(selected & missed) / len(missed) if missed else None,
        "hit": 1.0 if missed and selected & missed else 0.0 if missed else None,
    }


def update_agg(agg: dict[str, Any], score: dict[str, Any], k: int) -> None:
    agg["n"] += 1
    agg["k_sum"] += k
    agg["recall_sum"] += score["recall"]
    if score["missed_recovery"] is not None:
        agg["missed_n"] += 1
        agg["missed_recovery_sum"] += score["missed_recovery"]
        agg["hit_sum"] += score["hit"]


def make_aggs() -> dict[str, dict[str, dict[str, Any]]]:
    return defaultdict(
        lambda: defaultdict(
            lambda: {"n": 0, "missed_n": 0, "k_sum": 0.0, "recall_sum": 0.0, "missed_recovery_sum": 0.0, "hit_sum": 0.0}
        )
    )


def summarize_aggs(aggs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for method, by_stratum in aggs.items():
        out[method] = {}
        for stratum, agg in by_stratum.items():
            n = agg["n"]
            missed_n = agg["missed_n"]
            out[method][stratum] = {
                "n": int(n),
                "missed_n": int(missed_n),
                "avg_k": round(agg["k_sum"] / n, 4) if n else None,
                "recall": round(agg["recall_sum"] / n, 4) if n else None,
                "missed_recovery": round(agg["missed_recovery_sum"] / missed_n, 4) if missed_n else None,
                "hit_rate": round(agg["hit_sum"] / missed_n, 4) if missed_n else None,
            }
    return out


def print_summary(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        print("\n" + "=" * 88)
        print(f"DATASET {report['dataset']} n={report['n']} base_k={report['base_k']} small_k={report['small_k']}")
        print(f"candidate_source={report['candidate_source']}")
        print(f"target_avg_k={report.get('target_avg_k')}")
        print(f"expansion_metrics={report['expansion_metrics']}")
        if report.get("budget_plan_metrics"):
            print("budget_plan_metrics:")
            for method, metrics in sorted(report["budget_plan_metrics"].items()):
                print(f"  {method}: {metrics}")
        if report.get("slot_model_metrics"):
            print(f"slot_model_metrics={report['slot_model_metrics']}")
        if report.get("slot_plan_metrics"):
            print("slot_plan_metrics:")
            for method, metrics in sorted(report["slot_plan_metrics"].items()):
                print(f"  {method}: {metrics}")
        for stratum in ("ALL", "dense_miss"):
            print(f"\n{stratum}")
            rows = []
            for method, by_stratum in report["metrics"].items():
                row = by_stratum.get(stratum)
                if row:
                    rows.append((method, row))
            rows.sort(key=lambda item: (item[1]["recall"] or 0.0, -(item[1]["avg_k"] or 0.0)), reverse=True)
            for method, row in rows[:40]:
                print(
                    f"{method}: R={row['recall']} missedRec={row['missed_recovery']} "
                    f"hit={row['hit_rate']} avgK={row['avg_k']}"
                )


def score_at(scores: list[float], idx: int) -> float:
    return float(scores[idx]) if idx < len(scores) else 0.0


def zscore_at(scores: list[float], idx: int) -> float:
    if idx < 0 or idx >= len(scores) or not scores:
        return 0.0
    std = float(np.std(scores))
    if std <= 1e-12:
        return 0.0
    return (float(scores[idx]) - float(np.mean(scores))) / std


def score_for_pid(ids: list[str], scores: list[float], pid: str) -> float:
    for idx, current in enumerate(ids):
        if current == pid:
            return score_at(scores, idx)
    return 0.0


def dedupe(ids: list[str]) -> list[str]:
    seen = set()
    out = []
    for pid in ids:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def query_terms(example: dict[str, Any]) -> set[str]:
    cache_key = "_query_terms"
    if cache_key in example:
        return example[cache_key]
    question = str(example.get("row", {}).get("question") or "")
    terms = set(content_tokens(question))
    example[cache_key] = terms
    return terms


def format_threshold(threshold: float) -> str:
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def budget_method_name(start_k: int, large_k: int, target_avg_k: float) -> str:
    return f"budget_graph_{start_k}_to_{large_k}_avg{format_threshold(target_avg_k)}"


def random_budget_method_name(start_k: int, large_k: int, target_avg_k: float) -> str:
    return f"random_{budget_method_name(start_k, large_k, target_avg_k)}"


def slot_method_name(
    base_k: int,
    max_k: int,
    cap: int,
    target_avg_k: float,
    *,
    variant: str = "full",
) -> str:
    base = f"slot_graph_{base_k}_to_{max_k}_cap{cap}_avg{format_threshold(target_avg_k)}"
    variant = normalize_slot_feature_variant(variant)
    if variant == "full":
        return base
    return f"slot_{variant}_graph_{base_k}_to_{max_k}_cap{cap}_avg{format_threshold(target_avg_k)}"


def random_slot_method_name(base_k: int, max_k: int, cap: int, target_avg_k: float) -> str:
    return f"random_{slot_method_name(base_k, max_k, cap, target_avg_k)}"


def score_slot_method_name(base_k: int, max_k: int, cap: int, target_avg_k: float) -> str:
    return f"score_{slot_method_name(base_k, max_k, cap, target_avg_k)}"


def oracle_slot_method_name(base_k: int, max_k: int, cap: int, target_avg_k: float) -> str:
    return f"oracle_{slot_method_name(base_k, max_k, cap, target_avg_k)}"


def normalize_slot_feature_variants(raw_variants: list[str] | None) -> list[str]:
    variants = raw_variants or ["full"]
    out: list[str] = []
    for variant in variants:
        normalized = normalize_slot_feature_variant(variant)
        if normalized not in out:
            out.append(normalized)
    return out or ["full"]


def normalize_slot_feature_variant(variant: str) -> str:
    normalized = str(variant).strip().lower().replace("-", "_")
    aliases = {
        "all": "full",
        "default": "full",
        "no_query_probe": "no_probe",
        "without_probe": "no_probe",
        "without_query_probe": "no_probe",
        "query_only": "probe_only",
        "query_probe_only": "probe_only",
        "rank_score": "graph_only",
        "graph_score": "graph_only",
        "standard_reranker": "passage_rerank",
        "passage_reranker": "passage_rerank",
        "query_candidate": "passage_rerank",
        "no_context": "passage_rerank",
        "no_prefix": "passage_rerank",
        "clean_passage_rerank": "text_rerank",
        "clean_rerank": "text_rerank",
        "standard_text_reranker": "text_rerank",
        "lexical_rerank": "text_rerank",
        "text_reranker": "text_rerank",
        "no_dense": "no_dense_support",
        "without_dense_support": "no_dense_support",
        "without_novelty": "no_novelty",
        "no_overlap": "no_novelty",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {
        "full",
        "no_probe",
        "probe_only",
        "slot_only",
        "passage_rerank",
        "text_rerank",
        "graph_only",
        "no_dense_support",
        "no_novelty",
    }
    if normalized not in supported:
        raise ValueError(
            f"Unsupported --slot-feature-variants value '{variant}'. "
            f"Supported values: {', '.join(sorted(supported))}"
        )
    return normalized


def budget_methods_for(
    start_k: int,
    large_k: int,
    target_avg_k: float,
    args: argparse.Namespace,
) -> list[str]:
    methods = [budget_method_name(start_k, large_k, target_avg_k)]
    if not args.disable_random_budget_baseline:
        methods.append(random_budget_method_name(start_k, large_k, target_avg_k))
    return methods


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**32)


def resolve_dataset_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.result_dirs:
        if len(args.result_dirs) != len(args.datasets):
            raise ValueError("--result-dirs must have the same length as --datasets.")
        return [(dataset, resolve_project_path(path)) for dataset, path in zip(args.datasets, args.result_dirs)]
    return [(dataset, resolve_project_path(DATASET_MAP[dataset])) for dataset in args.datasets]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
