"""Evaluate dense-probed counterfactual graph-operator selection.

This script asks whether there is enough headroom for a stronger second
contribution than a binary pure-bridge gate.

For each query, it builds several fixed evidence-construction operators, scores
each operator against gold passages, then evaluates:

  1. fixed operator baselines
  2. an oracle best-operator upper bound
  3. a 5-fold OOF predicted operator policy from dense/graph probe features

Default operators are fast and use saved retrieval rows only:

  dense_only
  graph_only
  rrf_dense_graph
  hippo_cached_B{1,2,3,5}

Use --include-bridge to add bridge-conditioned operators. That path is slower
because it loads HippoRAG graph state and recomputes PPR complements.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - cloud env dependency.
    raise SystemExit("Missing scikit-learn in this environment.") from exc


DATASET_MAP = {
    "hotpot": ("hipporag_cache/hotpot_shared_500", "results/study_hotpot_hipporag_colbert_500"),
    "2wiki": ("hipporag_cache/2wiki_shared_500", "results/study_2wiki_hipporag_colbert_500"),
    "nq": ("hipporag_cache/nq_shared_500", "results/study_nq_hipporag_colbert_500"),
}


FEATURES = [
    "dense_graph_overlap",
    "dense_top1_score",
    "dense_top1_top2_gap",
    "dense_topk_score_std",
    "dense_entity_coverage_ratio",
    "dense_unique_doc_count",
    "query_length_tokens",
    "query_entity_count",
    "conjunction_count",
    "has_comparison_cue",
    "graph_top1_top2_gap",
    "graph_score_std",
    "graph_new_doc_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual graph operator selection.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--cache-dirs", nargs="*", default=[])
    parser.add_argument("--dense-k", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--rrf-depth", type=int, default=20)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--include-bridge", action="store_true")
    parser.add_argument("--seed-source", default="hipporag_internal", choices=["hipporag_internal"])
    parser.add_argument(
        "--compression-thresholds",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9],
        help="Confidence thresholds for safe graph-evidence compression policies.",
    )
    parser.add_argument(
        "--compression-baseline",
        default="hippo_cached_B5",
        help="Strong fallback operator used by conservative compression.",
    )
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--bridge-share", type=float, default=0.35)
    parser.add_argument("--hipporag-llm-model", default="gpt-4.1")
    parser.add_argument("--hipporag-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--hipporag-llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument(
        "--hipporag-embedding-base-url",
        default=os.getenv("HIPPORAG_EMBEDDING_BASE_URL"),
    )
    parser.add_argument("--output", default="results/counterfactual_operator_selection_eval.json")
    parser.add_argument(
        "--per-sample-output",
        default="results/counterfactual_operator_selection_per_sample.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    all_per_sample: list[dict[str, Any]] = []
    for dataset, cache_dir, result_dir in resolve_dataset_pairs(args):
        report, per_sample = evaluate_dataset(dataset, cache_dir, result_dir, args)
        reports.append(report)
        all_per_sample.extend(per_sample)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    per_sample_output = resolve_project_path(args.per_sample_output)
    per_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_output.open("w", encoding="utf-8") as handle:
        for row in all_per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(reports)
    print(f"\nFull report: {output}")
    print(f"Per-sample report: {per_sample_output}")


def evaluate_dataset(
    dataset: str,
    cache_dir: Path,
    result_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n=== {dataset}: counterfactual operator selection ===", flush=True)
    start = time.time()
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    if args.max_queries:
        rows = rows[: args.max_queries]

    bridge_builder = None
    if args.include_bridge:
        bridge_builder = BridgeOperatorBuilder(cache_dir, args)

    examples = []
    for index, row in enumerate(rows, start=1):
        gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
        dense_ids = list(row.get("dense_ids") or [])
        graph_ids = list(row.get("graph_ids") or [])
        if not gold:
            continue

        method_to_ids = build_fast_operator_outputs(row, args)
        if bridge_builder is not None:
            method_to_ids.update(bridge_builder.build(row, args))

        records = {
            method: score_ids(ids, gold, set(dense_ids[: args.dense_k]), method)
            for method, ids in method_to_ids.items()
        }
        oracle_method = choose_oracle_method(records, method_tiebreak_order(args, include_bridge=bridge_builder is not None))
        examples.append(
            {
                "row": row,
                "gold": gold,
                "dense_miss": bool(gold - set(dense_ids[: args.dense_k])),
                "features": feature_vector(row, args.dense_k),
                "method_to_ids": method_to_ids,
                "records": records,
                "oracle_method": oracle_method,
            }
        )

        if index % 50 == 0:
            print(f"  prepared {index}/{len(rows)} [{time.time() - start:.1f}s]", flush=True)

    predicted_methods, policy_metrics = train_oof_operator_policy(examples, args)
    compression_methods, compression_metrics = train_oof_compression_policies(examples, args)
    oracle_compression_methods = oracle_compression_policy(examples, args)

    fixed_aggs = make_metric_aggs()
    oracle_aggs = make_metric_aggs()
    predicted_aggs = make_metric_aggs()
    oracle_compression_aggs = make_metric_aggs()
    compression_aggs = make_metric_aggs()
    label_counts = Counter()
    pred_counts = Counter()
    compression_counts: dict[str, Counter[str]] = defaultdict(Counter)
    oracle_compression_counts = Counter()
    per_sample: list[dict[str, Any]] = []

    for example in examples:
        row = example["row"]
        strata = ["ALL"]
        if example["dense_miss"]:
            strata.append("dense_miss")
        label = row.get("label")
        strata.append("label_tie_or_invalid" if label is None else f"label_{int(label)}")

        for method, record in example["records"].items():
            for stratum in strata:
                update_metric_agg(fixed_aggs[method][stratum], record)

        oracle_method = example["oracle_method"]
        predicted_method = predicted_methods.get(str(row["id"]), most_common_oracle(examples))
        oracle_compression_method = oracle_compression_methods.get(str(row["id"]), args.compression_baseline)
        label_counts[oracle_method] += 1
        pred_counts[predicted_method] += 1
        oracle_compression_counts[oracle_compression_method] += 1
        for stratum in strata:
            update_metric_agg(oracle_aggs["oracle_best"][stratum], example["records"][oracle_method])
            update_metric_agg(predicted_aggs["predicted_policy"][stratum], example["records"][predicted_method])
            update_metric_agg(
                oracle_compression_aggs["oracle_compression"][stratum],
                example["records"][oracle_compression_method],
            )
        for threshold in args.compression_thresholds:
            policy_name = compression_policy_name(threshold)
            compressed_method = compression_methods.get(policy_name, {}).get(str(row["id"]), args.compression_baseline)
            compression_counts[policy_name][compressed_method] += 1
            for stratum in strata:
                update_metric_agg(compression_aggs[policy_name][stratum], example["records"][compressed_method])

        per_sample.append(
            {
                "dataset": dataset,
                "id": row.get("id"),
                "dense_miss": example["dense_miss"],
                "oracle_method": oracle_method,
                "predicted_method": predicted_method,
                "oracle_compression_method": oracle_compression_method,
                "compression_methods": {
                    policy_name: by_id.get(str(row["id"]), args.compression_baseline)
                    for policy_name, by_id in compression_methods.items()
                },
                "oracle_recall": example["records"][oracle_method]["recall"],
                "predicted_recall": example["records"][predicted_method]["recall"],
                "method_scores": {
                    method: {
                        "recall": record["recall"],
                        "missed_recovery": record["missed_recovery"],
                        "hit": record["hit"],
                    }
                    for method, record in example["records"].items()
                },
            }
        )

    report = {
        "dataset": dataset,
        "n": len(examples),
        "dense_k": args.dense_k,
        "budgets": args.budgets,
        "include_bridge": bool(args.include_bridge),
        "features": FEATURES,
        "operator_metrics": summarize_nested_aggs(fixed_aggs),
        "oracle_metrics": summarize_nested_aggs(oracle_aggs),
        "predicted_metrics": summarize_nested_aggs(predicted_aggs),
        "oracle_compression_metrics": summarize_nested_aggs(oracle_compression_aggs),
        "compression_metrics": summarize_nested_aggs(compression_aggs),
        "oracle_label_counts": dict(label_counts),
        "predicted_label_counts": dict(pred_counts),
        "oracle_compression_label_counts": dict(oracle_compression_counts),
        "compression_label_counts": {name: dict(counts) for name, counts in compression_counts.items()},
        "policy_metrics": policy_metrics,
        "compression_policy_metrics": compression_metrics,
    }
    return report, per_sample


def build_fast_operator_outputs(row: dict[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
    dense_ids = list(row.get("dense_ids") or [])
    graph_ids = list(row.get("graph_ids") or [])
    out = {
        "dense_only": dedupe(dense_ids)[: args.dense_k],
        "graph_only": dedupe(graph_ids)[: args.dense_k],
        "rrf_dense_graph": rrf_merge(dense_ids[: args.rrf_depth], graph_ids[: args.rrf_depth], args)[: args.dense_k],
    }
    for budget in args.budgets:
        out[f"hippo_cached_B{budget}"] = splice_dense_complement(dense_ids, graph_ids, budget, args.dense_k)
    return out


class BridgeOperatorBuilder:
    def __init__(self, cache_dir: Path, args: argparse.Namespace) -> None:
        from eval_residual_graph_completion import (  # noqa: WPS433 - optional slow path.
            GraphContext,
            HippoInternalSeeder,
        )

        self.mod = __import__("eval_residual_graph_completion")
        self.ctx = GraphContext(cache_dir)
        self.seeder = HippoInternalSeeder(
            cache_dir=cache_dir,
            ctx=self.ctx,
            llm_model=args.hipporag_llm_model,
            embedding_model=args.hipporag_embedding_model,
            llm_base_url=args.hipporag_llm_base_url,
            embedding_base_url=args.hipporag_embedding_base_url,
            include_passage_weights=False,
        )

    def build(self, row: dict[str, Any], args: argparse.Namespace) -> dict[str, list[str]]:
        dense_ids = list(row.get("dense_ids") or [])[: args.dense_k]
        graph_ids = list(row.get("graph_ids") or [])
        dense_set = set(dense_ids)
        query_reset, _ = self.mod.build_query_reset(row, graph_ids, self.ctx, self.seeder, args)
        bridge_reset = self.mod.build_bridge_reset(self.ctx, dense_ids)
        query_bridge_reset = self.mod.combine_query_bridge_reset(query_reset, bridge_reset, args.bridge_share)
        pr_bridge = self.ctx.pagerank(query_bridge_reset, args.damping)
        max_budget = max(args.budgets)
        bridge_complement = [
            pid
            for pid, _ in self.mod.rank_passage_complements(self.ctx, pr_bridge, dense_set, max_budget)
        ]
        return {
            f"bridge_B{budget}": dedupe(dense_ids[: max(0, args.dense_k - budget)] + bridge_complement[:budget])[
                : args.dense_k
            ]
            for budget in args.budgets
        }


def score_ids(ids: list[str], gold: set[str], dense_top: set[str], method: str) -> dict[str, Any]:
    selected = set(ids)
    missed = gold - dense_top
    recall = len(selected & gold) / max(1, len(gold))
    missed_recovery = len(selected & missed) / len(missed) if missed else None
    hit = 1.0 if missed and selected & missed else 0.0 if missed else None
    return {
        "recall": float(recall),
        "missed_recovery": None if missed_recovery is None else float(missed_recovery),
        "hit": hit,
        "size": len(ids),
        "graph_budget": graph_budget_for_method(method),
    }


def choose_oracle_method(records: dict[str, dict[str, Any]], order: list[str]) -> str:
    order_index = {method: index for index, method in enumerate(order)}
    return max(
        records,
        key=lambda method: (
            records[method]["recall"],
            records[method]["missed_recovery"] if records[method]["missed_recovery"] is not None else -1.0,
            -order_index.get(method, 10_000),
        ),
    )


def train_oof_operator_policy(examples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    if not examples:
        return {}, {"error": "no_examples"}
    labels = [example["oracle_method"] for example in examples]
    ids = [str(example["row"]["id"]) for example in examples]
    x = np.asarray([example["features"] for example in examples], dtype=np.float64)
    y = np.asarray(labels)
    counts = Counter(labels)
    min_class = min(counts.values())
    if len(counts) < 2 or min_class < 2:
        majority = counts.most_common(1)[0][0]
        return {qid: majority for qid in ids}, {
            "mode": "majority",
            "oracle_label_counts": dict(counts),
            "accuracy": None,
        }

    n_splits = min(args.num_folds, min_class)
    predicted: dict[str, str] = {}
    y_true_all = []
    y_pred_all = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
    for train_idx, test_idx in cv.split(x, y):
        y_train = y[train_idx]
        if len(set(y_train.tolist())) < 2:
            clf = DummyClassifier(strategy="most_frequent")
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
        clf.fit(x[train_idx], y_train)
        fold_pred = clf.predict(x[test_idx])
        for local_idx, pred in zip(test_idx.tolist(), fold_pred.tolist()):
            predicted[ids[local_idx]] = str(pred)
            y_true_all.append(str(y[local_idx]))
            y_pred_all.append(str(pred))

    return predicted, {
        "mode": "multiclass_oof",
        "n_splits": n_splits,
        "oracle_label_counts": dict(counts),
        "accuracy": round(float(accuracy_score(y_true_all, y_pred_all)), 4) if y_true_all else None,
    }


def train_oof_compression_policies(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if not examples:
        return {}, {"error": "no_examples"}
    baseline = args.compression_baseline
    if any(baseline not in example["records"] for example in examples):
        return {}, {"error": f"missing_baseline:{baseline}"}

    candidate_methods = compression_candidate_methods(args)
    candidate_methods = [
        method
        for method in candidate_methods
        if method != baseline and all(method in example["records"] for example in examples)
    ]
    ids = [str(example["row"]["id"]) for example in examples]
    x = np.asarray([example["features"] for example in examples], dtype=np.float64)
    safe_probs: dict[str, dict[str, float]] = {method: {} for method in candidate_methods}
    classifier_metrics: dict[str, Any] = {}

    for method in candidate_methods:
        y = np.asarray(
            [
                1
                if example["records"][method]["recall"] >= example["records"][baseline]["recall"] - 1e-12
                else 0
                for example in examples
            ],
            dtype=np.int64,
        )
        counts = Counter(y.tolist())
        classifier_metrics[method] = {
            "positive_n": int(np.sum(y == 1)),
            "positive_rate": round(float(np.mean(y)), 4),
        }
        if len(counts) < 2 or min(counts.values()) < 2:
            const_prob = float(np.mean(y))
            for qid in ids:
                safe_probs[method][qid] = const_prob
            classifier_metrics[method]["mode"] = "constant"
            continue

        n_splits = min(args.num_folds, min(counts.values()))
        y_true_all = []
        y_pred_all = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
        for train_idx, test_idx in cv.split(x, y):
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
            clf.fit(x[train_idx], y[train_idx])
            fold_prob = clf.predict_proba(x[test_idx])[:, 1]
            fold_pred = (fold_prob >= 0.5).astype(np.int64)
            for local_idx, prob in zip(test_idx.tolist(), fold_prob.tolist()):
                safe_probs[method][ids[local_idx]] = float(prob)
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(fold_pred.tolist())
        classifier_metrics[method]["mode"] = "binary_oof"
        classifier_metrics[method]["n_splits"] = n_splits
        classifier_metrics[method]["accuracy@0.5"] = round(float(accuracy_score(y_true_all, y_pred_all)), 4)

    policies: dict[str, dict[str, str]] = {}
    for threshold in args.compression_thresholds:
        policy_name = compression_policy_name(threshold)
        policies[policy_name] = {}
        for qid in ids:
            selected = baseline
            for method in candidate_methods:
                if safe_probs[method].get(qid, 0.0) >= threshold:
                    selected = method
                    break
            policies[policy_name][qid] = selected

    return policies, {
        "mode": "safe_compression_oof",
        "baseline": baseline,
        "candidate_methods": candidate_methods,
        "thresholds": args.compression_thresholds,
        "classifiers": classifier_metrics,
    }


def oracle_compression_policy(examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    baseline = args.compression_baseline
    candidate_methods = [
        method
        for method in compression_candidate_methods(args)
        if all(method in example["records"] for example in examples)
    ]
    out = {}
    for example in examples:
        qid = str(example["row"]["id"])
        baseline_recall = example["records"][baseline]["recall"]
        selected = baseline
        for method in candidate_methods:
            if example["records"][method]["recall"] >= baseline_recall - 1e-12:
                selected = method
                break
        out[qid] = selected
    return out


def compression_candidate_methods(args: argparse.Namespace) -> list[str]:
    methods = ["dense_only"]
    methods.extend(f"hippo_cached_B{budget}" for budget in sorted(args.budgets))
    return dedupe(methods)


def compression_policy_name(threshold: float) -> str:
    return f"safe_compression_t{threshold:g}"


def feature_vector(row: dict[str, Any], dense_k: int) -> list[float]:
    dense_ids = list(row.get("dense_ids") or [])[:dense_k]
    graph_ids = list(row.get("graph_ids") or [])[:dense_k]
    dense_set = set(dense_ids)
    graph_scores = [float(value) for value in (row.get("graph_scores") or [])[:dense_k]]
    overlap = len(dense_set & set(graph_ids)) / max(1, dense_k)
    graph_new_doc_count = len([pid for pid in graph_ids if pid not in dense_set])
    graph_gap = graph_scores[0] - graph_scores[1] if len(graph_scores) > 1 else 0.0
    graph_std = float(np.std(graph_scores)) if graph_scores else 0.0

    values = {
        "dense_graph_overlap": overlap,
        "graph_top1_top2_gap": graph_gap,
        "graph_score_std": graph_std,
        "graph_new_doc_count": float(graph_new_doc_count),
    }
    for key in FEATURES:
        if key not in values:
            values[key] = float(row.get(key, 0.0) or 0.0)
    return [float(values[key]) for key in FEATURES]


def rrf_merge(dense_ids: list[str], graph_ids: list[str], args: argparse.Namespace) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    counter = 0
    for source in (dense_ids, graph_ids):
        for rank, pid in enumerate(source, start=1):
            if pid not in first_seen:
                first_seen[pid] = counter
                counter += 1
            scores[pid] += 1.0 / (args.rrf_k + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], first_seen[pid]))


def splice_dense_complement(dense_ids: list[str], graph_ids: list[str], budget: int, dense_k: int) -> list[str]:
    dense_prefix = dense_ids[: max(0, dense_k - budget)]
    used = set(dense_prefix)
    complement = [pid for pid in graph_ids if pid not in used]
    return dedupe(dense_prefix + complement[:budget])[:dense_k]


def make_metric_aggs() -> dict[str, dict[str, dict[str, Any]]]:
    return defaultdict(lambda: defaultdict(new_metric_agg))


def new_metric_agg() -> dict[str, Any]:
    return {
        "n": 0,
        "recall_sum": 0.0,
        "missed_n": 0,
        "missed_recovery_sum": 0.0,
        "hit_sum": 0.0,
        "size_sum": 0.0,
        "graph_budget_sum": 0.0,
    }


def update_metric_agg(agg: dict[str, Any], record: dict[str, Any]) -> None:
    agg["n"] += 1
    agg["recall_sum"] += record["recall"]
    agg["size_sum"] += record["size"]
    agg["graph_budget_sum"] += record["graph_budget"]
    if record["missed_recovery"] is not None:
        agg["missed_n"] += 1
        agg["missed_recovery_sum"] += record["missed_recovery"]
        agg["hit_sum"] += record["hit"]


def summarize_nested_aggs(aggs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for method, by_stratum in aggs.items():
        out[method] = {}
        for stratum, agg in by_stratum.items():
            n = agg["n"]
            missed_n = agg["missed_n"]
            out[method][stratum] = {
                "n": int(n),
                "missed_n": int(missed_n),
                "recall@5": round(agg["recall_sum"] / n, 4) if n else None,
                "missed_recovery": round(agg["missed_recovery_sum"] / missed_n, 4) if missed_n else None,
                "hit_rate": round(agg["hit_sum"] / missed_n, 4) if missed_n else None,
                "avg_size": round(agg["size_sum"] / n, 4) if n else None,
                "avg_graph_budget": round(agg["graph_budget_sum"] / n, 4) if n else None,
            }
    return out


def method_tiebreak_order(args: argparse.Namespace, include_bridge: bool) -> list[str]:
    order = ["dense_only"]
    order += [f"hippo_cached_B{budget}" for budget in sorted(args.budgets)]
    order += ["graph_only", "rrf_dense_graph"]
    if include_bridge:
        order += [f"bridge_B{budget}" for budget in sorted(args.budgets)]
    return order


def graph_budget_for_method(method: str) -> int:
    if method == "dense_only":
        return 0
    if method.startswith("hippo_cached_B") or method.startswith("bridge_B"):
        try:
            return int(method.rsplit("B", 1)[1])
        except (IndexError, ValueError):
            return 5
    return 5


def most_common_oracle(examples: list[dict[str, Any]]) -> str:
    return Counter(example["oracle_method"] for example in examples).most_common(1)[0][0]


def print_summary(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        print("\n" + "=" * 88)
        print(
            f"DATASET {report['dataset']} n={report['n']} "
            f"include_bridge={report['include_bridge']}"
        )
        print(f"policy={report['policy_metrics']}")
        print(f"compression_policy={report['compression_policy_metrics']}")
        print(f"oracle labels={report['oracle_label_counts']}")
        print(f"pred labels={report['predicted_label_counts']}")
        print(f"oracle compression labels={report['oracle_compression_label_counts']}")
        print(f"compression labels={report['compression_label_counts']}")
        print("\nALL")
        print_methods(report, "ALL")
        print("\ndense_miss")
        print_methods(report, "dense_miss")


def print_methods(report: dict[str, Any], stratum: str) -> None:
    operator_metrics = report["operator_metrics"]
    method_rows = []
    for method, by_stratum in operator_metrics.items():
        row = by_stratum.get(stratum)
        if row:
            method_rows.append((method, row))
    method_rows.sort(key=lambda item: item[1]["recall@5"] or 0.0, reverse=True)
    for method, row in method_rows:
        print(
            f"{method}: R={row['recall@5']} missedRec={row['missed_recovery']} "
            f"hit={row['hit_rate']} avgB={row['avg_graph_budget']}"
        )
    for group_name in ("oracle_metrics", "predicted_metrics", "oracle_compression_metrics", "compression_metrics"):
        for method, by_stratum in report[group_name].items():
            row = by_stratum.get(stratum)
            if row:
                print(
                    f"{method}: R={row['recall@5']} missedRec={row['missed_recovery']} "
                    f"hit={row['hit_rate']} avgB={row['avg_graph_budget']}"
                )


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_dataset_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.result_dirs or args.cache_dirs:
        if len(args.result_dirs) != len(args.cache_dirs):
            raise ValueError("--result-dirs and --cache-dirs must have the same length.")
        return [
            (Path(result_dir).name, resolve_project_path(cache_dir), resolve_project_path(result_dir))
            for cache_dir, result_dir in zip(args.cache_dirs, args.result_dirs)
        ]
    pairs = []
    for dataset in args.datasets:
        cache_raw, result_raw = DATASET_MAP[dataset]
        pairs.append((dataset, resolve_project_path(cache_raw), resolve_project_path(result_raw)))
    return pairs


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
