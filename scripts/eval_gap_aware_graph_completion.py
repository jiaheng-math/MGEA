"""Evaluate an operational gap-aware graph-completion gate.

This script tests the deployable version of the diagnostic finding from
``eval_residual_graph_completion.py``:

  - Bridge-conditioned graph completion helps mainly on A_bridge_visible gaps.
  - It hurts query-entity and other gaps.

We therefore learn an out-of-fold detector for A_bridge_visible using only
available query/dense/graph features, then choose:

  if predicted A_bridge_visible:
      bridge-conditioned completion (lambda=0)
  else:
      cached HippoRAG complement

The script recomputes the two candidate complement rankings so that the gate is
evaluated per sample rather than from aggregate bucket summaries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - cloud env dependency.
    raise SystemExit("Missing scikit-learn in this environment.") from exc

from eval_residual_graph_completion import (
    DATASET_MAP,
    GraphContext,
    HippoInternalSeeder,
    build_bridge_reset,
    build_dense_reset,
    build_query_reset,
    classify_query_bucket,
    combine_query_bridge_reset,
    load_jsonl,
    rank_passage_complements,
    resolve_project_path,
)


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
    parser = argparse.ArgumentParser(description="Evaluate predicted A-gap graph completion.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--dense-k", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--bridge-share", type=float, default=0.35)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed-source", default="hipporag_internal", choices=["hipporag_internal"])
    parser.add_argument("--hipporag-llm-model", default="gpt-4.1")
    parser.add_argument("--hipporag-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--hipporag-llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument(
        "--hipporag-embedding-base-url",
        default=os.getenv("HIPPORAG_EMBEDDING_BASE_URL"),
    )
    parser.add_argument(
        "--refined-gap-signatures",
        default=None,
        help=(
            "Optional per-missed JSONL from diagnose_refined_gap_signatures.py. "
            "When provided, query-level action classes replace legacy A/B/C buckets."
        ),
    )
    parser.add_argument(
        "--refined-target-mode",
        default="action_class",
        choices=["action_class", "pure_bridge_exposed"],
        help=(
            "How to collapse per-missed refined signatures into query-level labels. "
            "action_class uses the unique action class or multi_evidence_conflict. "
            "pure_bridge_exposed creates a label only when every missed evidence item "
            "is bridge_exposed_reachable and none is question-visible."
        ),
    )
    parser.add_argument(
        "--gate-positive-class",
        default=None,
        help=(
            "Positive class for oracle/predicted gate. Defaults to A_bridge_visible for legacy, "
            "bridge_exposed_reachable for refined action_class mode, and pure_bridge_exposed "
            "for refined pure_bridge_exposed mode."
        ),
    )
    parser.add_argument("--output", default="results/gap_aware_graph_completion_eval.json")
    parser.add_argument(
        "--generation-input-output",
        default=None,
        help="Optional JSONL generation input containing B=3 gap-aware retrieval methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    generation_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        cache_raw, result_raw = DATASET_MAP[dataset]
        report, dataset_generation_rows = evaluate_dataset(
            dataset=dataset,
            cache_dir=resolve_project_path(cache_raw),
            result_dir=resolve_project_path(result_raw),
            args=args,
        )
        reports.append(report)
        generation_rows.extend(dataset_generation_rows)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)
    if args.generation_input_output:
        generation_output = resolve_project_path(args.generation_input_output)
        generation_output.parent.mkdir(parents=True, exist_ok=True)
        with generation_output.open("w", encoding="utf-8") as handle:
            for row in generation_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Generation input: {generation_output}")

    print_summary(reports, args)
    print(f"\nFull report: {output}")


def evaluate_dataset(
    dataset: str,
    cache_dir: Path,
    result_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n=== {dataset}: gap-aware completion ===", flush=True)
    start = time.time()
    ctx = GraphContext(cache_dir)
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_by_id = {
        row["id"]: row
        for row in load_jsonl(result_dir / "retrieval_results.jsonl")
    }
    refined_query_classes = load_refined_query_classes(args.refined_gap_signatures, dataset, args.refined_target_mode)
    gate_positive_class = args.gate_positive_class
    if gate_positive_class is None:
        if refined_query_classes is None:
            gate_positive_class = "A_bridge_visible"
        elif args.refined_target_mode == "pure_bridge_exposed":
            gate_positive_class = "pure_bridge_exposed"
        else:
            gate_positive_class = "bridge_exposed_reachable"
    if args.max_queries:
        rows = rows[: args.max_queries]

    print(f"loaded {len(rows)} rows, graph nodes={ctx.n} [{time.time() - start:.1f}s]", flush=True)
    seeder = HippoInternalSeeder(
        cache_dir=cache_dir,
        ctx=ctx,
        llm_model=args.hipporag_llm_model,
        embedding_model=args.hipporag_embedding_model,
        llm_base_url=args.hipporag_llm_base_url,
        embedding_base_url=args.hipporag_embedding_base_url,
        include_passage_weights=False,
    )

    examples = []
    for row in rows:
        gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
        dense_ids = list(row.get("dense_ids") or [])[: args.dense_k]
        missed = gold - set(dense_ids)
        legacy_bucket = classify_query_bucket(row, retrieval_by_id.get(row["id"], {}), missed, args.dense_k)
        if refined_query_classes is None:
            bucket = legacy_bucket
        elif not missed:
            bucket = "none"
        else:
            bucket = refined_query_classes.get(str(row["id"]), "refined_missing")
        examples.append(
            {
                "row": row,
                "gold": gold,
                "dense_ids": dense_ids,
                "graph_ids": list(row.get("graph_ids") or []),
                "missed": missed,
                "bucket": bucket,
                "legacy_bucket": legacy_bucket,
                "features": feature_vector(row, args.dense_k),
            }
        )

    gate_probs, gate_metrics = train_oof_a_gate(examples, args, gate_positive_class)
    print(
        f"A-gate AUC={gate_metrics.get('auc_mean')} pos_rate={gate_metrics.get('positive_rate')} "
        f"train_n={gate_metrics.get('train_n')}",
        flush=True,
    )

    aggregators = make_aggregators(args.budgets)
    bucket_counts = Counter(example["bucket"] for example in examples)
    generation_rows: list[dict[str, Any]] = []
    generation_budget = min(args.budgets, key=lambda value: abs(value - 3))
    errors = 0

    for index, example in enumerate(examples, start=1):
        row = example["row"]
        try:
            dense_ids = example["dense_ids"]
            graph_ids = example["graph_ids"]
            dense_set = set(dense_ids)

            query_reset, _ = build_query_reset(row, graph_ids, ctx, seeder, args)
            bridge_reset = build_bridge_reset(ctx, dense_ids)
            dense_reset = build_dense_reset(ctx, dense_ids, row.get("dense_scores") or [])
            query_bridge_reset = combine_query_bridge_reset(query_reset, bridge_reset, args.bridge_share)
            pr_bridge = ctx.pagerank(query_bridge_reset, args.damping)
            pr_dense = ctx.pagerank(dense_reset, args.damping)
            bridge_scores = pr_bridge  # lambda=0: bridge-conditioned completion.

            hippo_complement = [pid for pid in graph_ids if pid not in dense_set]
            bridge_complement = [
                pid
                for pid, _ in rank_passage_complements(ctx, bridge_scores, dense_set, max(args.budgets))
            ]
            oracle_complement = bridge_complement if example["bucket"] == gate_positive_class else hippo_complement
            predicted_use_bridge = gate_probs.get(row["id"], 0.0) >= args.gate_threshold
            predicted_complement = bridge_complement if predicted_use_bridge else hippo_complement

            for budget in args.budgets:
                evaluate_policy(aggregators["hipporag_cached"][budget], example, hippo_complement[:budget], budget, args)
                evaluate_policy(aggregators["always_bridge"][budget], example, bridge_complement[:budget], budget, args)
                evaluate_policy(aggregators["oracle_A_gate"][budget], example, oracle_complement[:budget], budget, args)
                evaluate_policy(aggregators["predicted_A_gate"][budget], example, predicted_complement[:budget], budget, args)
            if args.generation_input_output:
                generation_rows.append(
                    build_generation_row(
                        example=example,
                        retrieval_row=retrieval_by_id.get(row["id"], {}),
                        hippo_complement=hippo_complement,
                        bridge_complement=bridge_complement,
                        oracle_complement=oracle_complement,
                        predicted_complement=predicted_complement,
                        budget=generation_budget,
                        args=args,
                    )
                )
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  [warn] {row.get('id')}: {exc}", flush=True)

        if index % 50 == 0:
            print(f"  progress {index}/{len(examples)} [{time.time() - start:.1f}s]", flush=True)

    return {
        "dataset": dataset,
        "num_rows": len(examples),
        "errors": errors,
        "budgets": args.budgets,
        "gate_threshold": args.gate_threshold,
        "gate_positive_class": gate_positive_class,
        "gap_label_source": "refined" if refined_query_classes is not None else "legacy",
        "refined_target_mode": args.refined_target_mode if refined_query_classes is not None else None,
        "features": FEATURES,
        "bucket_counts": dict(bucket_counts),
        "gate_metrics": gate_metrics,
        "methods": summarize_aggregators(aggregators),
    }, generation_rows


def build_generation_row(
    *,
    example: dict[str, Any],
    retrieval_row: dict[str, Any],
    hippo_complement: list[str],
    bridge_complement: list[str],
    oracle_complement: list[str],
    predicted_complement: list[str],
    budget: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    retrieval = retrieval_row.get("retrieval", {})
    lookup = passage_lookup(retrieval)
    method_to_ids = {
        "dense_only": [p.get("id") for p in list(retrieval.get("dense", []))[: args.dense_k]],
        "graph_only": [p.get("id") for p in list(retrieval.get("graph", []))[: args.dense_k]],
        f"hipporag_cached_B{budget}": splice_ids(example["dense_ids"], hippo_complement, budget, args.dense_k),
        f"always_bridge_B{budget}": splice_ids(example["dense_ids"], bridge_complement, budget, args.dense_k),
        f"oracle_A_gate_B{budget}": splice_ids(example["dense_ids"], oracle_complement, budget, args.dense_k),
        f"predicted_A_gate_B{budget}": splice_ids(example["dense_ids"], predicted_complement, budget, args.dense_k),
    }
    output_retrieval = {
        method: [lookup[pid] for pid in ids if pid in lookup][: args.dense_k]
        for method, ids in method_to_ids.items()
    }
    return {
        "id": retrieval_row.get("id", example["row"]["id"]),
        "question": retrieval_row.get("question", example["row"].get("question")),
        "answer": retrieval_row.get("answer", example["row"].get("answer")),
        "gold_answer": retrieval_row.get("gold_answer", retrieval_row.get("answer", example["row"].get("answer"))),
        "gold_answers": retrieval_row.get("gold_answers", example["row"].get("gold_answers", [example["row"].get("answer")])),
        "dataset_name": retrieval_row.get("dataset_name", example["row"].get("dataset_name")),
        "workload": retrieval_row.get("workload", example["row"].get("workload")),
        "question_type": retrieval_row.get("question_type", example["row"].get("question_type")),
        "gap_bucket": example["bucket"],
        "retrieval": output_retrieval,
        "main_table_methods": list(output_retrieval),
        "main_table_decisions": {
            f"hipporag_cached_B{budget}": {"budget": budget, "policy": "hipporag_cached"},
            f"always_bridge_B{budget}": {"budget": budget, "policy": "always_bridge"},
            f"oracle_A_gate_B{budget}": {"budget": budget, "policy": "oracle_A_gate"},
            f"predicted_A_gate_B{budget}": {"budget": budget, "policy": "predicted_A_gate"},
        },
    }


def load_refined_query_classes(path: str | None, dataset: str, target_mode: str) -> dict[str, str] | None:
    if not path:
        return None
    rows = load_jsonl(resolve_project_path(path))
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    saw_dataset_match = False
    for row in rows:
        row_dataset = row.get("dataset")
        if row_dataset is not None and row_dataset != dataset:
            continue
        if row_dataset == dataset:
            saw_dataset_match = True
        qid = row.get("id")
        action = row.get("action_class")
        if qid is None or not action:
            continue
        by_id[str(qid)].append(row)

    # If the file predates dataset-tagged rows, fall back to all rows.
    if not by_id and not saw_dataset_match:
        for row in rows:
            qid = row.get("id")
            action = row.get("action_class")
            if qid is None or not action:
                continue
            by_id[str(qid)].append(row)

    if target_mode == "pure_bridge_exposed":
        return {qid: pure_bridge_label(items) for qid, items in by_id.items()}

    output = {}
    for qid, items in by_id.items():
        actions = {str(item.get("action_class")) for item in items if item.get("action_class")}
        output[qid] = next(iter(actions)) if len(actions) == 1 else "multi_evidence_conflict"
    return output


def pure_bridge_label(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    all_bridge = all(item.get("action_class") == "bridge_exposed_reachable" for item in items)
    any_query_visible = any(bool(item.get("q_visible")) for item in items)
    if all_bridge and not any_query_visible:
        return "pure_bridge_exposed"
    actions = {str(item.get("action_class")) for item in items if item.get("action_class")}
    if len(actions) == 1:
        return next(iter(actions))
    return "multi_evidence_conflict"


def train_oof_a_gate(
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    gate_positive_class: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_examples = [example for example in examples if example["bucket"] != "none"]
    y = np.asarray([1 if example["bucket"] == gate_positive_class else 0 for example in train_examples], dtype=np.int64)
    x = np.asarray([example["features"] for example in train_examples], dtype=np.float64)
    ids = [str(example["row"]["id"]) for example in train_examples]

    probs: dict[str, float] = {}
    fold_aucs = []
    if len(set(y.tolist())) < 2:
        return probs, {"error": "single_class", "train_n": len(train_examples), "positive_class": gate_positive_class}

    min_class = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    n_splits = min(args.num_folds, min_class)
    if n_splits < 2:
        return probs, {"error": "not_enough_minority", "train_n": len(train_examples), "positive_class": gate_positive_class}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
    for train_idx, test_idx in cv.split(x, y):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=args.random_seed)),
            ]
        )
        clf.fit(x[train_idx], y[train_idx])
        fold_prob = clf.predict_proba(x[test_idx])[:, 1]
        for local_idx, prob in zip(test_idx.tolist(), fold_prob.tolist()):
            probs[ids[local_idx]] = float(prob)
        if len(set(y[test_idx].tolist())) > 1:
            fold_aucs.append(float(roc_auc_score(y[test_idx], fold_prob)))

    # Dense-sufficient rows are not A gaps; default them to cached HippoRAG.
    for example in examples:
        probs.setdefault(str(example["row"]["id"]), 0.0)

    return probs, {
        "train_n": len(train_examples),
        "positive_class": gate_positive_class,
        "positive_n": int(np.sum(y == 1)),
        "positive_rate": round(float(np.mean(y)), 4),
        "auc_mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "auc_std": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
        "n_splits": n_splits,
    }


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


def evaluate_policy(
    agg: dict[str, Any],
    example: dict[str, Any],
    complement_ids: list[str],
    budget: int,
    args: argparse.Namespace,
) -> None:
    dense_ids = example["dense_ids"]
    gold = example["gold"]
    missed = example["missed"]
    final_ids = dedupe(dense_ids[: max(0, args.dense_k - budget)] + complement_ids)[: args.dense_k]
    final_recall = len(set(final_ids) & gold) / max(1, len(gold))
    missed_recovery = len(set(complement_ids) & missed) / len(missed) if missed else None
    hit = 1.0 if set(complement_ids) & missed else 0.0

    strata = ["ALL", f"gap_{example['bucket']}"]
    label = example["row"].get("label")
    strata.append("label_tie_or_invalid" if label is None else f"label_{int(label)}")
    if missed:
        strata.append("dense_miss")

    for stratum in strata:
        agg[stratum]["n"] += 1
        agg[stratum]["final_recall@5_sum"] += final_recall
        agg[stratum]["avg_complement_size_sum"] += len(complement_ids)
        if missed_recovery is not None:
            agg[stratum]["missed_n"] += 1
            agg[stratum]["missed_recovery_sum"] += missed_recovery
            agg[stratum]["complement_hit_sum"] += hit


def make_aggregators(budgets: list[int]) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        method: {
            budget: defaultdict(
                lambda: {
                    "n": 0,
                    "missed_n": 0,
                    "final_recall@5_sum": 0.0,
                    "missed_recovery_sum": 0.0,
                    "complement_hit_sum": 0.0,
                    "avg_complement_size_sum": 0.0,
                }
            )
            for budget in budgets
        }
        for method in ("hipporag_cached", "always_bridge", "oracle_A_gate", "predicted_A_gate")
    }


def summarize_aggregators(aggregators: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method, by_budget in aggregators.items():
        out[method] = {}
        for budget, by_stratum in by_budget.items():
            out[method][f"B={budget}"] = {}
            for stratum, vals in by_stratum.items():
                n = vals["n"]
                missed_n = vals["missed_n"]
                out[method][f"B={budget}"][stratum] = {
                    "n": int(n),
                    "missed_n": int(missed_n),
                    "final_recall@5": round(vals["final_recall@5_sum"] / n, 4) if n else None,
                    "missed_recovery": round(vals["missed_recovery_sum"] / missed_n, 4) if missed_n else None,
                    "complement_hit_rate": round(vals["complement_hit_sum"] / missed_n, 4) if missed_n else None,
                    "avg_complement_size": round(vals["avg_complement_size_sum"] / n, 4) if n else None,
                }
    return out


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


def splice_ids(dense_ids: list[str], complement_ids: list[str], budget: int, dense_k: int) -> list[str]:
    return dedupe(dense_ids[: max(0, dense_k - budget)] + complement_ids[:budget])[:dense_k]


def print_summary(reports: list[dict[str, Any]], args: argparse.Namespace) -> None:
    selected_budget = min(args.budgets, key=lambda value: abs(value - 3))
    key = f"B={selected_budget}"
    print(f"\n=== Gap-Aware Summary: {key} ===")
    for report in reports:
        print(
            f"\n{report['dataset']} source={report.get('gap_label_source')} "
            f"positive={report.get('gate_positive_class')} gate={report['gate_metrics']}"
        )
        for method in ("hipporag_cached", "always_bridge", "oracle_A_gate", "predicted_A_gate"):
            row = report["methods"][method][key].get("dense_miss")
            if row is None:
                print(f"{method}: no dense_miss rows recorded")
                continue
            print(
                f"{method}: finalR={row['final_recall@5']} "
                f"missedRec={row['missed_recovery']} hit={row['complement_hit_rate']}"
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


if __name__ == "__main__":
    main()
