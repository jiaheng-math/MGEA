"""Evaluate learned passage-level dense/graph evidence fusion.

This is a clean alternative second-contribution probe. It avoids the earlier
B-slot budget, A/B/C gap taxonomy, and oracle operator labels.

For each query:
  1. collect candidates from dense top-N and graph top-N
  2. featurize each passage by dense/graph rank, scores, overlap, and query probes
  3. train an OOF pointwise relevance model on gold passage ids
  4. rank candidates by predicted relevance and take top-k
  5. optionally select the final evidence set with a greedy coverage/diversity
     objective instead of a pure pointwise top-k

The method asks whether dense-first probes can calibrate dense and graph
evidence at the passage/set level, instead of relying on fixed source budgets
or hand-designed gap types.
"""
from __future__ import annotations

import argparse
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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - cloud env dependency.
    raise SystemExit("Missing scikit-learn in this environment.") from exc


DATASET_MAP = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki": "results/study_2wiki_hipporag_colbert_500",
    "nq": "results/study_nq_hipporag_colbert_500",
}


QUERY_FEATURES = [
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
    parser = argparse.ArgumentParser(description="Evaluate learned dense/graph passage fusion.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--training-target",
        choices=["all_gold", "dense_missed_gold"],
        default="all_gold",
        help=(
            "Candidate relevance target. all_gold learns generic support passage relevance. "
            "dense_missed_gold learns residual recovery for gold evidence absent from dense top-k."
        ),
    )
    parser.add_argument("--graph-anchors", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--deep-retrieval-files",
        nargs="*",
        default=[],
        help=(
            "Optional retrieval_results-style JSONL files with dense/graph top-N. "
            "Order must match --datasets/--result-dirs. If omitted, the saved "
            "result-dir retrieval_results.jsonl is used."
        ),
    )
    parser.add_argument(
        "--packing-score-weight",
        type=float,
        default=1.0,
        help="Weight on the learned pointwise relevance score inside greedy evidence packing.",
    )
    parser.add_argument("--packing-coverage-weight", type=float, default=0.06)
    parser.add_argument("--packing-redundancy-weight", type=float, default=0.04)
    parser.add_argument("--packing-graph-anchors", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--residual-gate-thresholds",
        type=float,
        nargs="*",
        default=[],
        help=(
            "For dense_missed_gold training, evaluate conservative graph-anchored residual replacement "
            "only when the residual score exceeds each fixed threshold."
        ),
    )
    parser.add_argument(
        "--swap-gate-thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Evaluate an OOF counterfactual one-swap policy at these confidence thresholds.",
    )
    parser.add_argument(
        "--include-oracle-swap",
        action="store_true",
        help="Report the best single replacement of one graph top-k slot as an upper bound.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", default="results/learned_dense_graph_fusion_eval.json")
    parser.add_argument("--per-sample-output", default="results/learned_dense_graph_fusion_per_sample.jsonl")
    parser.add_argument(
        "--generation-input-output",
        default=None,
        help="Optional JSONL generation input for batch_generate_from_retrieval.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    per_sample_all: list[dict[str, Any]] = []
    generation_rows_all: list[dict[str, Any]] = []
    dataset_dirs = resolve_dataset_dirs(args)
    deep_files = resolve_deep_retrieval_files(args, dataset_dirs)
    for dataset, result_dir in dataset_dirs:
        report, per_sample, generation_rows = evaluate_dataset(
            dataset=dataset,
            result_dir=result_dir,
            deep_retrieval_path=deep_files.get(dataset),
            args=args,
        )
        reports.append(report)
        per_sample_all.extend(per_sample)
        generation_rows_all.extend(generation_rows)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

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
        print(f"Generation input: {generation_output}")

    print_summary(reports)
    print(f"\nFull report: {output}")
    print(f"Per-sample report: {per_sample_output}")


def evaluate_dataset(
    dataset: str,
    result_dir: Path,
    deep_retrieval_path: Path | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    base_retrieval_by_id = {str(row["id"]): row for row in load_jsonl(result_dir / "retrieval_results.jsonl")}
    candidate_retrieval_by_id = base_retrieval_by_id
    candidate_source_path = result_dir / "retrieval_results.jsonl"
    if deep_retrieval_path is not None:
        candidate_source_path = deep_retrieval_path
        candidate_retrieval_by_id = {str(row["id"]): row for row in load_jsonl(deep_retrieval_path)}
    if args.max_queries:
        rows = rows[: args.max_queries]
    rows = [row for row in rows if row.get("gold_passage_ids") or row.get("gold_titles")]

    query_examples = [
        build_query_example(
            row=row,
            retrieval_row=candidate_retrieval_by_id.get(str(row["id"]), base_retrieval_by_id.get(str(row["id"]), {})),
            args=args,
        )
        for row in rows
    ]
    fusion_rankings, candidate_score_maps, model_metrics = train_oof_fusion(query_examples, args)
    swap_rankings, swap_metrics = train_oof_counterfactual_swap(query_examples, args)

    aggs = make_aggs()
    source_counts = Counter()
    depth_counts = Counter(
        (min(len(example["dense_ids"]), args.candidate_depth), min(len(example["graph_ids"]), args.candidate_depth))
        for example in query_examples
    )
    per_sample: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for example in query_examples:
        qid = example["id"]
        gold = example["gold"]
        learned_ranking = fusion_rankings[qid]
        score_map = candidate_score_maps[qid]
        learned_suffix = learned_method_suffix(args)
        packing_suffix = packing_method_suffix(args)
        rankings = {
            "dense_only": example["dense_ids"][: args.top_k],
            "graph_only": example["graph_ids"][: args.top_k],
            "rrf_dense_graph": rrf_merge(example["dense_ids"], example["graph_ids"], args)[: args.top_k],
            f"{learned_suffix}_fusion": learned_ranking[: args.top_k],
            "oracle_union": oracle_union_ranking(example, args)[: args.top_k],
            f"{packing_suffix}_packing": greedy_evidence_packing(example, score_map, [], args),
        }
        if args.include_oracle_swap:
            rankings["oracle_one_swap"] = oracle_one_swap_ranking(example, args)
        for method, by_query in swap_rankings.items():
            rankings[method] = by_query[qid]
        for anchor in args.graph_anchors:
            if anchor <= 0 or anchor >= args.top_k:
                continue
            rankings[f"graph_anchor{anchor}_{learned_suffix}"] = graph_anchored_ranking(example, learned_ranking, anchor, args)
        for anchor in args.packing_graph_anchors:
            if anchor <= 0 or anchor >= args.top_k:
                continue
            prefix = dedupe(example["graph_ids"])[:anchor]
            rankings[f"graph_anchor{anchor}_{packing_suffix}"] = greedy_evidence_packing(example, score_map, prefix, args)
        if args.training_target == "dense_missed_gold":
            for anchor in args.graph_anchors:
                if anchor <= 0 or anchor >= args.top_k:
                    continue
                for threshold in args.residual_gate_thresholds:
                    method = f"graph_anchor{anchor}_residual_gate_t{format_threshold(threshold)}"
                    rankings[method] = gated_residual_graph_ranking(
                        example=example,
                        residual_ranking=learned_ranking,
                        score_map=score_map,
                        anchor=anchor,
                        threshold=threshold,
                        args=args,
                    )
        strata = ["ALL"]
        if gold - set(example["dense_ids"][: args.top_k]):
            strata.append("dense_miss")
        label = example["row"].get("label")
        strata.append("label_tie_or_invalid" if label is None else f"label_{int(label)}")

        scores = {}
        for method, ids in rankings.items():
            score = score_ranking(ids, gold, set(example["dense_ids"][: args.top_k]))
            scores[method] = score
            for stratum in strata:
                update_agg(aggs[method][stratum], score)
            if "learned" in method or "residual" in method:
                for pid in ids:
                    source_counts[candidate_source(pid, example)] += 1

        per_sample.append(
            {
                "dataset": dataset,
                "id": qid,
                "dense_miss": "dense_miss" in strata,
                "rankings": rankings,
                "scores": scores,
            }
        )
        if args.generation_input_output:
            generation_rows.append(
                build_generation_row(
                    example=example,
                    retrieval_row=candidate_retrieval_by_id.get(qid, base_retrieval_by_id.get(qid, {})),
                    rankings=rankings,
                    args=args,
                )
            )

    return {
        "dataset": dataset,
        "n": len(query_examples),
        "candidate_depth": args.candidate_depth,
        "candidate_source": str(candidate_source_path),
        "observed_depth_counts": {f"dense{key[0]}_graph{key[1]}": value for key, value in depth_counts.items()},
        "top_k": args.top_k,
        "training_target": args.training_target,
        "model_metrics": model_metrics,
        "packing": {
            "score_weight": args.packing_score_weight,
            "coverage_weight": args.packing_coverage_weight,
            "redundancy_weight": args.packing_redundancy_weight,
            "graph_anchors": args.packing_graph_anchors,
            "residual_gate_thresholds": args.residual_gate_thresholds,
            "swap_gate_thresholds": args.swap_gate_thresholds,
        },
        "swap_model_metrics": swap_metrics,
        "metrics": summarize_aggs(aggs),
        "learned_fusion_source_counts": dict(source_counts),
    }, per_sample, generation_rows


def build_query_example(
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    retrieval = retrieval_row.get("retrieval", {}) if isinstance(retrieval_row, dict) else {}
    dense_passages = list(retrieval.get("dense") or [])[: args.candidate_depth]
    graph_passages = list(retrieval.get("graph") or [])[: args.candidate_depth]
    dense_ids = [str(passage.get("id")) for passage in dense_passages if passage.get("id") is not None]
    graph_ids = [str(passage.get("id")) for passage in graph_passages if passage.get("id") is not None]
    dense_scores = [float(passage.get("score", 0.0) or 0.0) for passage in dense_passages]
    graph_scores = [float(passage.get("score", 0.0) or 0.0) for passage in graph_passages]
    if not dense_ids:
        dense_ids = list(row.get("dense_ids") or [])[: args.candidate_depth]
        dense_scores = list(row.get("dense_scores") or [])[: args.candidate_depth]
    if not graph_ids:
        graph_ids = list(row.get("graph_ids") or [])[: args.candidate_depth]
        graph_scores = list(row.get("graph_scores") or [])[: args.candidate_depth]
    candidates = dedupe(dense_ids + graph_ids)
    lookup = passage_lookup(retrieval)
    dense_rank = rank_map(dense_ids)
    graph_rank = rank_map(graph_ids)
    dense_score_map = score_map(dense_ids, dense_scores)
    graph_score_map = score_map(graph_ids, graph_scores)
    dense_norm = normalize_scores(dense_ids, dense_scores)
    graph_norm = normalize_scores(graph_ids, graph_scores)
    gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
    target_gold = target_gold_for_row(gold, dense_ids, args)
    return {
        "id": str(row["id"]),
        "row": row,
        "gold": gold,
        "target_gold": target_gold,
        "dense_ids": dense_ids,
        "graph_ids": graph_ids,
        "candidates": candidates,
        "dense_rank": dense_rank,
        "graph_rank": graph_rank,
        "dense_score": dense_score_map,
        "graph_score": graph_score_map,
        "dense_norm": dense_norm,
        "graph_norm": graph_norm,
        "query_features": query_feature_values(row, args.top_k),
        "passage_lookup": lookup,
    }


def build_generation_row(
    example: dict[str, Any],
    retrieval_row: dict[str, Any],
    rankings: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lookup = passage_lookup(retrieval_row.get("retrieval", {}))
    method_order = [
        "dense_only",
        "graph_only",
        "rrf_dense_graph",
        "learned_fusion",
        "residual_fusion",
        "evidence_packing",
        "residual_packing",
        "graph_anchor1_learned",
        "graph_anchor2_learned",
        "graph_anchor3_learned",
        "graph_anchor4_learned",
        "graph_anchor1_residual",
        "graph_anchor2_residual",
        "graph_anchor3_residual",
        "graph_anchor4_residual",
        "graph_anchor1_packing",
        "graph_anchor2_packing",
        "graph_anchor3_packing",
        "graph_anchor4_packing",
        "graph_anchor1_residual_packing",
        "graph_anchor2_residual_packing",
        "graph_anchor3_residual_packing",
        "graph_anchor4_residual_packing",
    ]
    method_order = [method for method in method_order if method in rankings]
    method_order.extend(method for method in sorted(rankings) if method not in set(method_order))
    retrieval = {
        method: [lookup[pid] for pid in rankings[method][: args.top_k] if pid in lookup]
        for method in method_order
    }
    row = example["row"]
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "gold_answer": row.get("gold_answer", row.get("answer")),
        "gold_answers": row.get("gold_answers", [row.get("answer")]),
        "dataset_name": row.get("dataset_name"),
        "workload": row.get("workload"),
        "question_type": row.get("question_type"),
        "retrieval": retrieval,
        "main_table_methods": method_order,
        "main_table_decisions": {
            method: {"selected_path": method, "top_k": args.top_k}
            for method in method_order
        },
    }


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


def train_oof_fusion(
    query_examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict[str, Any]]:
    query_labels = [1 if ex["target_gold"] else 0 for ex in query_examples]
    n_splits = min(args.num_folds, min(Counter(query_labels).values())) if len(set(query_labels)) > 1 else args.num_folds
    n_splits = max(2, min(args.num_folds, n_splits))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)

    rankings: dict[str, list[str]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    y_true_all: list[int] = []
    y_score_all: list[float] = []
    for train_q_idx, test_q_idx in splitter.split(np.zeros(len(query_examples)), query_labels):
        x_train, y_train = flatten_candidates([query_examples[i] for i in train_q_idx])
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
            example = query_examples[idx]
            x_test = np.asarray([candidate_features(example, pid) for pid in example["candidates"]], dtype=np.float64)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(x_test)
                if probs.shape[1] == 1:
                    scores = np.full(len(example["candidates"]), float(clf.classes_[0]))
                else:
                    pos_col = list(clf.classes_).index(1)
                    scores = probs[:, pos_col]
            else:
                scores = clf.predict(x_test)
            ranked = [
                pid
                for pid, _ in sorted(
                    zip(example["candidates"], scores.tolist()),
                    key=lambda item: (-item[1], candidate_tiebreak(example, item[0])),
                )
            ]
            rankings[example["id"]] = ranked
            score_maps[example["id"]] = {
                pid: float(score)
                for pid, score in zip(example["candidates"], scores.tolist())
            }
            for pid, score in zip(example["candidates"], scores.tolist()):
                y_true_all.append(1 if pid in example["target_gold"] else 0)
                y_score_all.append(float(score))

    metrics = {
        "n_splits": n_splits,
        "candidate_positive_rate": round(float(np.mean(y_true_all)), 4) if y_true_all else None,
    }
    if len(set(y_true_all)) > 1:
        metrics["candidate_auc"] = round(float(roc_auc_score(y_true_all, y_score_all)), 4)
        metrics["candidate_ap"] = round(float(average_precision_score(y_true_all, y_score_all)), 4)
    return rankings, score_maps, metrics


def flatten_candidates(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for example in examples:
        for pid in example["candidates"]:
            x_rows.append(candidate_features(example, pid))
            y_rows.append(1 if pid in example["target_gold"] else 0)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def train_oof_counterfactual_swap(
    query_examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    thresholds = list(args.swap_gate_thresholds)
    if not thresholds:
        return {}, {}

    query_labels = [1 if has_positive_swap(example, args) else 0 for example in query_examples]
    n_splits = min(args.num_folds, min(Counter(query_labels).values())) if len(set(query_labels)) > 1 else args.num_folds
    n_splits = max(2, min(args.num_folds, n_splits))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)

    rankings_by_method: dict[str, dict[str, list[str]]] = {
        f"counterfactual_swap_t{format_threshold(threshold)}": {}
        for threshold in thresholds
    }
    y_true_all: list[int] = []
    y_score_all: list[float] = []

    for train_q_idx, test_q_idx in splitter.split(np.zeros(len(query_examples)), query_labels):
        x_train, y_train = flatten_swaps([query_examples[i] for i in train_q_idx], args)
        if len(y_train) == 0:
            for idx in test_q_idx:
                example = query_examples[idx]
                base = dedupe(example["graph_ids"])[: args.top_k]
                for by_query in rankings_by_method.values():
                    by_query[example["id"]] = base
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
            example = query_examples[idx]
            swaps = enumerate_swaps(example, args)
            base = dedupe(example["graph_ids"])[: args.top_k]
            if not swaps:
                for by_query in rankings_by_method.values():
                    by_query[example["id"]] = base
                continue

            x_test = np.asarray(
                [swap_features(example, evict_pid, candidate_pid, slot, args) for slot, evict_pid, candidate_pid in swaps],
                dtype=np.float64,
            )
            scores = positive_scores(clf, x_test)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_swap = swaps[best_idx]
            for threshold in thresholds:
                method = f"counterfactual_swap_t{format_threshold(threshold)}"
                if best_score >= threshold:
                    rankings_by_method[method][example["id"]] = apply_swap(base, best_swap)
                else:
                    rankings_by_method[method][example["id"]] = base
            for (slot, evict_pid, candidate_pid), score in zip(swaps, scores.tolist()):
                y_true_all.append(1 if swap_delta_positive(example, evict_pid, candidate_pid) else 0)
                y_score_all.append(float(score))

    metrics = {
        "n_splits": n_splits,
        "swap_positive_rate": round(float(np.mean(y_true_all)), 4) if y_true_all else None,
    }
    if len(set(y_true_all)) > 1:
        metrics["swap_auc"] = round(float(roc_auc_score(y_true_all, y_score_all)), 4)
        metrics["swap_ap"] = round(float(average_precision_score(y_true_all, y_score_all)), 4)
    return rankings_by_method, metrics


def flatten_swaps(examples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for example in examples:
        for slot, evict_pid, candidate_pid in enumerate_swaps(example, args):
            x_rows.append(swap_features(example, evict_pid, candidate_pid, slot, args))
            y_rows.append(1 if swap_delta_positive(example, evict_pid, candidate_pid) else 0)
    if not x_rows:
        width = len(candidate_features(examples[0], examples[0]["candidates"][0])) * 2 + 8 if examples and examples[0]["candidates"] else 8
        return np.zeros((0, width), dtype=np.float64), np.asarray([], dtype=np.int64)
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def positive_scores(clf: Any, x_rows: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_rows)
        if probs.shape[1] == 1:
            return np.full(x_rows.shape[0], float(clf.classes_[0]))
        pos_col = list(clf.classes_).index(1)
        return probs[:, pos_col]
    return clf.predict(x_rows)


def candidate_features(example: dict[str, Any], pid: str) -> list[float]:
    dense_rank = example["dense_rank"].get(pid)
    graph_rank = example["graph_rank"].get(pid)
    in_dense = 1.0 if dense_rank is not None else 0.0
    in_graph = 1.0 if graph_rank is not None else 0.0
    features = [
        in_dense,
        in_graph,
        1.0 if in_dense and in_graph else 0.0,
        reciprocal_rank(dense_rank),
        reciprocal_rank(graph_rank),
        float(dense_rank or 999),
        float(graph_rank or 999),
        example["dense_score"].get(pid, 0.0),
        example["graph_score"].get(pid, 0.0),
        example["dense_norm"].get(pid, 0.0),
        example["graph_norm"].get(pid, 0.0),
        example["graph_norm"].get(pid, 0.0) - example["dense_norm"].get(pid, 0.0),
    ]
    features.extend(example["query_features"])
    return features


def enumerate_swaps(example: dict[str, Any], args: argparse.Namespace) -> list[tuple[int, str, str]]:
    base = dedupe(example["graph_ids"])[: args.top_k]
    base_set = set(base)
    swaps = []
    for slot, evict_pid in enumerate(base):
        for candidate_pid in example["candidates"]:
            if candidate_pid in base_set:
                continue
            swaps.append((slot, evict_pid, candidate_pid))
    return swaps


def swap_features(
    example: dict[str, Any],
    evict_pid: str,
    candidate_pid: str,
    slot: int,
    args: argparse.Namespace,
) -> list[float]:
    candidate = candidate_features(example, candidate_pid)
    evicted = candidate_features(example, evict_pid)
    candidate_terms_set = candidate_terms(example, candidate_pid)
    evicted_terms_set = candidate_terms(example, evict_pid)
    base_without_slot_terms = set()
    for idx, pid in enumerate(dedupe(example["graph_ids"])[: args.top_k]):
        if idx != slot:
            base_without_slot_terms.update(candidate_terms(example, pid))
    coverage_gain = len(candidate_terms_set - base_without_slot_terms) / max(1, len(candidate_terms_set))
    slot_rank = slot + 1
    candidate_graph_rank = float(example["graph_rank"].get(candidate_pid, 999))
    evicted_graph_rank = float(example["graph_rank"].get(evict_pid, 999))
    extra = [
        float(slot_rank),
        reciprocal_rank(slot_rank),
        candidate_graph_rank - evicted_graph_rank,
        example["graph_norm"].get(candidate_pid, 0.0) - example["graph_norm"].get(evict_pid, 0.0),
        example["dense_norm"].get(candidate_pid, 0.0) - example["dense_norm"].get(evict_pid, 0.0),
        jaccard_terms(candidate_terms_set, evicted_terms_set),
        coverage_gain,
        1.0 if candidate_pid in example["dense_rank"] and candidate_pid not in example["graph_rank"] else 0.0,
    ]
    return candidate + evicted + extra


def has_positive_swap(example: dict[str, Any], args: argparse.Namespace) -> bool:
    return any(
        swap_delta_positive(example, evict_pid, candidate_pid)
        for _, evict_pid, candidate_pid in enumerate_swaps(example, args)
    )


def swap_delta_positive(example: dict[str, Any], evict_pid: str, candidate_pid: str) -> bool:
    gold = example["gold"]
    return bool(gold) and candidate_pid in gold and evict_pid not in gold


def apply_swap(base: list[str], swap: tuple[int, str, str]) -> list[str]:
    slot, _, candidate_pid = swap
    ranking = list(base)
    if 0 <= slot < len(ranking):
        ranking[slot] = candidate_pid
    return dedupe(ranking)[: len(base)]


def oracle_one_swap_ranking(example: dict[str, Any], args: argparse.Namespace) -> list[str]:
    base = dedupe(example["graph_ids"])[: args.top_k]
    best = list(base)
    best_score = score_ranking(best, example["gold"], set(example["dense_ids"][: args.top_k]))["recall"]
    for swap in enumerate_swaps(example, args):
        candidate = apply_swap(base, swap)
        score = score_ranking(candidate, example["gold"], set(example["dense_ids"][: args.top_k]))["recall"]
        if score > best_score:
            best = candidate
            best_score = score
    return best


def query_feature_values(row: dict[str, Any], top_k: int) -> list[float]:
    dense_ids = list(row.get("dense_ids") or [])[:top_k]
    graph_ids = list(row.get("graph_ids") or [])[:top_k]
    dense_set = set(dense_ids)
    graph_scores = [float(value) for value in (row.get("graph_scores") or [])[:top_k]]
    values = {
        "dense_graph_overlap": len(dense_set & set(graph_ids)) / max(1, top_k),
        "graph_top1_top2_gap": graph_scores[0] - graph_scores[1] if len(graph_scores) > 1 else 0.0,
        "graph_score_std": float(np.std(graph_scores)) if graph_scores else 0.0,
        "graph_new_doc_count": float(len([pid for pid in graph_ids if pid not in dense_set])),
    }
    for key in QUERY_FEATURES:
        values.setdefault(key, float(row.get(key, 0.0) or 0.0))
    return [float(values[key]) for key in QUERY_FEATURES]


def score_ranking(ids: list[str], gold: set[str], dense_top: set[str]) -> dict[str, Any]:
    selected = set(ids)
    missed = gold - dense_top
    recall = len(selected & gold) / max(1, len(gold))
    return {
        "recall": float(recall),
        "missed_recovery": len(selected & missed) / len(missed) if missed else None,
        "hit": 1.0 if missed and selected & missed else 0.0 if missed else None,
    }


def oracle_union_ranking(example: dict[str, Any], args: argparse.Namespace) -> list[str]:
    gold_first = [pid for pid in example["candidates"] if pid in example["gold"]]
    rest = rrf_merge(example["dense_ids"], example["graph_ids"], args)
    return dedupe(gold_first + rest)


def graph_anchored_ranking(example: dict[str, Any], learned_ranking: list[str], anchor: int, args: argparse.Namespace) -> list[str]:
    prefix = dedupe(example["graph_ids"])[:anchor]
    used = set(prefix)
    suffix = [pid for pid in learned_ranking if pid not in used]
    return (prefix + suffix)[: args.top_k]


def gated_residual_graph_ranking(
    example: dict[str, Any],
    residual_ranking: list[str],
    score_map: dict[str, float],
    anchor: int,
    threshold: float,
    args: argparse.Namespace,
) -> list[str]:
    selected = dedupe(example["graph_ids"])[:anchor]
    selected_set = set(selected)
    for pid in residual_ranking:
        if len(selected) >= args.top_k:
            break
        if pid in selected_set:
            continue
        if score_map.get(pid, 0.0) < threshold:
            break
        selected.append(pid)
        selected_set.add(pid)
    for pid in example["graph_ids"]:
        if len(selected) >= args.top_k:
            break
        if pid not in selected_set:
            selected.append(pid)
            selected_set.add(pid)
    return selected[: args.top_k]


def greedy_evidence_packing(
    example: dict[str, Any],
    score_map: dict[str, float],
    prefix: list[str],
    args: argparse.Namespace,
) -> list[str]:
    selected = dedupe([pid for pid in prefix if pid in example["candidates"]])[: args.top_k]
    selected_set = set(selected)
    covered_terms = set()
    for pid in selected:
        covered_terms.update(candidate_terms(example, pid))

    while len(selected) < args.top_k:
        best_pid = None
        best_value = None
        for pid in example["candidates"]:
            if pid in selected_set:
                continue
            terms = candidate_terms(example, pid)
            coverage_gain = len(terms - covered_terms) / max(1, len(terms))
            redundancy = max((jaccard_terms(terms, candidate_terms(example, old)) for old in selected), default=0.0)
            source_bonus = 0.015 if pid in example["dense_rank"] and pid in example["graph_rank"] else 0.0
            value = (
                args.packing_score_weight * score_map.get(pid, 0.0)
                + args.packing_coverage_weight * coverage_gain
                + source_bonus
                - args.packing_redundancy_weight * redundancy
            )
            key = (value, -float(example["graph_rank"].get(pid, 999)), -float(example["dense_rank"].get(pid, 999)), pid)
            if best_value is None or key > best_value:
                best_value = key
                best_pid = pid
        if best_pid is None:
            break
        selected.append(best_pid)
        selected_set.add(best_pid)
        covered_terms.update(candidate_terms(example, best_pid))
    return selected[: args.top_k]


def candidate_terms(example: dict[str, Any], pid: str) -> set[str]:
    cache = example.setdefault("_term_cache", {})
    if pid in cache:
        return cache[pid]
    passage = example.get("passage_lookup", {}).get(pid, {})
    text = " ".join(
        [
            str(passage.get("title") or pid),
            str(passage.get("text") or ""),
        ]
    )
    terms = content_terms(text)
    cache[pid] = terms
    return terms


def content_terms(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "by",
        "is", "are", "was", "were", "be", "been", "this", "that", "from", "as",
        "at", "it", "its", "which", "who", "what", "when", "where", "how",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop
    }


def jaccard_terms(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def rrf_merge(dense_ids: list[str], graph_ids: list[str], args: argparse.Namespace) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    first_seen = {}
    counter = 0
    for source in (dense_ids, graph_ids):
        for rank, pid in enumerate(source, start=1):
            first_seen.setdefault(pid, counter)
            counter += 1
            scores[pid] += 1.0 / (60.0 + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], first_seen[pid]))


def update_agg(agg: dict[str, Any], score: dict[str, Any]) -> None:
    agg["n"] += 1
    agg["recall_sum"] += score["recall"]
    if score["missed_recovery"] is not None:
        agg["missed_n"] += 1
        agg["missed_recovery_sum"] += score["missed_recovery"]
        agg["hit_sum"] += score["hit"]


def make_aggs() -> dict[str, dict[str, dict[str, Any]]]:
    return defaultdict(lambda: defaultdict(lambda: {"n": 0, "missed_n": 0, "recall_sum": 0.0, "missed_recovery_sum": 0.0, "hit_sum": 0.0}))


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
                "recall@5": round(agg["recall_sum"] / n, 4) if n else None,
                "missed_recovery": round(agg["missed_recovery_sum"] / missed_n, 4) if missed_n else None,
                "hit_rate": round(agg["hit_sum"] / missed_n, 4) if missed_n else None,
            }
    return out


def print_summary(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        print("\n" + "=" * 88)
        print(f"DATASET {report['dataset']} n={report['n']} depth={report['candidate_depth']}")
        print(f"candidate_source={report.get('candidate_source')}")
        print(f"observed_depth_counts={report.get('observed_depth_counts')}")
        print(f"training_target={report.get('training_target')}")
        print(f"model={report['model_metrics']}")
        if report.get("swap_model_metrics"):
            print(f"swap_model={report['swap_model_metrics']}")
        print(f"packing={report.get('packing')}")
        print(f"learned source counts={report['learned_fusion_source_counts']}")
        for stratum in ("ALL", "dense_miss"):
            print(f"\n{stratum}")
            rows = []
            for method, by_stratum in report["metrics"].items():
                row = by_stratum.get(stratum)
                if row:
                    rows.append((method, row))
            rows.sort(key=lambda item: item[1]["recall@5"] or 0.0, reverse=True)
            for method, row in rows:
                print(
                    f"{method}: R={row['recall@5']} missedRec={row['missed_recovery']} "
                    f"hit={row['hit_rate']}"
                )


def candidate_source(pid: str, example: dict[str, Any]) -> str:
    in_dense = pid in example["dense_rank"]
    in_graph = pid in example["graph_rank"]
    if in_dense and in_graph:
        return "both"
    if in_dense:
        return "dense_only_candidate"
    if in_graph:
        return "graph_only_candidate"
    return "unknown"


def target_gold_for_row(gold: set[str], dense_ids: list[str], args: argparse.Namespace) -> set[str]:
    if args.training_target == "all_gold":
        return set(gold)
    if args.training_target == "dense_missed_gold":
        return set(gold) - set(dense_ids[: args.top_k])
    raise ValueError(f"Unsupported training target: {args.training_target}")


def learned_method_suffix(args: argparse.Namespace) -> str:
    return "learned" if args.training_target == "all_gold" else "residual"


def packing_method_suffix(args: argparse.Namespace) -> str:
    return "evidence" if args.training_target == "all_gold" else "residual"


def format_threshold(threshold: float) -> str:
    text = f"{threshold:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def candidate_tiebreak(example: dict[str, Any], pid: str) -> tuple[float, float, str]:
    return (
        float(example["graph_rank"].get(pid, 999)),
        float(example["dense_rank"].get(pid, 999)),
        pid,
    )


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / float(rank)


def rank_map(ids: list[str]) -> dict[str, int]:
    out = {}
    for idx, pid in enumerate(ids, start=1):
        out.setdefault(pid, idx)
    return out


def score_map(ids: list[str], scores: list[Any]) -> dict[str, float]:
    return {pid: float(scores[idx]) for idx, pid in enumerate(ids) if idx < len(scores)}


def normalize_scores(ids: list[str], scores: list[Any]) -> dict[str, float]:
    if not ids or not scores:
        return {}
    arr = np.asarray([float(value) for value in scores[: len(ids)]], dtype=np.float64)
    if arr.size == 0:
        return {}
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        normed = np.ones_like(arr)
    else:
        normed = (arr - lo) / (hi - lo)
    return {pid: float(normed[idx]) for idx, pid in enumerate(ids[: len(normed)])}


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_dataset_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.result_dirs:
        return [(Path(path).name, resolve_project_path(path)) for path in args.result_dirs]
    return [(dataset, resolve_project_path(DATASET_MAP[dataset])) for dataset in args.datasets]


def resolve_deep_retrieval_files(
    args: argparse.Namespace,
    dataset_dirs: list[tuple[str, Path]],
) -> dict[str, Path]:
    if not args.deep_retrieval_files:
        return {}
    if len(args.deep_retrieval_files) != len(dataset_dirs):
        raise ValueError("--deep-retrieval-files must have the same length/order as datasets/result dirs.")
    return {
        dataset: resolve_project_path(path)
        for (dataset, _), path in zip(dataset_dirs, args.deep_retrieval_files)
    }


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
