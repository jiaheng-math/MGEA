"""Evaluate learned evidence-budget correction policies.

This script tests whether correction should be treated as a budget-selection
problem instead of a fixed dense-vs-graph hard switch.

Actions are graph evidence budgets B in {0..K}:
  B=0: dense top-K
  B>0: dense[:K-B] + first B graph passages not already kept from dense

It reports:
  - fixed-B baselines
  - oracle best-B upper bound
  - learned pre-graph budget policy using query+dense probe features only
  - learned post-graph budget policy using query+dense+graph features
  - two-stage policy: pre-graph gap detector, then post-graph budget selector
  - matched random policies

The post-graph policy is an evidence-packing upper bound: it assumes graph has
already been invoked. The two-stage policy is the operational version.

Usage:
  python scripts/eval_budgeted_correction.py \
    --datasets hotpot 2wiki nq \
    --output results/budgeted_correction_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError as exc:  # pragma: no cover - cloud env should have sklearn.
    raise SystemExit(
        "Missing scikit-learn. Run this script in the graph-routing cloud "
        "environment where the project training scripts already work."
    ) from exc

from src.features import probe_feature_names, query_feature_names


DATASETS = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki": "results/study_2wiki_hipporag_colbert_500",
    "nq": "results/study_nq_hipporag_colbert_500",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate budgeted correction policies.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki", "nq"])
    parser.add_argument(
        "--result-dirs",
        nargs="*",
        default=[],
        help="Optional explicit result directories. If set, --datasets is ignored.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-prob-key", default="query_plus_probe_probability")
    parser.add_argument(
        "--overlap-thresholds",
        default="0.2,0.6",
        help=(
            "Low,high dense-graph overlap thresholds for the heuristic budget baseline. "
            "Under the OOF router gate: overlap >= high -> B=1; overlap >= low -> B=3; "
            "otherwise B=5."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=["all", "valid"],
        default="all",
        help="Use all rows with gold evidence, or only rows with non-null correction labels.",
    )
    parser.add_argument(
        "--actions",
        default="0,1,2,3,4,5",
        help="Comma-separated allowed B actions. Must include 0.",
    )
    parser.add_argument(
        "--slot-penalties",
        default="0,0.002,0.005,0.01,0.02,0.05",
        help=(
            "Comma-separated penalties subtracted per graph slot for value-model "
            "policies: utility = predicted_recall - penalty * B."
        ),
    )
    parser.add_argument("--output", default="results/budgeted_correction_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    actions = sorted({int(item.strip()) for item in args.actions.split(",") if item.strip()})
    if 0 not in actions:
        raise ValueError("--actions must include 0.")
    if any(action < 0 or action > args.top_k for action in actions):
        raise ValueError(f"All actions must be in [0, {args.top_k}]. Got {actions}.")
    slot_penalties = parse_float_list(args.slot_penalties)
    overlap_thresholds = parse_float_list(args.overlap_thresholds)
    if len(overlap_thresholds) != 2 or overlap_thresholds[0] > overlap_thresholds[1]:
        raise ValueError("--overlap-thresholds must be two ascending floats, e.g. 0.2,0.6")

    result_dirs = [resolve_project_path(p) for p in args.result_dirs]
    if not result_dirs:
        result_dirs = [resolve_project_path(DATASETS[name]) for name in args.datasets]

    reports = []
    for result_dir in result_dirs:
        if not result_dir.exists():
            print(f"skip missing result dir: {result_dir}", file=sys.stderr)
            continue
        reports.append(
            evaluate_dataset(
                result_dir=result_dir,
                top_k=args.top_k,
                actions=actions,
                num_folds=args.num_folds,
                random_seed=args.random_seed,
                scope=args.scope,
                slot_penalties=slot_penalties,
                router_threshold=args.router_threshold,
                router_prob_key=args.router_prob_key,
                overlap_thresholds=tuple(overlap_thresholds),
            )
        )

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)

    print_summary(reports, actions)
    print(f"\nFull report: {output}")


def evaluate_dataset(
    result_dir: Path,
    top_k: int,
    actions: list[int],
    num_folds: int,
    random_seed: int,
    scope: str,
    slot_penalties: list[float],
    router_threshold: float,
    router_prob_key: str,
    overlap_thresholds: tuple[float, float],
) -> dict[str, Any]:
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_path = result_dir / "retrieval_results.jsonl"
    retrieval = {row["id"]: row for row in load_jsonl(retrieval_path)} if retrieval_path.exists() else {}

    rows = [row for row in routing if row.get("gold_passage_ids")]
    if scope == "valid":
        rows = [row for row in rows if row.get("label") is not None]
    if not rows:
        raise ValueError(f"No evaluable rows in {result_dir}")

    oracle_actions = [best_action(row, actions, top_k) for row in rows]
    oracle_b = [item[0] for item in oracle_actions]
    oracle_recall = [item[1] for item in oracle_actions]

    pre_names = query_feature_names() + probe_feature_names()
    post_names = pre_names + graph_feature_names(top_k)
    feature_rows = [attach_graph_features(row, top_k) for row in rows]
    oof_gate = load_oof_gate(result_dir, rows, router_prob_key, router_threshold)

    fixed = {
        f"fixed_B={b}": evaluate_action_predictions(rows, [b] * len(rows), top_k, retrieval)
        for b in actions
    }
    oracle = evaluate_action_predictions(rows, oracle_b, top_k, retrieval)
    oracle["action_distribution"] = dict(Counter(oracle_b))

    learned_pre = cross_validated_budget_policy(
        rows=feature_rows,
        labels=oracle_b,
        feature_names=pre_names,
        actions=actions,
        top_k=top_k,
        num_folds=num_folds,
        random_seed=random_seed,
        retrieval=retrieval,
    )
    learned_post = cross_validated_budget_policy(
        rows=feature_rows,
        labels=oracle_b,
        feature_names=post_names,
        actions=actions,
        top_k=top_k,
        num_folds=num_folds,
        random_seed=random_seed,
        retrieval=retrieval,
    )
    learned_post["requires_graph_features"] = True
    learned_post["operational_graph_invocation_rate"] = 1.0
    two_stage = cross_validated_two_stage_policy(
        rows=feature_rows,
        labels=oracle_b,
        pre_feature_names=pre_names,
        post_feature_names=post_names,
        actions=actions,
        top_k=top_k,
        num_folds=num_folds,
        random_seed=random_seed,
        retrieval=retrieval,
    )
    value_policies: dict[str, Any] = {}
    for penalty in slot_penalties:
        suffix = format_penalty(penalty)
        value_pre = cross_validated_value_policy(
            rows=feature_rows,
            feature_names=pre_names,
            actions=actions,
            top_k=top_k,
            num_folds=num_folds,
            random_seed=random_seed,
            slot_penalty=penalty,
            retrieval=retrieval,
        )
        value_post = cross_validated_value_policy(
            rows=feature_rows,
            feature_names=post_names,
            actions=actions,
            top_k=top_k,
            num_folds=num_folds,
            random_seed=random_seed,
            slot_penalty=penalty,
            retrieval=retrieval,
        )
        value_post["requires_graph_features"] = True
        value_post["operational_graph_invocation_rate"] = 1.0
        value_two_stage = cross_validated_two_stage_value_policy(
            rows=feature_rows,
            pre_feature_names=pre_names,
            post_feature_names=post_names,
            actions=actions,
            top_k=top_k,
            num_folds=num_folds,
            random_seed=random_seed,
            slot_penalty=penalty,
            retrieval=retrieval,
        )
        value_policies[f"value_pre_graph_slot_penalty={suffix}"] = strip_predictions(value_pre)
        value_policies[f"value_post_graph_upper_bound_slot_penalty={suffix}"] = strip_predictions(value_post)
        value_policies[f"value_two_stage_slot_penalty={suffix}"] = strip_predictions(value_two_stage)
        if oof_gate is not None:
            router_gate_value = cross_validated_fixed_gate_value_policy(
                rows=feature_rows,
                gate_decisions=oof_gate,
                feature_names=post_names,
                actions=actions,
                top_k=top_k,
                num_folds=num_folds,
                random_seed=random_seed,
                slot_penalty=penalty,
                retrieval=retrieval,
            )
            router_gate_value["router_prob_key"] = router_prob_key
            router_gate_value["router_threshold"] = router_threshold
            value_policies[f"oof_router_gate_value_budget_slot_penalty={suffix}"] = strip_predictions(router_gate_value)

    random_pre = random_matched_policy(
        rows=rows,
        reference_predictions=learned_pre["predicted_actions"],
        actions=actions,
        top_k=top_k,
        random_seed=random_seed,
        retrieval=retrieval,
    )
    random_two_stage = random_matched_policy(
        rows=rows,
        reference_predictions=two_stage["predicted_actions"],
        actions=actions,
        top_k=top_k,
        random_seed=random_seed + 1,
        retrieval=retrieval,
    )
    oof_gate_fixed: dict[str, Any] = {}
    if oof_gate is not None:
        for action in actions:
            predictions = [action if choose_graph else 0 for choose_graph in oof_gate]
            report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
            report["router_prob_key"] = router_prob_key
            report["router_threshold"] = router_threshold
            report["router_gate_rate"] = round(sum(oof_gate) / len(oof_gate), 6)
            oof_gate_fixed[f"oof_router_gate_fixed_B={action}"] = report
        overlap_predictions = [
            overlap_budget_action(row, choose_graph, top_k, overlap_thresholds)
            for row, choose_graph in zip(rows, oof_gate)
        ]
        overlap_report = evaluate_action_predictions(rows, overlap_predictions, top_k, retrieval)
        overlap_report["router_prob_key"] = router_prob_key
        overlap_report["router_threshold"] = router_threshold
        overlap_report["router_gate_rate"] = round(sum(oof_gate) / len(oof_gate), 6)
        overlap_report["overlap_thresholds"] = list(overlap_thresholds)
        oof_gate_fixed["oof_router_gate_overlap_heuristic"] = overlap_report

    return {
        "dataset": result_dir.name,
        "result_dir": str(result_dir),
        "scope": scope,
        "top_k": top_k,
        "actions": actions,
        "n": len(rows),
        "label_distribution": label_distribution(rows),
        "oracle_best_action_distribution": dict(Counter(oracle_b)),
        "oracle_mean_recall": round(mean(oracle_recall), 6),
        "oof_router_gate_available": oof_gate is not None,
        "oof_router_gate_rate": round(sum(oof_gate) / len(oof_gate), 6) if oof_gate is not None else None,
        "methods": {
            **fixed,
            "oracle_budget": oracle,
            "learned_pre_graph_budget": strip_predictions(learned_pre),
            "learned_post_graph_budget_upper_bound": strip_predictions(learned_post),
            "two_stage_pre_gap_post_budget": strip_predictions(two_stage),
            **oof_gate_fixed,
            **value_policies,
            "random_matched_pre_graph_budget": random_pre,
            "random_matched_two_stage": random_two_stage,
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_oof_gate(
    result_dir: Path,
    rows: list[dict[str, Any]],
    prob_key: str,
    threshold: float,
) -> list[bool] | None:
    path = result_dir / "pareto_strict_cv" / "oof_predictions.jsonl"
    if not path.exists():
        return None
    oof = {str(row["id"]): row for row in load_jsonl(path)}
    out = []
    for row in rows:
        prob = oof.get(str(row["id"]), {}).get(prob_key)
        out.append(False if prob is None else float(prob) >= threshold)
    return out


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_penalty(value: float) -> str:
    return f"{value:g}"


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def recall_at_k(pred_ids: list[str], gold_ids: set[str], top_k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(pred_ids[:top_k]) & gold_ids) / len(gold_ids)


def ranking_for_action(row: dict[str, Any], budget: int, top_k: int) -> list[str]:
    dense_ids = list(row.get("dense_ids") or [])
    graph_ids = list(row.get("graph_ids") or [])
    if budget <= 0:
        return dense_ids[:top_k]

    keep_dense = dense_ids[: max(0, top_k - budget)]
    used = set(keep_dense)
    graph_extras = [pid for pid in graph_ids if pid not in used]
    ranking = keep_dense + graph_extras[:budget]

    deduped = []
    seen = set()
    for pid in ranking:
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(pid)
        if len(deduped) >= top_k:
            break
    return deduped


def best_action(row: dict[str, Any], actions: list[int], top_k: int) -> tuple[int, float]:
    gold = set(row.get("gold_passage_ids") or [])
    scored = [
        (recall_at_k(ranking_for_action(row, action, top_k), gold, top_k), -action, action)
        for action in actions
    ]
    best_recall, _, action = max(scored)
    return action, best_recall


def graph_feature_names(top_k: int) -> list[str]:
    return [
        "graph_top1_score",
        "graph_top1_top2_gap",
        "graph_topk_score_std",
        "graph_top1_over_sum_topk",
        "dense_graph_overlap_ratio",
        "graph_new_count",
        "graph_dense_rank_disagreement",
    ]


def attach_graph_features(row: dict[str, Any], top_k: int) -> dict[str, Any]:
    payload = dict(row)
    graph_scores = [float(value) for value in (row.get("graph_scores") or [])[:top_k]]
    dense_ids = list(row.get("dense_ids") or [])[:top_k]
    graph_ids = list(row.get("graph_ids") or [])[:top_k]
    dense_rank = {pid: idx for idx, pid in enumerate(dense_ids, start=1)}
    graph_rank = {pid: idx for idx, pid in enumerate(graph_ids, start=1)}
    overlap = set(dense_ids) & set(graph_ids)

    payload["graph_top1_score"] = graph_scores[0] if graph_scores else 0.0
    payload["graph_top1_top2_gap"] = (
        graph_scores[0] - graph_scores[1] if len(graph_scores) >= 2 else 0.0
    )
    payload["graph_topk_score_std"] = float(np.std(graph_scores)) if graph_scores else 0.0
    score_sum = sum(graph_scores)
    payload["graph_top1_over_sum_topk"] = graph_scores[0] / score_sum if score_sum > 0 else 0.0
    payload["dense_graph_overlap_ratio"] = len(overlap) / max(1, top_k)
    payload["graph_new_count"] = len([pid for pid in graph_ids if pid not in set(dense_ids)])
    if overlap:
        payload["graph_dense_rank_disagreement"] = mean(
            abs(dense_rank[pid] - graph_rank[pid]) for pid in overlap
        )
    else:
        payload["graph_dense_rank_disagreement"] = float(top_k)
    return payload


def cross_validated_budget_policy(
    rows: list[dict[str, Any]],
    labels: list[int],
    feature_names: list[str],
    actions: list[int],
    top_k: int,
    num_folds: int,
    random_seed: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    predictions = [0] * len(rows)
    for train_idx, test_idx in make_folds(labels, num_folds, random_seed):
        y_train = [labels[i] for i in train_idx]
        model = make_classifier(y_train, random_seed)
        model.fit(matrix(rows, train_idx, feature_names), y_train)
        pred = model.predict(matrix(rows, test_idx, feature_names)).tolist()
        for idx, action in zip(test_idx, pred):
            predictions[idx] = int(nearest_action(int(action), actions))
    report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
    report["predicted_action_distribution"] = dict(Counter(predictions))
    report["predicted_actions"] = predictions
    return report


def cross_validated_two_stage_policy(
    rows: list[dict[str, Any]],
    labels: list[int],
    pre_feature_names: list[str],
    post_feature_names: list[str],
    actions: list[int],
    top_k: int,
    num_folds: int,
    random_seed: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    positive_actions = [action for action in actions if action > 0]
    if not positive_actions:
        raise ValueError("two-stage policy requires at least one positive action.")

    predictions = [0] * len(rows)
    gap_labels = [1 if label > 0 else 0 for label in labels]

    for train_idx, test_idx in make_folds(gap_labels, num_folds, random_seed):
        y_gap_train = [gap_labels[i] for i in train_idx]
        gap_model = make_classifier(y_gap_train, random_seed)
        gap_model.fit(matrix(rows, train_idx, pre_feature_names), y_gap_train)
        gap_pred = gap_model.predict(matrix(rows, test_idx, pre_feature_names)).tolist()

        positive_train_idx = [i for i in train_idx if labels[i] > 0]
        if positive_train_idx:
            y_budget_train = [labels[i] for i in positive_train_idx]
            budget_model = make_classifier(y_budget_train, random_seed)
            budget_model.fit(matrix(rows, positive_train_idx, post_feature_names), y_budget_train)
            budget_pred = budget_model.predict(matrix(rows, test_idx, post_feature_names)).tolist()
        else:
            budget_pred = [min(positive_actions)] * len(test_idx)

        for idx, wants_graph, action in zip(test_idx, gap_pred, budget_pred):
            if int(wants_graph) <= 0:
                predictions[idx] = 0
            else:
                predictions[idx] = int(nearest_action(max(1, int(action)), positive_actions))

    report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
    report["predicted_action_distribution"] = dict(Counter(predictions))
    report["predicted_actions"] = predictions
    return report


def cross_validated_value_policy(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    actions: list[int],
    top_k: int,
    num_folds: int,
    random_seed: int,
    slot_penalty: float,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    labels = [best_action(row, actions, top_k)[0] for row in rows]
    predictions = [0] * len(rows)
    for train_idx, test_idx in make_folds(labels, num_folds, random_seed):
        models = train_action_value_models(
            rows=rows,
            indices=train_idx,
            feature_names=feature_names,
            actions=actions,
            top_k=top_k,
            random_seed=random_seed,
        )
        pred = predict_actions_from_value_models(
            models=models,
            rows=rows,
            indices=test_idx,
            feature_names=feature_names,
            actions=actions,
            slot_penalty=slot_penalty,
        )
        for idx, action in zip(test_idx, pred):
            predictions[idx] = int(action)

    report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
    report["slot_penalty"] = slot_penalty
    report["predicted_action_distribution"] = dict(Counter(predictions))
    report["predicted_actions"] = predictions
    return report


def cross_validated_two_stage_value_policy(
    rows: list[dict[str, Any]],
    pre_feature_names: list[str],
    post_feature_names: list[str],
    actions: list[int],
    top_k: int,
    num_folds: int,
    random_seed: int,
    slot_penalty: float,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    labels = [best_action(row, actions, top_k)[0] for row in rows]
    gap_labels = [1 if label > 0 else 0 for label in labels]
    positive_actions = [action for action in actions if action > 0]
    predictions = [0] * len(rows)

    for train_idx, test_idx in make_folds(gap_labels, num_folds, random_seed):
        y_gap_train = [gap_labels[i] for i in train_idx]
        gap_model = make_classifier(y_gap_train, random_seed)
        gap_model.fit(matrix(rows, train_idx, pre_feature_names), y_gap_train)
        gap_pred = gap_model.predict(matrix(rows, test_idx, pre_feature_names)).tolist()

        positive_train_idx = [i for i in train_idx if labels[i] > 0]
        if positive_train_idx:
            models = train_action_value_models(
                rows=rows,
                indices=positive_train_idx,
                feature_names=post_feature_names,
                actions=positive_actions,
                top_k=top_k,
                random_seed=random_seed,
            )
            budget_pred = predict_actions_from_value_models(
                models=models,
                rows=rows,
                indices=test_idx,
                feature_names=post_feature_names,
                actions=positive_actions,
                slot_penalty=slot_penalty,
            )
        else:
            budget_pred = [min(positive_actions)] * len(test_idx)

        for idx, wants_graph, action in zip(test_idx, gap_pred, budget_pred):
            predictions[idx] = int(action) if int(wants_graph) > 0 else 0

    report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
    report["slot_penalty"] = slot_penalty
    report["predicted_action_distribution"] = dict(Counter(predictions))
    report["predicted_actions"] = predictions
    return report


def cross_validated_fixed_gate_value_policy(
    rows: list[dict[str, Any]],
    gate_decisions: list[bool],
    feature_names: list[str],
    actions: list[int],
    top_k: int,
    num_folds: int,
    random_seed: int,
    slot_penalty: float,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    labels = [best_action(row, actions, top_k)[0] for row in rows]
    predictions = [0] * len(rows)

    for train_idx, test_idx in make_folds(labels, num_folds, random_seed):
        models = train_action_value_models(
            rows=rows,
            indices=train_idx,
            feature_names=feature_names,
            actions=actions,
            top_k=top_k,
            random_seed=random_seed,
        )
        budget_pred = predict_actions_from_value_models(
            models=models,
            rows=rows,
            indices=test_idx,
            feature_names=feature_names,
            actions=actions,
            slot_penalty=slot_penalty,
        )
        for idx, action in zip(test_idx, budget_pred):
            predictions[idx] = int(action) if gate_decisions[idx] else 0

    report = evaluate_action_predictions(rows, predictions, top_k, retrieval)
    report["slot_penalty"] = slot_penalty
    report["router_gate_rate"] = round(sum(gate_decisions) / len(gate_decisions), 6)
    report["predicted_action_distribution"] = dict(Counter(predictions))
    report["predicted_actions"] = predictions
    return report


def train_action_value_models(
    rows: list[dict[str, Any]],
    indices: list[int],
    feature_names: list[str],
    actions: list[int],
    top_k: int,
    random_seed: int,
) -> dict[int, Any]:
    X = matrix(rows, indices, feature_names)
    models = {}
    for action in actions:
        y = np.asarray(
            [
                recall_at_k(
                    ranking_for_action(rows[idx], action, top_k),
                    set(rows[idx].get("gold_passage_ids") or []),
                    top_k,
                )
                for idx in indices
            ],
            dtype=np.float64,
        )
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=8,
            random_state=random_seed + int(action),
            n_jobs=-1,
        )
        model.fit(X, y)
        models[action] = model
    return models


def predict_actions_from_value_models(
    models: dict[int, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    feature_names: list[str],
    actions: list[int],
    slot_penalty: float,
) -> list[int]:
    X = matrix(rows, indices, feature_names)
    predicted_values = {
        action: models[action].predict(X) - slot_penalty * float(action)
        for action in actions
    }
    outputs = []
    for row_idx in range(len(indices)):
        scored = [(predicted_values[action][row_idx], -action, action) for action in actions]
        outputs.append(max(scored)[2])
    return outputs


def make_folds(labels: list[int], num_folds: int, random_seed: int) -> list[tuple[list[int], list[int]]]:
    counts = Counter(labels)
    if len(counts) < 2:
        idx = list(range(len(labels)))
        return [(idx, idx)]
    n_splits = min(num_folds, min(counts.values()))
    if n_splits < 2:
        idx = list(range(len(labels)))
        return [(idx, idx)]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    indices = np.arange(len(labels))
    y = np.asarray(labels)
    return [
        (indices[train_idx].tolist(), indices[test_idx].tolist())
        for train_idx, test_idx in splitter.split(indices, y)
    ]


def make_classifier(labels: list[int], random_seed: int):
    if len(set(labels)) < 2:
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    random_state=random_seed,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def matrix(rows: list[dict[str, Any]], indices: list[int], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[finite_float(rows[idx].get(name, 0.0)) for name in feature_names] for idx in indices],
        dtype=np.float64,
    )


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def nearest_action(action: int, actions: list[int]) -> int:
    return min(actions, key=lambda candidate: (abs(candidate - action), candidate))


def evaluate_action_predictions(
    rows: list[dict[str, Any]],
    actions: list[int],
    top_k: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    recalls = []
    bucket_recalls: dict[str, list[float]] = defaultdict(list)
    for row, action in zip(rows, actions):
        gold = set(row.get("gold_passage_ids") or [])
        ranking = ranking_for_action(row, int(action), top_k)
        value = recall_at_k(ranking, gold, top_k)
        recalls.append(value)
        bucket_recalls[bucket_for_row(row, retrieval, top_k)].append(value)

    positive_actions = [action for action in actions if action > 0]
    return {
        "recall@k": round(mean(recalls), 6),
        "graph_invocation_rate": round(len(positive_actions) / len(actions), 6),
        "avg_graph_slots": round(mean(actions), 6),
        "action_distribution": dict(Counter(actions)),
        "bucket_breakdown": {
            bucket: {"n": len(values), "recall@k": round(mean(values), 6)}
            for bucket, values in sorted(bucket_recalls.items())
            if values
        },
    }


def random_matched_policy(
    rows: list[dict[str, Any]],
    reference_predictions: list[int],
    actions: list[int],
    top_k: int,
    random_seed: int,
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    distribution = Counter(reference_predictions)
    sampled = []
    population = list(distribution)
    weights = [distribution[action] for action in population]
    for _ in rows:
        sampled.append(int(rng.choices(population, weights=weights, k=1)[0]))
    report = evaluate_action_predictions(rows, sampled, top_k, retrieval)
    report["matched_to_distribution"] = dict(distribution)
    return report


def overlap_budget_action(
    row: dict[str, Any],
    choose_graph: bool,
    top_k: int,
    thresholds: tuple[float, float],
) -> int:
    if not choose_graph:
        return 0
    low, high = thresholds
    dense_ids = set((row.get("dense_ids") or [])[:top_k])
    graph_ids = set((row.get("graph_ids") or [])[:top_k])
    overlap = len(dense_ids & graph_ids) / max(1, top_k)
    if overlap >= high:
        return 1
    if overlap >= low:
        return min(3, top_k)
    return top_k


def bucket_for_row(row: dict[str, Any], retrieval: dict[str, Any], top_k: int) -> str:
    label = row.get("label")
    if label == 0:
        return "label0_dense_sufficient"

    gold = set(row.get("gold_passage_ids") or [])
    dense_top = set((row.get("dense_ids") or [])[:top_k])
    missed = gold - dense_top
    if not missed:
        return "label1_no_miss"

    dense_passages = retrieval.get(row["id"], {}).get("retrieval", {}).get("dense", [])[:top_k]
    dense_text = norm(" ".join((p.get("text") or "") + " " + (p.get("title") or "") for p in dense_passages))
    question = norm(row.get("question") or "")
    kinds = set()
    for title in missed:
        title_norm = norm(title)
        if title_norm and title_norm in question:
            kinds.add("C_query_entity_miss")
        elif title_norm and title_norm in dense_text:
            kinds.add("A_bridge_visible")
        else:
            kinds.add("B_hop1_miss")
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter("None" if row.get("label") is None else str(row.get("label")) for row in rows)
    return dict(counts)


def strip_predictions(report: dict[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.pop("predicted_actions", None)
    return payload


def print_summary(reports: list[dict[str, Any]], actions: list[int]) -> None:
    print("\n================ Budgeted Correction Summary ================")
    method_order = [
        *(f"fixed_B={action}" for action in actions),
        "oracle_budget",
        "learned_pre_graph_budget",
        "learned_post_graph_budget_upper_bound",
        "two_stage_pre_gap_post_budget",
    ]
    for report in reports:
        print(f"\n{report['dataset']}  n={report['n']}  scope={report['scope']}")
        print(f"oracle best-B dist: {report['oracle_best_action_distribution']}")
        print(f"{'method':<40s} {'R@k':>8s} {'graph%':>8s} {'avgB':>8s} actions")
        extra_methods = sorted(
            name for name in report["methods"]
            if name.startswith("value_")
            or name.startswith("random_")
            or name.startswith("oof_")
        )
        for method in method_order + extra_methods:
            if method not in report["methods"]:
                continue
            item = report["methods"][method]
            print(
                f"{method:<40s} "
                f"{item['recall@k']:>8.4f} "
                f"{100 * item['graph_invocation_rate']:>7.1f}% "
                f"{item['avg_graph_slots']:>8.3f} "
                f"{item['action_distribution']}"
            )


if __name__ == "__main__":
    main()
