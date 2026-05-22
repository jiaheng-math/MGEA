"""Evaluate targeted correction vs hard-switch routing at equal budget K=5.

For each query, let the router decide (probability >= 0.5 ⇒ route to graph).
When routed, produce the final top-5 as:
    dense[:5-B]  ⊕  (graph_ids minus those)[:B]
Where B in {0,1,2,3,4,5}:
  - B=0 is pure dense (no correction)
  - B=5 is hard-switch (current main-line result)
  - B in {1,2,3,4} is targeted correction (preserve dense, add graph complement)

When NOT routed, output is dense[:5] regardless of B (router skips graph).

Metric: gold passage recall@5 averaged over all queries in the dataset.
Also reports breakdown by ground-truth label (1 vs 0) to see where gains come from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki":  "results/study_2wiki_hipporag_colbert_500",
    "nq":     "results/study_nq_hipporag_colbert_500",
}


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def recall_at_k(pred_ids: list[str], gold: set[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    picked = set(pred_ids[:k])
    return len(picked & gold) / len(gold)


def evaluate_dataset(result_dir: Path, budgets: list[int], threshold: float, prob_key: str) -> dict:
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    oof_list = load_jsonl(result_dir / "pareto_strict_cv" / "oof_predictions.jsonl")
    oof = {r["id"]: r for r in oof_list}

    n_total = len(routing)
    # Build top-5 under each budget and count recall
    # Aggregate: B -> (hits, total) over: ALL, label=1, label=0, routed, not_routed
    strata = {"ALL": [], "label=1": [], "label=0": [], "routed": [], "not_routed": []}
    recall_per_B: dict[int, dict[str, list[float]]] = {
        b: {k: [] for k in strata} for b in budgets
    }
    route_decisions = []  # per-query whether we route

    skipped = 0
    for row in routing:
        qid = row["id"]
        gold = set(row.get("gold_passage_ids") or [])
        if not gold:
            skipped += 1
            continue
        dense_ids = row.get("dense_ids") or []
        graph_ids = row.get("graph_ids") or []
        label = row.get("label")  # ground truth

        # Router decision from OOF prob. Tie/missing => follow config (dense by default)
        oof_row = oof.get(qid, {})
        prob = oof_row.get(prob_key)
        if prob is None:
            routed = False  # missing probability => default to dense (tie policy)
        else:
            routed = prob >= threshold
        route_decisions.append(routed)

        for B in budgets:
            if not routed or B == 0:
                picked = list(dense_ids[:5])
            else:
                keep_dense = dense_ids[: 5 - B]
                used = set(keep_dense)
                extras = [g for g in graph_ids if g not in used]
                picked = list(keep_dense) + extras[:B]
            r = recall_at_k(picked, gold, k=5)
            recall_per_B[B]["ALL"].append(r)
            if label == 1:
                recall_per_B[B]["label=1"].append(r)
            elif label == 0:
                recall_per_B[B]["label=0"].append(r)
            if routed:
                recall_per_B[B]["routed"].append(r)
            else:
                recall_per_B[B]["not_routed"].append(r)

    route_rate = sum(route_decisions) / max(1, len(route_decisions))

    out = {
        "dataset": result_dir.name,
        "n_total": n_total,
        "n_gold_present": n_total - skipped,
        "route_rate": round(route_rate, 4),
        "threshold": threshold,
        "prob_key": prob_key,
        "results": {},
    }
    for B in budgets:
        out["results"][f"B={B}"] = {
            stratum: {
                "n": len(vals),
                "recall@5": round(sum(vals) / len(vals), 4) if vals else None,
            }
            for stratum, vals in recall_per_B[B].items()
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki", "nq"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--prob-key", default="query_plus_probe_probability")
    ap.add_argument("--output", default="results/targeted_correction_eval.json")
    args = ap.parse_args()

    all_reports = []
    for ds in args.datasets:
        res_p = Path(DATASETS[ds])
        if not res_p.exists():
            print(f"skip {ds}: missing {res_p}")
            continue
        r = evaluate_dataset(res_p, args.budgets, args.threshold, args.prob_key)
        all_reports.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(all_reports, f, indent=2)

    # Console summary
    print(f"\n{'='*78}")
    print(f"Targeted correction vs hard-switch at fixed K=5")
    print(f"  B=0: dense only    B=5: hard-switch (full graph replace when routed)")
    print(f"  B in 1..4: dense[:5-B] ⊕ graph_extra[:B] when routed")
    print(f"{'='*78}")
    for rep in all_reports:
        print(f"\n{rep['dataset']}  (n={rep['n_total']}, route_rate={rep['route_rate']})")
        header = f"  {'stratum':<15s}" + "".join(f"  B={b:<6d}" for b in args.budgets)
        print(header)
        for stratum in ["ALL", "label=1", "label=0", "routed", "not_routed"]:
            line = f"  {stratum:<15s}"
            for b in args.budgets:
                v = rep["results"][f"B={b}"][stratum]
                line += f"  {v['recall@5'] if v['recall@5'] is not None else '  -  ':<6}  "
            n_line = rep["results"][f"B={args.budgets[0]}"][stratum]["n"]
            line += f"  (n={n_line})"
            print(line)
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
