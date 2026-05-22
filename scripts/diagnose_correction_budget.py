"""Framing-1 feasibility: at what budget B does graph recover the missed gold?

For each label=1 query and each missed gold (gold not in dense top-5), find the
rank of that gold in the FULL graph_ids list (top-K, K=len(graph_ids)). Then
report: at budget B in {1,2,3,4,5}, what fraction of missed golds is recovered?

Inputs: routing_rows.jsonl (already has graph_ids in rank order) and
retrieval_results.jsonl (for full graph ranking beyond top-5, if stored).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(result_dir: Path, dense_budget: int = 5) -> dict:
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval = {r["id"]: r for r in load_jsonl(result_dir / "retrieval_results.jsonl")}

    ranks_of_missed_gold = []
    unreachable = 0
    per_query_min_budget = []

    for row in routing:
        if row.get("label") != 1:
            continue
        gold = set(row.get("gold_passage_ids") or [])
        dense_top = set((row.get("dense_ids") or [])[:dense_budget])
        missed = gold - dense_top
        if not missed:
            continue

        # Full graph ranking from retrieval_results (may be longer than top-5)
        retr = retrieval.get(row["id"], {}).get("retrieval", {}).get("graph", [])
        graph_ranking = [p["id"] for p in retr]

        ranks_here = []
        for m in missed:
            if m in graph_ranking:
                r = graph_ranking.index(m) + 1  # 1-indexed
                ranks_of_missed_gold.append(r)
                ranks_here.append(r)
            else:
                unreachable += 1
        if ranks_here:
            per_query_min_budget.append(max(ranks_here))  # cover all missed golds

    def cumulative(ranks, budgets):
        total_missed = len(ranks) + unreachable
        out = {}
        for b in budgets:
            hit = sum(1 for r in ranks if r <= b)
            out[f"B={b}"] = f"{hit}/{total_missed} ({100*hit/total_missed:.1f}%)"
        out[f"B=max({max(ranks) if ranks else 0})"] = f"{len(ranks)}/{total_missed} ({100*len(ranks)/total_missed:.1f}%)"
        out["unreachable"] = f"{unreachable}/{total_missed} ({100*unreachable/total_missed:.1f}%)"
        return out

    # Also: per-query "how large a budget do we need to cover ALL missed golds for that query"
    per_query_budget_distribution = Counter()
    for b in per_query_min_budget:
        bucket = b if b <= 5 else ("6-10" if b <= 10 else ">10")
        per_query_budget_distribution[bucket] += 1

    return {
        "dir": str(result_dir),
        "n_missed_gold_total": len(ranks_of_missed_gold) + unreachable,
        "missed_gold_rank_in_graph": {
            "recovered_sample_size": len(ranks_of_missed_gold),
            "rank_stats": {
                "mean": round(statistics.mean(ranks_of_missed_gold), 2) if ranks_of_missed_gold else None,
                "median": statistics.median(ranks_of_missed_gold) if ranks_of_missed_gold else None,
                "p75": statistics.quantiles(ranks_of_missed_gold, n=4)[2] if len(ranks_of_missed_gold) >= 4 else None,
                "p90": statistics.quantiles(ranks_of_missed_gold, n=10)[8] if len(ranks_of_missed_gold) >= 10 else None,
                "max": max(ranks_of_missed_gold) if ranks_of_missed_gold else None,
            },
            "recovery_by_budget": cumulative(ranks_of_missed_gold, [1, 2, 3, 4, 5]),
        },
        "per_query_min_budget_to_cover_all_missed_golds": {
            "n_queries": len(per_query_min_budget),
            "distribution": dict(per_query_budget_distribution),
        },
    }


if __name__ == "__main__":
    dirs = sys.argv[1:] or [
        "results/study_hotpot_hipporag_colbert_500",
        "results/study_2wiki_hipporag_colbert_500",
    ]
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"skip (missing): {d}")
            continue
        print(json.dumps(analyze(p), indent=2, ensure_ascii=False))
        print("=" * 80)
