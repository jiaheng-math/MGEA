"""Proxy analysis of HippoRAG graph reliability as a correction backend.

Uses only routing_rows.jsonl (no access to the raw graph). Reports:
  1. Graph score shape: top1/top2 ratio, score variance — peaked vs flat
  2. Dense ∩ Graph overlap among top-5
  3. Ceiling of graph as correction source:
     - on label=1 queries, what fraction has graph_recall@5 >= dense_recall@5
     - on label=1 queries, graph_recall@5 distribution
  4. Graph hit on the specific gold that dense missed
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(result_dir: Path, top_k: int = 5) -> dict:
    rows = load_jsonl(result_dir / "routing_rows.jsonl")

    # 1. Graph score shape (all queries)
    top1_top2_ratios = []
    score_gini_like = []  # top1 / sum(top5)
    for r in rows:
        s = r.get("graph_scores") or []
        if len(s) >= 2 and s[1] > 0:
            top1_top2_ratios.append(s[0] / s[1])
        if len(s) >= 1 and sum(s[:top_k]) > 0:
            score_gini_like.append(s[0] / sum(s[:top_k]))

    # 2. Dense ∩ Graph overlap
    overlaps = []
    for r in rows:
        d = set((r.get("dense_ids") or [])[:top_k])
        g = set((r.get("graph_ids") or [])[:top_k])
        if d and g:
            overlaps.append(len(d & g) / top_k)

    # 3. Graph as correction source on label=1
    label1 = [r for r in rows if r.get("label") == 1]
    graph_helps = sum(
        1 for r in label1
        if (r.get("graph_recall@5") or 0) >= (r.get("dense_recall@5") or 0) + 1e-9
    )
    graph_recalls = [r.get("graph_recall@5") or 0.0 for r in label1]
    graph_r5_eq_1 = sum(1 for v in graph_recalls if v >= 0.999)
    graph_r5_eq_0 = sum(1 for v in graph_recalls if v < 1e-9)

    # 4. For label=1 queries where dense missed specific gold, did graph hit it?
    missed_total = 0
    missed_covered_by_graph = 0
    for r in label1:
        gold = set(r.get("gold_passage_ids") or [])
        d = set((r.get("dense_ids") or [])[:top_k])
        g = set((r.get("graph_ids") or [])[:top_k])
        missed = gold - d
        for m in missed:
            missed_total += 1
            if m in g:
                missed_covered_by_graph += 1

    def pct(n, d):
        return f"{n}/{d} ({100*n/d:.1f}%)" if d else f"{n}/0"

    def stats(vals):
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "p25": round(statistics.quantiles(vals, n=4)[0], 3) if len(vals) >= 4 else None,
            "p75": round(statistics.quantiles(vals, n=4)[2], 3) if len(vals) >= 4 else None,
        }

    return {
        "dir": str(result_dir),
        "n_queries": len(rows),
        "n_label1": len(label1),
        "graph_score_shape": {
            "top1_top2_ratio": stats(top1_top2_ratios),
            "top1_over_sum_top5": stats(score_gini_like),
            "interpretation": "ratio>>1 → peaked PPR; near 1 → flat, graph unsure",
        },
        "dense_graph_overlap_top5": stats(overlaps),
        "graph_as_correction_backend": {
            "label1_queries": len(label1),
            "graph_beats_or_equals_dense@5": pct(graph_helps, len(label1)),
            "graph_recall@5_distribution": stats(graph_recalls),
            "graph_recall@5==1.0": pct(graph_r5_eq_1, len(label1)),
            "graph_recall@5==0.0": pct(graph_r5_eq_0, len(label1)),
        },
        "missed_gold_recovery_by_graph": {
            "total_missed_gold_in_label1": missed_total,
            "also_retrieved_by_graph_top5": pct(missed_covered_by_graph, missed_total),
        },
    }


if __name__ == "__main__":
    dirs = sys.argv[1:] or [
        "results/study_hotpot_hipporag_colbert_500",
        "results/study_2wiki_hipporag_colbert_500",
        "results/study_nq_hipporag_colbert_500",
    ]
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"skip (missing): {d}")
            continue
        print(json.dumps(analyze(p), indent=2, ensure_ascii=False))
        print("=" * 80)
