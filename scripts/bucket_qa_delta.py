"""Per-gap-type bucket breakdown of QA delta (method vs baseline).

For each label=1 query, classify gap type (A/B/C/mixed) using string match against
missed-gold titles (same as diagnose_gap_types.py). Label=0 queries go to bucket
'label0'. Then report EM/F1 delta per bucket.

Usage:
  python3 scripts/bucket_qa_delta.py \\
      --per-sample results/.../qa_per_sample_graph_anchored_B1.jsonl \\
      --retrieval  results/.../retrieval_results.jsonl \\
      --routing    results/.../routing_rows.jsonl \\
      --baseline graph \\
      --compare  graph_plus_rewrite_B1,graph_plus_union_B1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def load(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def classify(question: str, dense_items: list[dict], missed_title: str) -> str:
    q_n = norm(question)
    blob = norm(" ".join((p.get("title") or "") + " " + (p.get("text") or "") for p in dense_items))
    t_n = norm(missed_title)
    if not t_n: return "unknown"
    if t_n in q_n: return "C"
    if t_n in blob: return "A"
    return "B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", required=True, type=Path)
    ap.add_argument("--retrieval", required=True, type=Path)
    ap.add_argument("--routing",  required=True, type=Path)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--compare",  required=True)
    args = ap.parse_args()

    per = {r["id"]: r for r in load(args.per_sample)}
    retr = {r["id"]: r for r in load(args.retrieval)}
    rout = {r["id"]: r for r in load(args.routing)}

    # bucket each id
    bucket_of: dict[str, str] = {}
    for qid, row in rout.items():
        if row.get("label") != 1:
            bucket_of[qid] = "label0"; continue
        gold = set(row.get("gold_passage_ids") or [])
        dense_items = (retr.get(qid, {}).get("retrieval", {}).get("dense") or [])[:5]
        dense_ids = {(p.get("id") or p.get("source_doc_id")) for p in dense_items}
        missed = gold - dense_ids
        if not missed:
            bucket_of[qid] = "label1_no_miss"; continue
        kinds = {classify(row["question"], dense_items, g) for g in missed}
        bucket_of[qid] = next(iter(kinds)) if len(kinds) == 1 else "mixed"

    compare = [m.strip() for m in args.compare.split(",") if m.strip()]

    def get_em_f1(qid: str, method: str) -> tuple[float, float] | None:
        rec = (per.get(qid, {}).get("methods") or {}).get(method)
        if not isinstance(rec, dict): return None
        return float(rec.get("exact_match", 0)), float(rec.get("f1", 0))

    buckets = ["B", "C", "A", "mixed", "label1_no_miss", "label0", "overall"]
    for m in compare:
        print(f"\n=== {m} vs {args.baseline} ===")
        print(f"{'bucket':<17} {'n':>4}  {'EM_base':>8} {'EM_cmp':>8} {'ΔEM':>8}  {'F1_base':>8} {'F1_cmp':>8} {'ΔF1':>8}  win lose")
        agg: dict[str, list] = defaultdict(list)
        for qid, bkt in bucket_of.items():
            base = get_em_f1(qid, args.baseline); cmp_ = get_em_f1(qid, m)
            if base is None or cmp_ is None: continue
            agg[bkt].append((base, cmp_))
            agg["overall"].append((base, cmp_))
        for bkt in buckets:
            items = agg.get(bkt, [])
            if not items: continue
            n = len(items)
            eb = sum(p[0][0] for p in items) / n
            ec = sum(p[1][0] for p in items) / n
            fb = sum(p[0][1] for p in items) / n
            fc = sum(p[1][1] for p in items) / n
            win = sum(1 for p in items if p[1][0] > p[0][0])
            lose = sum(1 for p in items if p[1][0] < p[0][0])
            print(f"{bkt:<17} {n:>4}  {eb:>8.3f} {ec:>8.3f} {ec-eb:>+8.3f}  {fb:>8.3f} {fc:>8.3f} {fc-fb:>+8.3f}  {win:>3} {lose:>4}")


if __name__ == "__main__":
    main()
