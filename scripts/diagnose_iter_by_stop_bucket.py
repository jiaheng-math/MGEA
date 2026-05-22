"""Crossed breakdown of iter_refine vs graph baseline by (stop_reason x bucket).

Separates the 141 RRF-modified queries (max_iter / jaccard_stall) from the 144
short-circuit queries (sufficient at R1 -> graph top-5 fallback, should Δ=0)
to see which stop_reason contributes the A/C regressions.

Usage:
  python3 scripts/diagnose_iter_by_stop_bucket.py \\
      --per-sample results/.../qa_per_sample_graph_anchored_B1.jsonl \\
      --iter       results/.../iterative_refine.rrf.jsonl \\
      --retrieval  results/.../retrieval_results.jsonl \\
      --routing    results/.../routing_rows.jsonl \\
      --baseline   graph \\
      --compare    iter_refine_T3_top7
"""
from __future__ import annotations

import argparse
import json
import re
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


def infer_stop_reason(row: dict) -> str:
    """Derive effective stop_reason from iter row for RRF gating purposes.

    Our rerank script routes by (label, any_retrieval_round) so the effective
    categories are:
      - 'label0'           : label=0, forced to graph top-5
      - 'zero_retrieval'   : label=1 but no iter_t round retrieved -> graph top-5
      - 'rrf_modified'     : label=1 with iter retrieval -> RRF reranked
    """
    if row.get("label") == 0:
        return "label0"
    iters = row.get("iterations") or []
    any_ret = any((it.get("retrieved_ids") or []) for it in iters)
    if not any_ret:
        return "zero_retrieval"
    return "rrf_modified"


def raw_stop_reason(row: dict) -> str:
    return row.get("stop_reason") or "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", required=True, type=Path)
    ap.add_argument("--iter",       required=True, type=Path)
    ap.add_argument("--retrieval",  required=True, type=Path)
    ap.add_argument("--routing",    required=True, type=Path)
    ap.add_argument("--baseline",   required=True)
    ap.add_argument("--compare",    required=True)
    args = ap.parse_args()

    per  = {r["id"]: r for r in load(args.per_sample)}
    itr  = {r["id"]: r for r in load(args.iter)}
    retr = {r["id"]: r for r in load(args.retrieval)}
    rout = {r["id"]: r for r in load(args.routing)}

    # Bucket assignment (same logic as bucket_qa_delta.py)
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

    def metric(qid: str, m: str) -> tuple[float, float] | None:
        rec = (per.get(qid, {}).get("methods") or {}).get(m)
        if not isinstance(rec, dict): return None
        return float(rec.get("exact_match", 0)), float(rec.get("f1", 0))

    # Group by (effective_stop, raw_stop, bucket)
    rows: dict[tuple[str, str, str], list[tuple[tuple[float,float], tuple[float,float]]]] = defaultdict(list)
    for qid in per:
        it = itr.get(qid)
        if it is None: continue
        eff  = infer_stop_reason(it)
        raw  = raw_stop_reason(it) if it.get("label") == 1 else "—"
        bkt  = bucket_of.get(qid, "unknown")
        b = metric(qid, args.baseline); c = metric(qid, args.compare)
        if b is None or c is None: continue
        rows[(eff, raw, bkt)].append((b, c))

    def fmt(items):
        n = len(items)
        if n == 0: return None
        eb = sum(p[0][0] for p in items)/n
        ec = sum(p[1][0] for p in items)/n
        fb = sum(p[0][1] for p in items)/n
        fc = sum(p[1][1] for p in items)/n
        win  = sum(1 for p in items if p[1][0] > p[0][0])
        lose = sum(1 for p in items if p[1][0] < p[0][0])
        return n, eb, ec, ec-eb, fb, fc, fc-fb, win, lose

    # ---- Summary 1: by effective gate (label0 / zero_retrieval / rrf_modified) ----
    print(f"\n=== {args.compare} vs {args.baseline}  — by EFFECTIVE GATE ===")
    print(f"{'gate':<18} {'n':>4}  {'EM_b':>6} {'EM_c':>6} {'ΔEM':>7}  {'F1_b':>6} {'F1_c':>6} {'ΔF1':>7}  win lose")
    by_gate: dict[str, list] = defaultdict(list)
    for (eff, _, _), items in rows.items():
        by_gate[eff].extend(items)
    for gate in ["label0", "zero_retrieval", "rrf_modified"]:
        r = fmt(by_gate.get(gate, []))
        if r is None: continue
        n, eb, ec, de, fb, fc, df, w, l = r
        print(f"{gate:<18} {n:>4}  {eb:>6.3f} {ec:>6.3f} {de:>+7.3f}  {fb:>6.3f} {fc:>6.3f} {df:>+7.3f}  {w:>3} {l:>4}")

    # ---- Summary 2: rrf_modified breakdown by raw stop_reason x bucket ----
    print(f"\n=== rrf_modified only — by stop_reason × bucket ===")
    print(f"{'stop':<14} {'bucket':<16} {'n':>4}  {'EM_b':>6} {'EM_c':>6} {'ΔEM':>7}  {'F1_b':>6} {'F1_c':>6} {'ΔF1':>7}  win lose")
    # collect
    by_stop_bucket: dict[tuple[str,str], list] = defaultdict(list)
    by_stop: dict[str, list] = defaultdict(list)
    by_bucket: dict[str, list] = defaultdict(list)
    for (eff, raw, bkt), items in rows.items():
        if eff != "rrf_modified": continue
        by_stop_bucket[(raw, bkt)].extend(items)
        by_stop[raw].extend(items)
        by_bucket[bkt].extend(items)
    stop_order = ["max_iter", "jaccard_stall", "sufficient", "none", "—"]
    bucket_order = ["B", "C", "A", "mixed", "label1_no_miss", "label0"]
    for stop in stop_order:
        for bkt in bucket_order:
            r = fmt(by_stop_bucket.get((stop, bkt), []))
            if r is None: continue
            n, eb, ec, de, fb, fc, df, w, l = r
            print(f"{stop:<14} {bkt:<16} {n:>4}  {eb:>6.3f} {ec:>6.3f} {de:>+7.3f}  {fb:>6.3f} {fc:>6.3f} {df:>+7.3f}  {w:>3} {l:>4}")

    print(f"\n--- rrf_modified rollup by stop_reason ---")
    for stop in stop_order:
        r = fmt(by_stop.get(stop, []))
        if r is None: continue
        n, eb, ec, de, fb, fc, df, w, l = r
        print(f"{stop:<14} {'(all)':<16} {n:>4}  {eb:>6.3f} {ec:>6.3f} {de:>+7.3f}  {fb:>6.3f} {fc:>6.3f} {df:>+7.3f}  {w:>3} {l:>4}")

    print(f"\n--- rrf_modified rollup by bucket ---")
    for bkt in bucket_order:
        r = fmt(by_bucket.get(bkt, []))
        if r is None: continue
        n, eb, ec, de, fb, fc, df, w, l = r
        print(f"{'(all)':<14} {bkt:<16} {n:>4}  {eb:>6.3f} {ec:>6.3f} {de:>+7.3f}  {fb:>6.3f} {fc:>6.3f} {df:>+7.3f}  {w:>3} {l:>4}")


if __name__ == "__main__":
    main()
