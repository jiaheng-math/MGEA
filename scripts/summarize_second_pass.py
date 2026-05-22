"""Summarize second-pass dense results against original dense + gold.

For each dataset dir, compute:
  - gap_type distribution (from rewritten_queries.jsonl)
  - parse_fail count
  - missed-gold recall for label=1 queries at correction budget B in {1, 2, 3, 5}:
      * baseline_B: does dense_v1[5:5+B] cover any missed gold?
                    (i.e. what dense alone recovers if you just take more chunks)
      * rewrite_B:  does dense_v2[:B] cover any missed gold? (excluding dense_v1 top-5)
      * oracle_B:   upper bound if any chunk in (dense_v2 top-20) covers missed gold
  - bucket breakdown (A/B/C/mixed) using string-match gap typing from diagnose_gap_types.py

Run:
  python3 scripts/summarize_second_pass.py \\
      results/study_hotpot_hipporag_colbert_500 \\
      results/study_2wiki_hipporag_colbert_500 \\
      results/study_nq_hipporag_colbert_500
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def classify_missed(question: str, dense_items: list[dict], missed_title: str) -> str:
    q_n = norm(question)
    dense_blob = norm(" ".join(
        (p.get("title") or "") + " " + (p.get("text") or "") for p in dense_items
    ))
    t_n = norm(missed_title)
    if not t_n:
        return "unknown"
    if t_n in q_n:
        return "C"
    if t_n in dense_blob:
        return "A"
    return "B"


def analyze(result_dir: Path) -> dict:
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval = {r["id"]: r for r in load_jsonl(result_dir / "retrieval_results.jsonl")}
    rewritten = {r["id"]: r for r in load_jsonl(result_dir / "rewritten_queries.jsonl")}
    second = {r["id"]: r for r in load_jsonl(result_dir / "second_pass_retrieval.jsonl")}

    n_label1 = 0
    n_label1_missed = 0
    gap_counts = Counter()
    parse_fail = 0

    # per-bucket counters for missed-gold recall
    # bucket ∈ {A, B, C, mixed, overall}, B ∈ {1, 2, 3, 5}, method ∈ {baseline, rewrite, oracle}
    Bs = [1, 2, 3, 5]
    recall_hits: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bucket_totals: dict = defaultdict(int)

    for row in routing:
        qid = row["id"]
        if row.get("label") != 1:
            continue
        n_label1 += 1

        # gap distribution from LLM (includes unclear / skip_label0)
        rw = rewritten.get(qid, {})
        gt = rw.get("gap_type", "missing")
        gap_counts[gt] += 1
        if rw.get("rationale") == "parse_failed":
            parse_fail += 1

        gold_ids = set(row.get("gold_passage_ids") or [])
        dense_items = (retrieval.get(qid, {}).get("retrieval", {}).get("dense") or [])
        dense_ids = [p.get("id") or p.get("source_doc_id") for p in dense_items]
        dense_top5 = set(dense_ids[:5])
        missed = gold_ids - dense_top5
        if not missed:
            continue
        n_label1_missed += 1

        # classify bucket (A/B/C/mixed) based on all missed golds for this query
        kinds = set()
        for g in missed:
            kinds.add(classify_missed(row["question"], dense_items[:5], g))
        if len(kinds) == 1:
            bucket = next(iter(kinds))
        else:
            bucket = "mixed"

        for b in [bucket, "overall"]:
            bucket_totals[b] += 1

        # graph correction baseline: HippoRAG graph top-5 minus dense top-5 (= "new" chunks
        # that graph offered as correction). This is the REAL baseline the rewriting must beat
        # — it's what your mask experiments tried to improve and couldn't.
        graph_items = (retrieval.get(qid, {}).get("retrieval", {}).get("graph") or [])
        graph_ids = [p.get("id") or p.get("source_doc_id") for p in graph_items]
        graph_excl = [x for x in graph_ids if x not in dense_top5]

        # dense_v2 candidates (already in top-20 from second-pass), exclude dense_top5
        sp = second.get(qid, {})
        v2_ids = [h.get("id") for h in (sp.get("dense_v2") or [])]
        v2_excl = [x for x in v2_ids if x not in dense_top5]

        for B in Bs:
            graph_B = set(graph_excl[:B])
            rew_B = set(v2_excl[:B])
            union_B = graph_B | rew_B  # optional: does combining help?
            oracle_v2 = set(v2_excl[:20])  # rewrite upper bound

            def hit(pool):
                return int(bool(missed & pool))

            for b in [bucket, "overall"]:
                recall_hits[b][B]["graph"] += hit(graph_B)
                recall_hits[b][B]["rewrite"] += hit(rew_B)
                recall_hits[b][B]["graph+rewrite"] += hit(union_B)
                recall_hits[b][B]["rewrite_oracle20"] += hit(oracle_v2)

    # format output
    out = {
        "n_label1": n_label1,
        "n_label1_with_missed_gold": n_label1_missed,
        "gap_dist": dict(gap_counts),
        "parse_fail": parse_fail,
        "recall_at_B": {},
    }
    for bucket in ["A", "B", "C", "mixed", "overall"]:
        tot = bucket_totals.get(bucket, 0)
        out["recall_at_B"][bucket] = {"n": tot}
        if tot == 0:
            continue
        for B in Bs:
            h = recall_hits[bucket][B]
            out["recall_at_B"][bucket][f"B={B}"] = {
                "graph":            round(h["graph"]            / tot, 3),
                "rewrite":          round(h["rewrite"]          / tot, 3),
                "graph+rewrite":    round(h["graph+rewrite"]    / tot, 3),
                "rewrite_oracle20": round(h["rewrite_oracle20"] / tot, 3),
            }
    return out


def main() -> None:
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        sys.exit("Usage: summarize_second_pass.py <result_dir> [<result_dir> ...]")
    for d in dirs:
        print(f"=== {d.name} ===")
        print(json.dumps(analyze(d), indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
