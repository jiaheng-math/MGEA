"""Diagnose gap types among label=1 queries.

For each missed gold passage in a label=1 query, classify:
  C (query_entity):   missed gold title appears in the question text
  A (bridge_visible): missed gold title appears in dense top-k passage text (but not in query)
  B (hop1_miss):      missed gold title appears neither in query nor in dense top-k text

Usage:
  python3 scripts/diagnose_gap_types.py <result_dir> [<result_dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(result_dir: Path, top_k: int = 5) -> dict:
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval = {r["id"]: r for r in load_jsonl(result_dir / "retrieval_results.jsonl")}

    per_query_types: list[set[str]] = []
    missed_gap_counter = Counter()
    label1 = 0
    total = len(routing)
    missed_total = 0
    examples = {"A": [], "B": [], "C": []}

    for row in routing:
        if row.get("label") != 1:
            continue
        label1 += 1
        qid = row["id"]
        question_norm = norm(row["question"])
        gold_ids = set(row.get("gold_passage_ids") or [])
        dense_ids = set((row.get("dense_ids") or [])[:top_k])
        missed = gold_ids - dense_ids
        if not missed:
            continue

        retr = retrieval.get(qid, {}).get("retrieval", {}).get("dense", [])[:top_k]
        dense_text_norm = norm(" ".join((p.get("text") or "") + " " + (p.get("title") or "") for p in retr))

        gap_types_here: set[str] = set()
        for gold_title in missed:
            missed_total += 1
            t_norm = norm(gold_title)
            if not t_norm:
                continue
            if t_norm in question_norm:
                kind = "C"
            elif t_norm in dense_text_norm:
                kind = "A"
            else:
                kind = "B"
            missed_gap_counter[kind] += 1
            gap_types_here.add(kind)
            if len(examples[kind]) < 3:
                examples[kind].append({
                    "qid": qid,
                    "question": row["question"],
                    "missed_gold": gold_title,
                    "dense_top5_titles": row["dense_ids"][:top_k],
                })
        per_query_types.append(gap_types_here)

    query_type_counter = Counter()
    for types in per_query_types:
        if types == {"A"}:
            query_type_counter["only_A"] += 1
        elif types == {"B"}:
            query_type_counter["only_B"] += 1
        elif types == {"C"}:
            query_type_counter["only_C"] += 1
        elif types:
            query_type_counter["mixed"] += 1

    def pct(n, d):
        return f"{n}/{d} ({100*n/d:.1f}%)" if d else f"{n}/0"

    return {
        "dir": str(result_dir),
        "total_queries": total,
        "label1_queries": label1,
        "label1_with_missed_gold": len(per_query_types),
        "missed_gold_total": missed_total,
        "per_missed_gold": {
            "A_bridge_visible": pct(missed_gap_counter["A"], missed_total),
            "B_hop1_miss":      pct(missed_gap_counter["B"], missed_total),
            "C_query_entity":   pct(missed_gap_counter["C"], missed_total),
        },
        "per_query": {
            "only_A": pct(query_type_counter["only_A"], len(per_query_types)),
            "only_B": pct(query_type_counter["only_B"], len(per_query_types)),
            "only_C": pct(query_type_counter["only_C"], len(per_query_types)),
            "mixed":  pct(query_type_counter["mixed"],  len(per_query_types)),
        },
        "examples": examples,
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
        report = analyze(p)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("=" * 80)
