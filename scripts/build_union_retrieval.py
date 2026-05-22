"""Build a retrieval_results.jsonl with union/rewrite/graph hybrid methods for QA.

For each query we produce (in retrieval[...]) several named methods at matched
total-budget = 5 + B, so downstream QA (batch_generate_from_retrieval.py --top-k 5+B)
compares correction strategies apples-to-apples.

Methods emitted per row:
  dense                     : original dense top-5 (pass-through)
  graph                     : original graph top-5 (pass-through)
  fusion                    : original fusion (if present, pass-through)
  hybrid_graph_B{B}         : dense top-5 + graph_excl top-B
  hybrid_rewrite_B{B}       : dense top-5 + v2_excl top-B
  hybrid_union_B{B}         : dense top-5 + dedup(graph_excl[:B], v2_excl[:B])   <-- the star

Applied only to label=1 rows. label=0 rows keep hybrid_* = dense top-5 (matches
probe-aware router behavior: no correction requested).

Dedup key for union = chunk id. "graph_excl" / "v2_excl" = already filtered to
exclude dense top-5 in second-pass-retrieval output; we re-apply the filter here
defensively in case a future caller passes raw graph/dense_v2.

Usage:
  python3 scripts/build_union_retrieval.py \\
      --orig-retrieval   results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \\
      --routing          results/study_hotpot_hipporag_colbert_500/routing_rows.jsonl \\
      --second-pass      results/study_hotpot_hipporag_colbert_500/second_pass_retrieval.jsonl \\
      --out              results/study_hotpot_hipporag_colbert_500/retrieval_results.union.jsonl \\
      --budgets 1,2,3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def chunk_id(p: dict) -> str:
    return p.get("id") or p.get("source_doc_id") or ""


def filter_exclude(items: list[dict], exclude_ids: set[str]) -> list[dict]:
    return [p for p in items if chunk_id(p) and chunk_id(p) not in exclude_ids]


def dedup_by_id(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in items:
        cid = chunk_id(p)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-retrieval", required=True, type=Path)
    ap.add_argument("--routing", required=True, type=Path)
    ap.add_argument("--second-pass", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budgets", default="1,2,3",
                    help="Comma-separated correction budgets B (extra chunks on top of dense top-5).")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    orig = {r["id"]: r for r in load_jsonl(args.orig_retrieval)}
    routing = {r["id"]: r for r in load_jsonl(args.routing)}
    second = {r["id"]: r for r in load_jsonl(args.second_pass)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_label1_applied = 0

    with args.out.open("w", encoding="utf-8") as out_f:
        for qid, row in orig.items():
            retr = dict(row.get("retrieval") or {})
            dense_items = list(retr.get("dense") or [])
            graph_items = list(retr.get("graph") or [])
            dense_top5_ids = {chunk_id(p) for p in dense_items[:5]}

            is_label1 = routing.get(qid, {}).get("label") == 1
            v2_items = (second.get(qid, {}) or {}).get("dense_v2") or []

            graph_excl = filter_exclude(graph_items, dense_top5_ids)
            v2_excl = filter_exclude(v2_items, dense_top5_ids)

            # --- graph-anchored variants: graph top-5 + rewrite/dense correction ---
            # graph_top5 is the strong prefix (best pure method in QA). We test whether
            # rewrite chunks that graph didn't already surface can push QA further.
            graph_top5 = graph_items[:5]
            graph_top5_ids = {chunk_id(p) for p in graph_top5}
            v2_new_vs_graph = filter_exclude(v2_items, graph_top5_ids)
            dense_new_vs_graph = filter_exclude(dense_items, graph_top5_ids)

            for B in budgets:
                if is_label1:
                    # (Legacy) dense-anchored variants — kept for apples-to-apples with the
                    # previous recall table. Expect these to underperform graph-anchored.
                    g_slice = graph_excl[:B]
                    r_slice = v2_excl[:B]
                    retr[f"dense_plus_graph_B{B}"] = dedup_by_id(dense_items[:5] + g_slice)
                    retr[f"dense_plus_rewrite_B{B}"] = dedup_by_id(dense_items[:5] + r_slice)
                    retr[f"dense_plus_union_B{B}"] = dedup_by_id(
                        dense_items[:5] + g_slice + r_slice
                    )

                    # Graph-anchored variants — the real comparison for QA.
                    rn = v2_new_vs_graph[:B]
                    dn = dense_new_vs_graph[:B]
                    retr[f"graph_plus_rewrite_B{B}"] = dedup_by_id(graph_top5 + rn)
                    retr[f"graph_plus_dense_B{B}"] = dedup_by_id(graph_top5 + dn)
                    retr[f"graph_plus_union_B{B}"] = dedup_by_id(graph_top5 + rn + dn)
                else:
                    # label=0: no correction; keep dense top-5 (router behavior)
                    retr[f"dense_plus_graph_B{B}"] = dense_items[:5]
                    retr[f"dense_plus_rewrite_B{B}"] = dense_items[:5]
                    retr[f"dense_plus_union_B{B}"] = dense_items[:5]
                    retr[f"graph_plus_rewrite_B{B}"] = dense_items[:5]
                    retr[f"graph_plus_dense_B{B}"] = dense_items[:5]
                    retr[f"graph_plus_union_B{B}"] = dense_items[:5]

            if is_label1:
                n_label1_applied += 1

            new_row = dict(row)
            new_row["retrieval"] = retr
            out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    print(f"wrote {args.out} — label1_applied={n_label1_applied} budgets={budgets}")


if __name__ == "__main__":
    main()
