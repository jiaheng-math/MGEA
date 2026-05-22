"""Rerank iterative_refine.final_ranking using reciprocal rank fusion across sources.

Motivation: the original iterative_refine.py merged per-source items with
max_score. But ColBERT scores (~10-25) and HippoRAG PPR scores (~1e-3)
live on different scales, so graph seed items get sunk to the bottom of the
merged ranking even though graph is our strongest missed-gold recoverer
(recall 0.933 in our Hotpot split). RRF fixes this by treating each source as
a rank list and fusing on rank alone:

    score_rrf(doc) = sum over sources where doc appears of  1 / (k + rank_in_source)

Sources per query:
  - S_dense : dense top-5 from retrieval_results.jsonl
  - S_graph : graph top-5 from retrieval_results.jsonl
  - S_iter_t (for each iteration t) : the retrieved_ids list in iterations[t]
    (already sorted by ColBERT score within the round)

Inputs:
  - results/.../iterative_refine.jsonl        (contains iterations[] + final_ranking with
                                                full passage dicts; we re-rank, don't re-retrieve)
  - results/.../retrieval_results.jsonl        (for seed dense + graph rank order)

Output:
  - results/.../iterative_refine.rrf.jsonl     (same schema, final_ranking re-sorted by RRF;
                                                adds "rrf_score" and "rrf_sources" per passage)

Usage:
  python3 scripts/rerank_iterative_rrf.py \\
      --iterative      results/study_hotpot_hipporag_colbert_500/iterative_refine.jsonl \\
      --retrieval      results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \\
      --out            results/study_hotpot_hipporag_colbert_500/iterative_refine.rrf.jsonl \\
      --seed-k 5 --rrf-k 60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def chunk_id(p: dict) -> str:
    return p.get("id") or p.get("source_doc_id") or ""


def rrf_rerank(passages: list[dict], source_rankings: dict[str, list[str]],
               rrf_k: int) -> list[dict]:
    """passages: full dicts; source_rankings: {source_name: [doc_id_in_rank_order]}."""
    scores: dict[str, float] = defaultdict(float)
    sources_for: dict[str, list[str]] = defaultdict(list)
    for src, ids in source_rankings.items():
        for rank, doc_id in enumerate(ids):
            if not doc_id:
                continue
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)  # +1 so rank starts at 1
            sources_for[doc_id].append(f"{src}@{rank+1}")

    # Attach scores; passages not in any source get 0 and sort to the end.
    out = []
    for p in passages:
        cid = chunk_id(p)
        pc = dict(p)
        pc["rrf_score"] = scores.get(cid, 0.0)
        pc["rrf_sources"] = sources_for.get(cid, [])
        out.append(pc)
    out.sort(key=lambda x: -x["rrf_score"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterative", required=True, type=Path)
    ap.add_argument("--retrieval", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed-k", type=int, default=5,
                    help="Top-k of dense/graph seed to include in RRF (matches iterative_refine seed).")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant k (standard default 60).")
    ap.add_argument("--anchor", choices=["none", "graph", "dense"], default="none",
                    help="If set, prepend that seed top-K as fixed prefix, dedup, then RRF the rest. "
                         "Matches the 'graph_plus_union_B{B}' pattern from build_union_retrieval.py.")
    args = ap.parse_args()

    retr = {r["id"]: r for r in load_jsonl(args.retrieval)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_reranked = n_label0_to_graph = n_zero_iter_to_graph = 0
    with args.iterative.open() as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            qid = row["id"]
            orig = retr.get(qid, {}).get("retrieval", {})
            seed_dense_items = (orig.get("dense") or [])[: args.seed_k]
            seed_graph_items = (orig.get("graph") or [])[: args.seed_k]

            # ---- Patch 3: label=0 -> mirror the graph baseline (graph top-5) ----
            # Rationale: in the evaluation, the `graph` baseline uses graph top-5 on
            # BOTH label=0 and label=1. If iter falls back to dense top-5 on label=0,
            # we're benchmarking fallback-policy differences rather than iter quality.
            # Align iter's label=0 with the baseline by using graph top-5 too.
            if row.get("label") == 0:
                row["final_ranking"] = seed_graph_items
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_label0_to_graph += 1
                continue

            iterations = row.get("iterations") or []
            any_retrieval_round = any((it.get("retrieved_ids") or []) for it in iterations)

            # ---- Patch 2: zero-retrieval label=1 -> graph top-5 only ----
            # If no AQR round actually retrieved anything (e.g., round-1 sufficient=True,
            # or aqr_empty/jaccard_stall before any retrieve), there is no NEW evidence
            # to add. In graph-anchored mode the "rest" would otherwise be dense seed,
            # which is pure ColBERT-keyword noise for queries graph already answered.
            # Drop the dense seed in that case: final = graph top-5 only.
            if args.anchor == "graph" and not any_retrieval_round:
                row["final_ranking"] = seed_graph_items
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_zero_iter_to_graph += 1
                continue

            passages = row.get("final_ranking") or []

            # Reconstruct per-source rank lists
            seed_dense_ids = [chunk_id(p) for p in seed_dense_items]
            seed_graph_ids = [chunk_id(p) for p in seed_graph_items]
            source_rankings: dict[str, list[str]] = {
                "seed_dense": seed_dense_ids,
                "seed_graph": seed_graph_ids,
            }
            for it in iterations:
                rids = it.get("retrieved_ids") or []
                if rids:
                    source_rankings[f"iter_{it['round']}"] = rids

            reranked = rrf_rerank(passages, source_rankings, args.rrf_k)

            if args.anchor == "graph":
                anchor_items = seed_graph_items
            elif args.anchor == "dense":
                anchor_items = seed_dense_items
            else:
                anchor_items = []

            if anchor_items:
                anchor_ids = {chunk_id(p) for p in anchor_items if chunk_id(p)}
                by_id = {chunk_id(p): p for p in passages}
                for p in anchor_items:
                    by_id.setdefault(chunk_id(p), p)
                prefix = [by_id[chunk_id(p)] for p in anchor_items if chunk_id(p)]
                # Patch 2b: in graph-anchored mode, restrict "rest" to chunks that
                # came from an actual iter_t retrieval round. Seed_dense is excluded
                # because it's been shown to be noisy on simple A/C-bucket queries.
                if args.anchor == "graph":
                    iter_ids: set[str] = set()
                    for it in iterations:
                        iter_ids.update(it.get("retrieved_ids") or [])
                    rest = [p for p in reranked
                            if chunk_id(p) not in anchor_ids and chunk_id(p) in iter_ids]
                else:
                    rest = [p for p in reranked if chunk_id(p) not in anchor_ids]
                final = prefix + rest
            else:
                final = reranked

            row["final_ranking"] = final
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_reranked += 1

    print(f"wrote {args.out}  reranked={n_reranked}  "
          f"label0->graph={n_label0_to_graph}  zero_iter->graph={n_zero_iter_to_graph}")


if __name__ == "__main__":
    main()
