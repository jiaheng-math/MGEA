"""Residual-PPR rerank of HippoRAG candidates, spliced into dense.

Core idea: HippoRAG's PPR is seeded on query entities only; its top-ranked
chunks often overlap with dense top-5 (redundant). We rerank HippoRAG's
top-N candidates by subtracting a "dense-neighborhood" score r_D(chunk),
computed as a PPR seeded on dense top-5 chunks themselves. Chunks highly
scored by r_D are ones dense already implicitly covers, so we penalize
them when choosing correction slots.

Pipeline per query:
  1. H_top  = graph_ids[:topN]         (from re-run study with top_k_values [3,5,30])
  2. r_D    = personalized_pagerank(seed = dense_ids[:5])
  3. For lambda in LAMBDAS:
        s_H(c) = 1 / (k0 + rank_H(c))           # HippoRAG's own rank, RRF-style
        s_D(c) = 1 / (k0 + rank_D(c))           # rank by r_D among H_top candidates
        s_final(c) = s_H(c) - lambda * s_D(c)
     rerank H_top by s_final descending; take top-B that are NOT in dense[:5]
  4. Final top-5 = dense[:5-B]  +  rerank_top_B
  5. Recall@5 vs gold_passage_ids, bucketed by gap type (A/B/C) and label.

Baselines:
  - hipporag_splice_naive:  just take H_top excluding dense, no rerank (= B=1..5 of current targeted_correction_eval)
  - oracle: insert up to B actual missed gold passages

Output: results/residual_rerank_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dense_bridge_ppr import GraphContext, dense_seeded_ppr


DATASET_MAP = {
    "hotpot": ("hipporag_cache/hotpot_shared_500", "results/study_hotpot_hipporag_colbert_500"),
    "2wiki":  ("hipporag_cache/2wiki_shared_500",  "results/study_2wiki_hipporag_colbert_500"),
    "nq":     ("hipporag_cache/nq_shared_500",     "results/study_nq_hipporag_colbert_500"),
}

LAMBDAS = [0.0, 0.5, 1.0, 2.0]      # 0.0 is the no-rerank (pure HippoRAG) baseline
BUDGETS = [0, 1, 2, 3, 4, 5]
TOP_N = 30                          # how many HippoRAG candidates to consider for rerank
DENSE_K = 5                         # dense slots
RRF_K0 = 60                         # standard RRF constant


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_gap_type(question: str, dense_passages_text: str, gold_title: str) -> str:
    q_n, d_n, g_n = norm(question), norm(dense_passages_text), norm(gold_title)
    if g_n and g_n in q_n:
        return "C"
    if g_n and g_n in d_n:
        return "A"
    return "B"


def per_query_gap_bucket(types: set[str]) -> str:
    if types == {"A"}: return "A"
    if types == {"B"}: return "B"
    if types == {"C"}: return "C"
    if types: return "mixed"
    return "none"


def rerank_top_b(
    ctx: GraphContext,
    dense_ids: list[str],
    graph_ids: list[str],
    lam: float,
    budget: int,
) -> list[str]:
    """Return top-B graph_ids excluding dense, reranked by RRF residual."""
    if budget == 0 or not graph_ids:
        return []
    candidates = graph_ids[:TOP_N]
    dense_set = set(dense_ids[:DENSE_K])

    if lam == 0.0:
        # No rerank: just filter dense out of HippoRAG order
        return [p for p in candidates if p not in dense_set][:budget]

    # r_D over full graph
    r_D = dense_seeded_ppr(ctx, dense_ids[:DENSE_K], damping=0.5)
    # Rank candidates by r_D (descending)
    cand_scores_D = []
    for p in candidates:
        cvid = ctx.passage_to_vid.get(p)
        s = float(r_D[cvid]) if cvid is not None else 0.0
        cand_scores_D.append(s)
    order_D = sorted(range(len(candidates)), key=lambda i: -cand_scores_D[i])
    rank_D = {i: r for r, i in enumerate(order_D)}  # candidate_index -> rank in r_D

    # s_final = RRF(rank_H) - lambda * RRF(rank_D)
    scored = []
    for i, p in enumerate(candidates):
        s_H = 1.0 / (RRF_K0 + i)
        s_D = 1.0 / (RRF_K0 + rank_D[i])
        s_final = s_H - lam * s_D
        scored.append((s_final, i, p))
    scored.sort(key=lambda x: -x[0])
    out: list[str] = []
    for _, _, p in scored:
        if p in dense_set:
            continue
        out.append(p)
        if len(out) >= budget:
            break
    return out


def evaluate_one_dataset(cache_dir: Path, result_dir: Path) -> dict:
    print(f"\n=== {result_dir.name} ===", flush=True)
    t0 = time.time()
    ctx = GraphContext(cache_dir)
    print(f"graph: {ctx.n} nodes, {len(ctx.passage_to_vid)} chunks [{time.time()-t0:.1f}s]", flush=True)

    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval = {r["id"]: r for r in load_jsonl(result_dir / "retrieval_results.jsonl")}

    # Sanity: ensure expanded top-N exists
    max_gl = max(len(r.get("graph_ids") or []) for r in routing)
    print(f"max graph_ids length across queries: {max_gl} (need >= {TOP_N})", flush=True)
    if max_gl < TOP_N:
        print(f"[ERROR] routing_rows.jsonl has fewer than {TOP_N} graph candidates per query.")
        print(f"        Did you rerun study_main with top_k_values: [3, 5, {TOP_N}]?")
        return {"dataset": result_dir.name, "error": "insufficient_graph_depth", "max_graph_ids": max_gl}

    # Buckets: lam -> B -> stratum -> list[recall@5]
    strata_names = ["ALL", "label=0", "label=1",
                    "gap_A", "gap_B", "gap_C", "gap_mixed",
                    "routed_yes_label1", "dense_sufficient_label0"]
    rec: dict = {lam: {b: {s: [] for s in strata_names} for b in BUDGETS} for lam in LAMBDAS}
    # Baseline: oracle
    oracle_rec: dict = {b: {s: [] for s in strata_names} for b in BUDGETS}

    errors = 0
    for i, row in enumerate(routing):
        try:
            qid = row["id"]
            gold = set(row.get("gold_passage_ids") or [])
            if not gold:
                continue
            dense_ids = row.get("dense_ids") or []
            graph_ids = row.get("graph_ids") or []
            label = row.get("label")

            dense_top = dense_ids[:DENSE_K]
            # Gap type bucket (based on dense miss)
            missed = gold - set(dense_top)
            bucket = "none"
            if missed:
                retr = retrieval.get(qid, {}).get("retrieval", {}).get("dense", [])[:DENSE_K]
                dense_text = " ".join((p.get("text") or "") + " " + (p.get("title") or "") for p in retr)
                types = {classify_gap_type(row["question"], dense_text, g) for g in missed}
                bucket = per_query_gap_bucket(types)

            # Precompute rerank list for each lambda (compute r_D once per query via lam>0 path)
            for lam in LAMBDAS:
                for B in BUDGETS:
                    extras = rerank_top_b(ctx, dense_ids, graph_ids, lam, B)
                    picked = list(dense_top[:DENSE_K - B]) + extras
                    # Dedup preserving order, truncate 5
                    seen = set(); final = []
                    for p in picked:
                        if p in seen: continue
                        seen.add(p); final.append(p)
                        if len(final) >= DENSE_K: break
                    r = len(set(final) & gold) / max(1, len(gold))

                    rec[lam][B]["ALL"].append(r)
                    if label == 0: rec[lam][B]["label=0"].append(r); rec[lam][B]["dense_sufficient_label0"].append(r)
                    elif label == 1: rec[lam][B]["label=1"].append(r); rec[lam][B]["routed_yes_label1"].append(r)
                    if bucket == "A": rec[lam][B]["gap_A"].append(r)
                    elif bucket == "B": rec[lam][B]["gap_B"].append(r)
                    elif bucket == "C": rec[lam][B]["gap_C"].append(r)
                    elif bucket == "mixed": rec[lam][B]["gap_mixed"].append(r)

            # Oracle (insert up to B actual missed gold)
            missed_list = list(gold - set(dense_top))
            for B in BUDGETS:
                extras = missed_list[:B]
                picked = list(dense_top[:DENSE_K - B]) + extras
                seen = set(); final = []
                for p in picked:
                    if p in seen: continue
                    seen.add(p); final.append(p)
                    if len(final) >= DENSE_K: break
                r = len(set(final) & gold) / max(1, len(gold))
                oracle_rec[B]["ALL"].append(r)
                if label == 0: oracle_rec[B]["label=0"].append(r); oracle_rec[B]["dense_sufficient_label0"].append(r)
                elif label == 1: oracle_rec[B]["label=1"].append(r); oracle_rec[B]["routed_yes_label1"].append(r)
                if bucket == "A": oracle_rec[B]["gap_A"].append(r)
                elif bucket == "B": oracle_rec[B]["gap_B"].append(r)
                elif bucket == "C": oracle_rec[B]["gap_C"].append(r)
                elif bucket == "mixed": oracle_rec[B]["gap_mixed"].append(r)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [warn] {row.get('id')}: {e}", flush=True)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(routing)} [{time.time()-t0:.1f}s]", flush=True)

    def agg(vals_map):
        out = {}
        for s, vs in vals_map.items():
            out[s] = {"n": len(vs), "recall@5": round(sum(vs)/len(vs), 4) if vs else None}
        return out

    report = {
        "dataset": result_dir.name,
        "n_total": len(routing),
        "errors": errors,
        "lambdas": LAMBDAS,
        "budgets": BUDGETS,
        "top_n": TOP_N,
        "rrf_k0": RRF_K0,
        "by_lambda": {
            str(lam): {f"B={B}": agg(rec[lam][B]) for B in BUDGETS}
            for lam in LAMBDAS
        },
        "oracle": {f"B={B}": agg(oracle_rec[B]) for B in BUDGETS},
    }
    print(f"done {result_dir.name} [{time.time()-t0:.1f}s]", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki", "nq"])
    ap.add_argument("--output", default="results/residual_rerank_eval.json")
    args = ap.parse_args()

    reports = []
    for ds in args.datasets:
        cache, res = DATASET_MAP[ds]
        cache_p, res_p = Path(cache), Path(res)
        if not cache_p.exists() or not res_p.exists():
            print(f"skip {ds}: missing {cache_p if not cache_p.exists() else res_p}")
            continue
        reports.append(evaluate_one_dataset(cache_p, res_p))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reports, f, indent=2)

    # Compact summary: ALL stratum, B=3 (the budget where naive splice starts winning)
    print("\n================== SUMMARY (ALL, B=3) ==================")
    print(f"{'dataset':<32s} " + "  ".join(f"lam={lam}" for lam in LAMBDAS) + "   oracle")
    for r in reports:
        if "error" in r: continue
        line = f"{r['dataset']:<32s} "
        for lam in LAMBDAS:
            v = r["by_lambda"][str(lam)]["B=3"]["ALL"]["recall@5"]
            line += f"  {v}"
        oracle_b3 = r["oracle"]["B=3"]["ALL"]["recall@5"]
        line += f"   {oracle_b3}"
        print(line)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
