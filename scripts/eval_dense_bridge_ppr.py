"""Evaluate Dense-Bridge PPR vs Standard PPR vs Oracle on label=1 queries.

For each label=1 query and each budget B in {1,2,3}:
  - Standard PPR: top-B chunks (excluding dense top-5), seeds = query entities only
  - Dense-Bridge PPR: top-B chunks (excluding dense top-5), seeds include bridges,
    dense chunks masked from teleport
  - Oracle: insert up to B of the actual missed gold passages

Metric: fraction of missed gold recovered by the B correction slots.
Bucketed by gap type (A_bridge_visible / B_hop1_miss / C_query_entity / mixed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).parent))
from dense_bridge_ppr import (
    GraphContext,
    extract_bridges,
    extract_query_entity_vids,
    dense_bridge_ppr,
    standard_ppr,
    dense_seeded_ppr,
    residual_score,
    rank_correction_chunks,
    norm_entity,
)


RESIDUAL_LAMBDAS = [0.5, 1.0, 2.0]


NLP = None
def _nlp():
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


def extract_query_entities(question: str) -> list[str]:
    doc = _nlp()(question)
    ents = [e.text for e in doc.ents]
    if ents:
        return ents
    # Fallback: simple capitalized spans + quoted spans
    out = re.findall(r'"([^"]+)"|\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)\b', question)
    return [a or b for a, b in out if (a or b)]


def classify_gap_types(question: str, dense_passages: list[dict], missed_gold_ids: set[str]) -> dict[str, str]:
    """Return missed_gold_id -> gap_type(A/B/C)."""
    q_norm = re.sub(r"\s+", " ", question.lower()).strip()
    dense_text = " ".join((p.get("text") or "") + " " + (p.get("title") or "") for p in dense_passages)
    dense_text_norm = re.sub(r"\s+", " ", dense_text.lower()).strip()
    out = {}
    for gid in missed_gold_ids:
        g_norm = re.sub(r"\s+", " ", gid.lower()).strip()
        if g_norm in q_norm:
            out[gid] = "C"
        elif g_norm in dense_text_norm:
            out[gid] = "A"
        else:
            out[gid] = "B"
    return out


def per_query_gap_bucket(types: dict[str, str]) -> str:
    s = set(types.values())
    if s == {"A"}: return "A"
    if s == {"B"}: return "B"
    if s == {"C"}: return "C"
    if s: return "mixed"
    return "none"


def evaluate_dataset(
    cache_dir: Path,
    result_dir: Path,
    budgets: list[int],
    damping: float,
    bridge_share: float,
    max_queries: int | None,
) -> dict:
    print(f"\n=== {result_dir.name} ===", flush=True)
    t0 = time.time()
    ctx = GraphContext(cache_dir)
    print(f"graph loaded: {ctx.n} nodes, {len(ctx.passage_to_vid)} chunks "
          f"[{time.time()-t0:.1f}s]", flush=True)

    routing = [json.loads(l) for l in open(result_dir / "routing_rows.jsonl")]
    retrieval = {r["id"]: r for r in (json.loads(l) for l in open(result_dir / "retrieval_results.jsonl"))}

    label1 = [r for r in routing if r.get("label") == 1]
    if max_queries:
        label1 = label1[:max_queries]
    print(f"label=1 queries: {len(label1)}", flush=True)

    # Aggregate: per-method, per-B, per-bucket -> [hits, total_missed]
    agg: dict[tuple[str,int,str], list[int]] = {}
    def bump(method, b, bucket, hit, total):
        key = (method, b, bucket)
        v = agg.setdefault(key, [0, 0])
        v[0] += hit; v[1] += total

    max_b = max(budgets)
    errors = 0
    for i, row in enumerate(label1):
        try:
            qid = row["id"]
            dense_ids = (row.get("dense_ids") or [])[:5]
            gold = set(row.get("gold_passage_ids") or [])
            missed = gold - set(dense_ids)
            if not missed: continue

            retr = retrieval.get(qid, {}).get("retrieval", {})
            dense_passages = retr.get("dense", [])[:5]
            gap_types = classify_gap_types(row["question"], dense_passages, missed)
            bucket = per_query_gap_bucket(gap_types)

            # Query entities via spacy
            qent_strings = extract_query_entities(row["question"])
            qvids = extract_query_entity_vids(ctx, qent_strings)

            # === Standard PPR (query entities only) ===
            pr_std = standard_ppr(ctx, qvids, damping=damping)
            std_corr = rank_correction_chunks(ctx, pr_std, set(dense_ids), max_b)
            std_ids = [p for p, _ in std_corr]

            # === Dense-Bridge PPR (three mask variants) ===
            bridges = extract_bridges(ctx, dense_ids, qvids)
            db_ids_by_mode: dict[str, list[str]] = {}
            for mode in ("none", "teleport", "graph"):
                pr_db = dense_bridge_ppr(
                    ctx, qvids, bridges, dense_ids,
                    damping=damping, bridge_share=bridge_share,
                    mask_mode=mode,
                )
                db_corr = rank_correction_chunks(ctx, pr_db, set(dense_ids), max_b)
                db_ids_by_mode[mode] = [p for p, _ in db_corr]

            # === Residual PPR: r_Q - lam * r_D ===
            # r_D = PPR seeded on dense chunks themselves (what dense already covers)
            pr_dense = dense_seeded_ppr(ctx, dense_ids, damping=damping)
            resid_ids_by_lam: dict[float, list[str]] = {}
            resid_bridge_ids_by_lam: dict[float, list[str]] = {}
            # Bridge-seeded (mask=none) variant used as the "Q" side of a combined residual
            pr_bridge_none = dense_bridge_ppr(
                ctx, qvids, bridges, dense_ids,
                damping=damping, bridge_share=bridge_share,
                mask_mode="none",
            )
            for lam in RESIDUAL_LAMBDAS:
                resid_ids_by_lam[lam] = [
                    p for p, _ in rank_correction_chunks(
                        ctx, residual_score(pr_std, pr_dense, lam),
                        set(dense_ids), max_b,
                    )
                ]
                resid_bridge_ids_by_lam[lam] = [
                    p for p, _ in rank_correction_chunks(
                        ctx, residual_score(pr_bridge_none, pr_dense, lam),
                        set(dense_ids), max_b,
                    )
                ]

            # === HippoRAG cached (just use graph_ids excluding dense) ===
            hippo_full = (row.get("graph_ids") or [])
            hippo_ids = [p for p in hippo_full if p not in dense_ids][:max_b]

            # === Oracle (best case: directly insert missed gold) ===
            oracle_ids = list(missed)[:max_b]

            method_ids: list[tuple[str, list[str]]] = [
                ("standard_ppr", std_ids),
                ("dense_bridge_ppr_none", db_ids_by_mode["none"]),
                ("dense_bridge_ppr_teleport", db_ids_by_mode["teleport"]),
                ("dense_bridge_ppr_graph", db_ids_by_mode["graph"]),
                ("hipporag_cached", hippo_ids),
                ("oracle", oracle_ids),
            ]
            for lam in RESIDUAL_LAMBDAS:
                method_ids.append((f"residual_ppr_lam{lam}", resid_ids_by_lam[lam]))
                method_ids.append((f"residual_bridge_ppr_lam{lam}", resid_bridge_ids_by_lam[lam]))

            for b in budgets:
                # Per bucket + total, count missed-gold recovery under budget b
                for method, ids in method_ids:
                    picked = set(ids[:b])
                    hit = len(picked & missed)
                    bump(method, b, bucket, hit, len(missed))
                    bump(method, b, "ALL", hit, len(missed))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [warn] query {row.get('id')} failed: {e}", flush=True)

        if (i+1) % 50 == 0:
            print(f"  progress {i+1}/{len(label1)} [{time.time()-t0:.1f}s]", flush=True)

    # Format
    report: dict = {"dataset": result_dir.name, "n_label1": len(label1), "errors": errors, "budgets": budgets, "results": {}}
    buckets = ["ALL", "A", "B", "C", "mixed"]
    methods = [
        "standard_ppr",
        "dense_bridge_ppr_none",
        "dense_bridge_ppr_teleport",
        "dense_bridge_ppr_graph",
        *[f"residual_ppr_lam{lam}" for lam in RESIDUAL_LAMBDAS],
        *[f"residual_bridge_ppr_lam{lam}" for lam in RESIDUAL_LAMBDAS],
        "hipporag_cached",
        "oracle",
    ]
    for bucket in buckets:
        report["results"][bucket] = {}
        for method in methods:
            row_out = {}
            for b in budgets:
                k = (method, b, bucket)
                if k not in agg: continue
                hit, tot = agg[k]
                row_out[f"B={b}"] = {"recall": round(hit/tot, 4) if tot else None, "hit": hit, "total": tot}
            report["results"][bucket][method] = row_out
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki", "nq"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--bridge-share", type=float, default=0.5)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--output", default="results/dense_bridge_ppr_eval.json")
    args = ap.parse_args()

    DATASET_MAP = {
        "hotpot": ("hipporag_cache/hotpot_shared_500", "results/study_hotpot_hipporag_colbert_500"),
        "2wiki":  ("hipporag_cache/2wiki_shared_500", "results/study_2wiki_hipporag_colbert_500"),
        "nq":     ("hipporag_cache/nq_shared_500", "results/study_nq_hipporag_colbert_500"),
    }

    all_reports = []
    for ds in args.datasets:
        cache, res = DATASET_MAP[ds]
        cache_p, res_p = Path(cache), Path(res)
        if not cache_p.exists() or not res_p.exists():
            print(f"skip {ds}: missing {cache_p if not cache_p.exists() else res_p}")
            continue
        r = evaluate_dataset(cache_p, res_p, args.budgets, args.damping, args.bridge_share, args.max_queries)
        all_reports.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_reports, f, indent=2)

    # Print compact summary
    print("\n\n================== SUMMARY ==================")
    for r in all_reports:
        print(f"\n{r['dataset']} (label=1 n={r['n_label1']})")
        for bucket in ["ALL", "A", "B", "C", "mixed"]:
            if bucket not in r["results"]: continue
            print(f"  [{bucket}]")
            for method in r["results"][bucket]:
                res = r["results"][bucket].get(method, {})
                line = f"    {method:28s}"
                for b in args.budgets:
                    v = res.get(f"B={b}", {})
                    r_ = v.get("recall")
                    n_ = v.get("total", 0)
                    line += f"  B={b}: {r_}" + (f" (n={n_})" if b == args.budgets[0] else "")
                print(line)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
