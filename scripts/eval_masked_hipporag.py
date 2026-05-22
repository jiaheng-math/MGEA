"""Evaluate chunk-masked HippoRAG vs vanilla HippoRAG on label=1 queries.

Compares, on the correction subset (label=1 queries with missed gold):
  - hipporag_baseline: cached HippoRAG graph_ids (from routing_rows.jsonl),
    with dense_top_5 excluded from the ranking
  - hipporag_masked: live HippoRAG with dense top-5 chunk nodes zeroed out of
    the PPR seed's passage_weights (entity seeds untouched), with dense_top_5
    also excluded from the output
  - oracle: directly insert up to B missed-gold passages

Metric: missed-gold recall under correction budget B in {1,2,3,5}, bucketed
by gap type (A = bridge visible in dense text, B = hop-1 miss, C = query
entity).

Usage (cloud):
  python3 scripts/eval_masked_hipporag.py \
    --datasets hotpot 2wiki nq \
    --budgets 1 2 3 5 \
    --llm-model gpt-4.1 \
    --embedding-model text-embedding-3-small \
    --llm-base-url https://<your-openai-compatible-endpoint>/v1 \
    --embedding-base-url https://<your-openai-compatible-endpoint>/v1 \
    --output results/masked_hipporag_eval_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure local hipporag source is importable if running from repo layout
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hipporag_chunk_mask import HippoRAGMasked  # noqa: E402


DATASET_MAP = {
    "hotpot": {
        "cache": "hipporag_cache/hotpot_shared_500",
        "result": "results/study_hotpot_hipporag_colbert_500",
        "corpus": "data/hotpot_dev_distractor_500_seed42_corpus.json",
    },
    "2wiki": {
        "cache": "hipporag_cache/2wiki_shared_500",
        "result": "results/study_2wiki_hipporag_colbert_500",
        "corpus": "data/2wikimultihopqa_dev_500_seed42_corpus.json",
    },
    "nq": {
        "cache": "hipporag_cache/nq_shared_500",
        "result": "results/study_nq_hipporag_colbert_500",
        "corpus": "data/nq_dev_500_seed42_corpus.json",
    },
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def classify_gap_types(
    question: str, dense_passages: list[dict], missed_gold_ids: set[str]
) -> dict[str, str]:
    q_norm = norm(question)
    dense_text_norm = norm(
        " ".join((p.get("text") or "") + " " + (p.get("title") or "") for p in dense_passages)
    )
    out = {}
    for gid in missed_gold_ids:
        g_norm = norm(gid)
        if g_norm in q_norm:
            out[gid] = "C"
        elif g_norm in dense_text_norm:
            out[gid] = "A"
        else:
            out[gid] = "B"
    return out


def per_query_bucket(types: dict[str, str]) -> str:
    s = set(types.values())
    if s == {"A"}: return "A"
    if s == {"B"}: return "B"
    if s == {"C"}: return "C"
    return "mixed" if s else "none"


def build_corpus_docs(corpus_path: Path) -> list[tuple[str, str]]:
    """Return list of (passage_id, serialized_doc) matching the pilot serializer.

    Must match src/retrieval_graph_hipporag.py::_serialize_passage exactly so
    HippoRAG's cached chunk hash_ids line up with the existing index.
    """
    with corpus_path.open() as f:
        raw = json.load(f)
    passages = raw if isinstance(raw, list) else raw.get("passages") or raw.get("corpus") or []
    docs: list[tuple[str, str]] = []
    for p in passages:
        pid = p.get("id") or p.get("passage_id") or p.get("doc_id")
        title = p.get("title") or ""
        text = p.get("text") or p.get("passage") or ""
        serialized = "\n".join([f"PASSAGE_ID::{pid}", f"TITLE::{title}", "TEXT::", text])
        docs.append((pid, serialized))
    return docs


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_dataset(
    dataset_name: str,
    cache_dir: Path,
    result_dir: Path,
    corpus_path: Path,
    llm_model: str,
    embedding_model: str,
    llm_base_url: str | None,
    embedding_base_url: str | None,
    budgets: list[int],
    max_queries: int | None,
) -> dict:
    print(f"\n=== {dataset_name}: {result_dir.name} ===", flush=True)
    t0 = time.time()

    # --- load retrieval rows ---
    routing = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_idx = {r["id"]: r for r in load_jsonl(result_dir / "retrieval_results.jsonl")}
    label1 = [r for r in routing if r.get("label") == 1]
    if max_queries:
        label1 = label1[:max_queries]
    print(f"label=1 queries: {len(label1)}", flush=True)

    # --- build HippoRAGMasked pointing at cached index ---
    docs = build_corpus_docs(corpus_path)
    doc_texts = [d[1] for d in docs]

    init_kwargs = {
        "save_dir": str(cache_dir),
        "llm_model_name": llm_model,
        "embedding_model_name": embedding_model,
    }
    if llm_base_url:
        init_kwargs["llm_base_url"] = llm_base_url
    if embedding_base_url:
        init_kwargs["embedding_base_url"] = embedding_base_url

    hippo = HippoRAGMasked(**init_kwargs)
    # ensure index is up-to-date (no-op if already indexed w/ same corpus)
    hippo.index(docs=doc_texts)
    print(f"index ready [{time.time()-t0:.1f}s]", flush=True)

    # --- run multiple mask variants for all label=1 queries ---
    queries = [r["question"] for r in label1]
    dense_ids_list = [(r.get("dense_ids") or [])[:5] for r in label1]

    max_b = max(budgets)
    retrieve_n = max(max_b * 4, 20)

    variants: list[tuple[str, dict]] = [
        ("chunk_mask",         {"chunk_mask": True,  "entity_mask_mode": "none"}),
        ("entity_sat_1.0",     {"chunk_mask": False, "entity_mask_mode": "saturation", "entity_mask_threshold": 1.0}),
        ("entity_sat_0.75",    {"chunk_mask": False, "entity_mask_mode": "saturation", "entity_mask_threshold": 0.75}),
        ("entity_sat_0.5",     {"chunk_mask": False, "entity_mask_mode": "saturation", "entity_mask_threshold": 0.5}),
        ("chunk+entity_1.0",   {"chunk_mask": True,  "entity_mask_mode": "saturation", "entity_mask_threshold": 1.0}),
    ]

    variant_rankings: dict[str, list[list[tuple[str, float]]]] = {}
    for name, kwargs in variants:
        print(f"  variant: {name}  {kwargs}", flush=True)
        variant_rankings[name] = hippo.retrieve_with_mask(
            queries=queries,
            dense_ids_per_query=dense_ids_list,
            num_to_retrieve=retrieve_n,
            exclude_dense_from_output=True,
            **kwargs,
        )
        print(f"    done [{time.time()-t0:.1f}s]", flush=True)

    # --- aggregate missed-gold recall, bucketed ---
    # agg[(method, B, bucket)] = [hit, total]
    agg: dict[tuple[str, int, str], list[int]] = {}
    def bump(method: str, b: int, bucket: str, hit: int, total: int) -> None:
        k = (method, b, bucket)
        v = agg.setdefault(k, [0, 0])
        v[0] += hit; v[1] += total

    for i, row in enumerate(label1):
        qid = row["id"]
        dense_ids = (row.get("dense_ids") or [])[:5]
        gold = set(row.get("gold_passage_ids") or [])
        missed = gold - set(dense_ids)
        if not missed:
            continue

        dense_passages = retrieval_idx.get(qid, {}).get("retrieval", {}).get("dense", [])[:5]
        gap_types = classify_gap_types(row["question"], dense_passages, missed)
        bucket = per_query_bucket(gap_types)

        # baseline: cached graph_ids minus dense
        hippo_full = row.get("graph_ids") or []
        baseline_ids = [p for p in hippo_full if p not in dense_ids][:max_b]

        # oracle
        oracle_ids = list(missed)[:max_b]

        methods: list[tuple[str, list[str]]] = [
            ("hipporag_baseline", baseline_ids),
        ]
        for name in variant_rankings:
            ids = [pid for pid, _ in variant_rankings[name][i]][:max_b]
            methods.append((f"masked_{name}", ids))
        methods.append(("oracle", oracle_ids))
        for b in budgets:
            for method, ids in methods:
                picked = set(ids[:b])
                hit = len(picked & missed)
                bump(method, b, bucket, hit, len(missed))
                bump(method, b, "ALL", hit, len(missed))

    # --- format ---
    report = {
        "dataset": result_dir.name,
        "n_label1": len(label1),
        "budgets": budgets,
        "results": {},
    }
    buckets = ["ALL", "A", "B", "C", "mixed"]
    methods = (
        ["hipporag_baseline"]
        + [f"masked_{name}" for name, _ in variants]
        + ["oracle"]
    )
    for bucket in buckets:
        report["results"][bucket] = {}
        for method in methods:
            row_out = {}
            for b in budgets:
                k = (method, b, bucket)
                if k not in agg:
                    continue
                hit, tot = agg[k]
                row_out[f"B={b}"] = {
                    "recall": round(hit / tot, 4) if tot else None,
                    "hit": hit,
                    "total": tot,
                }
            report["results"][bucket][method] = row_out
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki", "nq"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--llm-model", default="gpt-4.1")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    ap.add_argument("--embedding-base-url", default=os.getenv("HIPPORAG_EMBED_BASE_URL"))
    ap.add_argument("--output", default="results/masked_hipporag_eval.json")
    args = ap.parse_args()

    all_reports = []
    for ds in args.datasets:
        if ds not in DATASET_MAP:
            print(f"skip unknown dataset: {ds}")
            continue
        meta = DATASET_MAP[ds]
        cache_p = Path(meta["cache"])
        res_p = Path(meta["result"])
        corpus_p = Path(meta["corpus"])
        if not cache_p.exists() or not res_p.exists() or not corpus_p.exists():
            miss = [p for p in (cache_p, res_p, corpus_p) if not p.exists()]
            print(f"skip {ds}: missing {miss}")
            continue
        r = evaluate_dataset(
            dataset_name=ds,
            cache_dir=cache_p,
            result_dir=res_p,
            corpus_path=corpus_p,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            llm_base_url=args.llm_base_url,
            embedding_base_url=args.embedding_base_url,
            budgets=args.budgets,
            max_queries=args.max_queries,
        )
        all_reports.append(r)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_reports, f, indent=2)

    # compact summary
    print("\n\n================== SUMMARY ==================")
    for r in all_reports:
        print(f"\n{r['dataset']} (label=1 n={r['n_label1']})")
        for bucket in ["ALL", "A", "B", "C", "mixed"]:
            if bucket not in r["results"]:
                continue
            print(f"  [{bucket}]")
            for method, res in r["results"][bucket].items():
                line = f"    {method:22s}"
                for b in args.budgets:
                    v = res.get(f"B={b}", {})
                    rec = v.get("recall")
                    n_ = v.get("total", 0)
                    line += f"  B={b}: {rec}" + (f" (n={n_})" if b == args.budgets[0] else "")
                print(line)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
