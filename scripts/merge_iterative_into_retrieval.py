"""Inject iterative_refine final_ranking as a named method into retrieval_results.jsonl.

Produces a new retrieval_results file containing every method already present PLUS:
  retrieval["iter_refine_T{T}_top{K}"]  = final_ranking[:K]  for each query.

label=0 queries get dense top-5 (matches the router's "no correction" contract, same
convention as build_union_retrieval.py).

Downstream QA (batch_generate_from_retrieval.py) then runs unchanged with --top-k K.

Usage:
  python3 scripts/merge_iterative_into_retrieval.py \\
      --retrieval-in  results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \\
      --iterative     results/study_hotpot_hipporag_colbert_500/iterative_refine.jsonl \\
      --retrieval-out results/study_hotpot_hipporag_colbert_500/retrieval_results.iter.jsonl \\
      --T 3 --top-k 5,7,10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def chunk_id(p: dict) -> str:
    return p.get("id") or p.get("source_doc_id") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-in", required=True, type=Path)
    ap.add_argument("--iterative", required=True, type=Path)
    ap.add_argument("--retrieval-out", required=True, type=Path)
    ap.add_argument("--T", type=int, default=3, help="Label for method name (iter_refine_T{T}).")
    ap.add_argument("--top-k", default="5,7,10",
                    help="Comma-separated K values to emit as separate method names.")
    args = ap.parse_args()

    Ks = [int(k) for k in args.top_k.split(",") if k.strip()]
    iterr = {r["id"]: r for r in load_jsonl(args.iterative)}

    args.retrieval_out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_with_iter = 0
    with args.retrieval_in.open() as fin, args.retrieval_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            qid = row["id"]
            retr = dict(row.get("retrieval") or {})
            dense_top5 = (retr.get("dense") or [])[:5]

            ir = iterr.get(qid)
            if ir is None:
                # no iterative entry -> fall back to dense top-5 for all K
                final = dense_top5
            else:
                final = ir.get("final_ranking") or []
                if not final:
                    final = dense_top5
                else:
                    n_with_iter += 1

            for K in Ks:
                retr[f"iter_refine_T{args.T}_top{K}"] = final[:K]

            row["retrieval"] = retr
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"wrote {args.retrieval_out}  rows={n_written}  with_iter={n_with_iter}  Ks={Ks}")


if __name__ == "__main__":
    main()
