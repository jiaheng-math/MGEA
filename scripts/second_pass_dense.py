"""Second-pass dense retrieval using rewritten queries (from gap_conditioned_rewrite.py).

Pipeline position:
  [1st-pass colbert + HippoRAG]      -> retrieval_results.jsonl
  [gap_conditioned_rewrite.py]        -> rewritten_queries.jsonl
  [THIS SCRIPT]                       -> second_pass_retrieval.jsonl
  [eval script, next step]            -> merged ranking + missed-gold recall per bucket

What it does:
  - Reuses the existing colbert index (rebuild_index=False). No re-embedding of corpus.
  - For each rewritten row with gap_type in {B, C}, runs colbert on the rewritten_query.
  - For rows with gap_type in {skip_label0, unclear}, writes empty dense_v2 (these rows
    will fall back to the original dense ranking in the eval/merge step).

Config is reused verbatim from the study YAML; we only need the colbert paths and
the shared_corpus_path. No other side effects.

Usage:
  python3 scripts/second_pass_dense.py \\
      --config           configs/study_hotpot_hipporag_colbert_500.yaml \\
      --rewritten-file   results/study_hotpot_hipporag_colbert_500/rewritten_queries.jsonl \\
      --out              results/study_hotpot_hipporag_colbert_500/second_pass_retrieval.jsonl \\
      --top-k 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from tqdm import tqdm

# Running from the repository root (the entry point for all `python -m src.*` calls)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_shared_corpus
from src.retrieval_dense_colbert import ColBERTRetriever


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_retriever_from_config(config: dict, project_root: Path) -> ColBERTRetriever:
    shared_corpus_path = config.get("shared_corpus_path")
    if not shared_corpus_path:
        raise ValueError("config.shared_corpus_path is required")
    corpus = load_shared_corpus(str(resolve_path(project_root, str(shared_corpus_path))))

    colbert_root = str(resolve_path(project_root, config["colbert_root"]))
    return ColBERTRetriever(
        corpus=corpus,
        root=colbert_root,
        experiment_name=config["colbert_experiment_name"],
        index_name=config["colbert_index_name"],
        checkpoint=config.get("colbert_checkpoint", "colbert-ir/colbertv2.0"),
        nbits=config.get("colbert_nbits", 2),
        partitions=config.get("colbert_partitions"),
        doc_maxlen=config.get("colbert_doc_maxlen", 220),
        query_maxlen=config.get("colbert_query_maxlen", 64),
        kmeans_niters=config.get("colbert_kmeans_niters", 4),
        rebuild_index=False,   # critical: reuse existing index, never rebuild here
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--rewritten-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=20,
                   help="How many second-pass dense candidates to keep per query.")
    p.add_argument("--only-gap-types", default="B,C",
                   help="Comma-separated gap_type values to run second-pass on.")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    with args.config.open() as f:
        config = yaml.safe_load(f)

    only_types = {t.strip() for t in args.only_gap_types.split(",") if t.strip()}

    rewritten = load_jsonl(args.rewritten_file)
    runnable = [r for r in rewritten if r.get("gap_type") in only_types]
    print(f"Loaded {len(rewritten)} rewritten rows; will run second-pass on "
          f"{len(runnable)} (gap_type in {sorted(only_types)})", file=sys.stderr)

    retriever = build_retriever_from_config(config, project_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_hit_empty = 0
    with args.out.open("w", encoding="utf-8") as out_f:
        # Pass-through rows that won't be retrieved (keep id coverage for downstream join)
        pass_through = [r for r in rewritten if r.get("gap_type") not in only_types]
        for r in pass_through:
            out_f.write(json.dumps({
                "id": r["id"],
                "question": r["question"],
                "rewritten_query": r.get("rewritten_query", r["question"]),
                "gap_type": r.get("gap_type"),
                "dense_v2": [],
            }, ensure_ascii=False) + "\n")

        for r in tqdm(runnable, desc="Second-pass colbert"):
            rq = r["rewritten_query"]
            try:
                hits = retriever.retrieve(rq, args.top_k)
            except Exception as e:
                print(f"[warn] retrieve failed for {r['id']}: {e}", file=sys.stderr)
                hits = []
            if not hits:
                n_hit_empty += 1

            dense_v2 = []
            for rank, h in enumerate(hits):
                # RetrievedPassage is a dataclass; asdict keeps field names stable
                try:
                    d = asdict(h)
                except TypeError:
                    d = {"id": h.id, "title": getattr(h, "title", None),
                         "text": getattr(h, "text", None),
                         "score": float(getattr(h, "score", 0.0)),
                         "source_doc_id": getattr(h, "source_doc_id", None)}
                d["rank"] = rank
                dense_v2.append(d)

            out_f.write(json.dumps({
                "id": r["id"],
                "question": r["question"],
                "rewritten_query": rq,
                "gap_type": r.get("gap_type"),
                "dense_v2": dense_v2,
            }, ensure_ascii=False) + "\n")

    print(f"done. runnable={len(runnable)} empty_hits={n_hit_empty}", file=sys.stderr)


if __name__ == "__main__":
    main()
