"""Evaluate dense-anchored graph neighborhood retrieval.

This is a clean second-contribution candidate that does not depend on:

  - B-slot graph replacement
  - hand-designed gap types
  - HippoRAG query-time fact reranking
  - graph-only retrieval as a required candidate source

Instead, it treats the cached HippoRAG graph as an index structure. For each
query, dense top-k passages are graph entry points. We expand from those passage
nodes through entity nodes to nearby passage nodes, rank the neighborhood, and
compare it to full HippoRAG graph retrieval.

The first goal is diagnostic: determine whether dense-anchored graph
neighborhoods contain enough gold evidence to be a real retrieval path.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_MAP = {
    "hotpot": ("hipporag_cache/hotpot_shared_500", "results/study_hotpot_hipporag_colbert_500"),
    "2wiki": ("hipporag_cache/2wiki_shared_500", "results/study_2wiki_hipporag_colbert_500"),
    "nq": ("hipporag_cache/nq_shared_500", "results/study_nq_hipporag_colbert_500"),
}


class GraphContext:
    def __init__(self, cache_dir: Path) -> None:
        graph_path = cache_dir / "gpt-4.1_text-embedding-3-small" / "graph.pickle"
        openie_path = cache_dir / "openie_results_ner_gpt-4.1.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph pickle: {graph_path}")
        if not openie_path.exists():
            raise FileNotFoundError(f"Missing OpenIE cache: {openie_path}")

        with graph_path.open("rb") as handle:
            self.g = pickle.load(handle)
        with openie_path.open("r", encoding="utf-8") as handle:
            self.openie_docs = json.load(handle).get("docs", [])

        self.n = int(self.g.vcount())
        self.adj = self.g.get_adjlist(mode="all")
        self.passage_to_vid: dict[str, int] = {}
        self.vid_to_passage: dict[int, str] = {}
        self.node_key_to_vid: dict[str, int] = {}
        self.chunk_vid_to_triple_entity_vids: dict[int, list[int]] = {}
        self.entity_vid_to_chunk_vids: dict[int, list[int]] = defaultdict(list)

        for vertex in self.g.vs:
            key = vertex["name"] if "name" in vertex.attributes() else vertex["hash_id"]
            key = str(key)
            vid = int(vertex.index)
            self.node_key_to_vid[key] = vid
            content = str(vertex["content"]) if "content" in vertex.attributes() else ""
            if key.startswith("chunk-"):
                passage_id = extract_passage_id(content)
                if passage_id:
                    self.passage_to_vid[passage_id] = vid
                    self.vid_to_passage[vid] = passage_id

        for doc in self.openie_docs:
            chunk_key = str(doc.get("idx", ""))
            chunk_vid = self.node_key_to_vid.get(chunk_key)
            if chunk_vid is None:
                continue
            entity_vids = []
            for triple in doc.get("extracted_triples", []):
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                for surface in (triple[0], triple[2]):
                    entity_vid = self.node_key_to_vid.get(entity_node_key(surface))
                    if entity_vid is not None:
                        entity_vids.append(entity_vid)
            entity_vids = sorted(set(entity_vids))
            self.chunk_vid_to_triple_entity_vids[chunk_vid] = entity_vids
            for entity_vid in entity_vids:
                self.entity_vid_to_chunk_vids[entity_vid].append(chunk_vid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense-anchored graph neighborhood retrieval.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--cache-dirs", nargs="*", default=[])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--dense-k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hop", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", default="results/dense_anchored_graph_neighborhood_eval.json")
    parser.add_argument("--per-sample-output", default="results/dense_anchored_graph_neighborhood_per_sample.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    per_sample_all: list[dict[str, Any]] = []
    for dataset, cache_dir, result_dir in resolve_dataset_pairs(args):
        report, per_sample = evaluate_dataset(dataset, cache_dir, result_dir, args)
        reports.append(report)
        per_sample_all.extend(per_sample)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    per_sample_output = resolve_project_path(args.per_sample_output)
    per_sample_output.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_output.open("w", encoding="utf-8") as handle:
        for row in per_sample_all:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(reports)
    print(f"\nFull report: {output}")
    print(f"Per-sample report: {per_sample_output}")


def evaluate_dataset(
    dataset: str,
    cache_dir: Path,
    result_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(f"\n=== {dataset}: dense-anchored graph neighborhood ===", flush=True)
    start = time.time()
    ctx = GraphContext(cache_dir)
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    if args.max_queries:
        rows = rows[: args.max_queries]

    aggs = make_aggs()
    candidate_sizes = []
    expansion_times = []
    per_sample: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
        if not gold:
            continue
        dense_ids = list(row.get("dense_ids") or [])
        graph_ids = list(row.get("graph_ids") or [])
        dense_top = dense_ids[: args.dense_k]
        dense_scores = [float(value) for value in (row.get("dense_scores") or [])]

        t0 = time.time()
        candidates = expand_neighborhood(ctx, dense_top, args)
        expansion_times.append((time.time() - t0) * 1000.0)
        candidate_sizes.append(len(candidates))

        rankings = {
            "dense_only": dense_ids[: args.top_k],
            "full_hipporag_graph": graph_ids[: args.top_k],
            "neighborhood_distance": rank_by_distance(candidates, args)[: args.top_k],
            "neighborhood_dense_weighted": rank_by_dense_weight(candidates, dense_top, dense_scores, args)[: args.top_k],
            "neighborhood_oracle": rank_oracle(candidates, gold, args)[: args.top_k],
        }

        strata = ["ALL"]
        if gold - set(dense_ids[: args.top_k]):
            strata.append("dense_miss")
        label = row.get("label")
        strata.append("label_tie_or_invalid" if label is None else f"label_{int(label)}")

        scores = {}
        for method, ids in rankings.items():
            score = score_ranking(ids, gold, set(dense_ids[: args.top_k]))
            scores[method] = score
            for stratum in strata:
                update_agg(aggs[method][stratum], score)

        per_sample.append(
            {
                "dataset": dataset,
                "id": row.get("id"),
                "dense_miss": "dense_miss" in strata,
                "candidate_size": len(candidates),
                "expansion_ms": expansion_times[-1],
                "rankings": rankings,
                "scores": scores,
            }
        )

        if index % 50 == 0:
            print(f"  progress {index}/{len(rows)} [{time.time() - start:.1f}s]", flush=True)

    report = {
        "dataset": dataset,
        "n": len(per_sample),
        "dense_k": args.dense_k,
        "top_k": args.top_k,
        "max_hop": args.max_hop,
        "max_candidates": args.max_candidates,
        "avg_candidate_size": round(float(np.mean(candidate_sizes)), 4) if candidate_sizes else None,
        "median_candidate_size": round(float(np.median(candidate_sizes)), 4) if candidate_sizes else None,
        "avg_expansion_ms": round(float(np.mean(expansion_times)), 4) if expansion_times else None,
        "metrics": summarize_aggs(aggs),
    }
    return report, per_sample


def expand_neighborhood(ctx: GraphContext, dense_top: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    source_vids = [ctx.passage_to_vid[pid] for pid in dense_top if pid in ctx.passage_to_vid]
    candidates: dict[str, dict[str, Any]] = {}
    queue: deque[tuple[int, int, int]] = deque()
    seen: dict[int, int] = {}
    for source_rank, source_vid in enumerate(source_vids, start=1):
        queue.append((source_vid, 0, source_rank))
        seen[source_vid] = 0

    while queue:
        vid, dist, source_rank = queue.popleft()
        passage_id = ctx.vid_to_passage.get(vid)
        if passage_id is not None:
            record = candidates.setdefault(
                passage_id,
                {
                    "min_dist": dist,
                    "source_rank": source_rank,
                    "path_count": 0,
                },
            )
            record["min_dist"] = min(record["min_dist"], dist)
            record["source_rank"] = min(record["source_rank"], source_rank)
            record["path_count"] += 1
            if len(candidates) >= args.max_candidates:
                continue

        if dist >= args.max_hop:
            continue
        for nxt in ctx.adj[vid]:
            old = seen.get(nxt)
            if old is not None and old <= dist + 1:
                continue
            seen[nxt] = dist + 1
            queue.append((nxt, dist + 1, source_rank))
    return candidates


def rank_by_distance(candidates: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[str]:
    return sorted(
        candidates,
        key=lambda pid: (
            candidates[pid]["min_dist"],
            candidates[pid]["source_rank"],
            -candidates[pid]["path_count"],
            pid,
        ),
    )


def rank_by_dense_weight(
    candidates: dict[str, dict[str, Any]],
    dense_top: list[str],
    dense_scores: list[float],
    args: argparse.Namespace,
) -> list[str]:
    dense_score_map = {pid: float(dense_scores[idx]) for idx, pid in enumerate(dense_top) if idx < len(dense_scores)}
    max_score = max(dense_score_map.values()) if dense_score_map else 1.0
    return sorted(
        candidates,
        key=lambda pid: (
            candidates[pid]["min_dist"],
            -dense_score_map.get(pid, 0.0) / max(1e-12, max_score),
            candidates[pid]["source_rank"],
            -candidates[pid]["path_count"],
            pid,
        ),
    )


def rank_oracle(candidates: dict[str, dict[str, Any]], gold: set[str], args: argparse.Namespace) -> list[str]:
    base = rank_by_distance(candidates, args)
    return [pid for pid in base if pid in gold] + [pid for pid in base if pid not in gold]


def score_ranking(ids: list[str], gold: set[str], dense_top: set[str]) -> dict[str, Any]:
    selected = set(ids)
    missed = gold - dense_top
    return {
        "recall": len(selected & gold) / max(1, len(gold)),
        "missed_recovery": len(selected & missed) / len(missed) if missed else None,
        "hit": 1.0 if missed and selected & missed else 0.0 if missed else None,
    }


def make_aggs() -> dict[str, dict[str, dict[str, Any]]]:
    return defaultdict(lambda: defaultdict(lambda: {"n": 0, "missed_n": 0, "recall_sum": 0.0, "missed_sum": 0.0, "hit_sum": 0.0}))


def update_agg(agg: dict[str, Any], score: dict[str, Any]) -> None:
    agg["n"] += 1
    agg["recall_sum"] += score["recall"]
    if score["missed_recovery"] is not None:
        agg["missed_n"] += 1
        agg["missed_sum"] += score["missed_recovery"]
        agg["hit_sum"] += score["hit"]


def summarize_aggs(aggs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for method, by_stratum in aggs.items():
        out[method] = {}
        for stratum, agg in by_stratum.items():
            n = agg["n"]
            missed_n = agg["missed_n"]
            out[method][stratum] = {
                "n": int(n),
                "missed_n": int(missed_n),
                "recall@5": round(agg["recall_sum"] / n, 4) if n else None,
                "missed_recovery": round(agg["missed_sum"] / missed_n, 4) if missed_n else None,
                "hit_rate": round(agg["hit_sum"] / missed_n, 4) if missed_n else None,
            }
    return out


def print_summary(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        print("\n" + "=" * 88)
        print(
            f"DATASET {report['dataset']} n={report['n']} hop={report['max_hop']} "
            f"avgCand={report['avg_candidate_size']} avgMs={report['avg_expansion_ms']}"
        )
        for stratum in ("ALL", "dense_miss"):
            print(f"\n{stratum}")
            rows = []
            for method, by_stratum in report["metrics"].items():
                row = by_stratum.get(stratum)
                if row:
                    rows.append((method, row))
            rows.sort(key=lambda item: item[1]["recall@5"] or 0.0, reverse=True)
            for method, row in rows:
                print(
                    f"{method}: R={row['recall@5']} missedRec={row['missed_recovery']} "
                    f"hit={row['hit_rate']}"
                )


def resolve_dataset_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.result_dirs or args.cache_dirs:
        if len(args.result_dirs) != len(args.cache_dirs):
            raise ValueError("--result-dirs and --cache-dirs must have the same length.")
        return [
            (Path(result_dir).name, resolve_project_path(cache_dir), resolve_project_path(result_dir))
            for cache_dir, result_dir in zip(args.cache_dirs, args.result_dirs)
        ]
    return [
        (dataset, resolve_project_path(DATASET_MAP[dataset][0]), resolve_project_path(DATASET_MAP[dataset][1]))
        for dataset in args.datasets
    ]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_passage_id(content: str) -> str | None:
    match = re.search(r"PASSAGE_ID::([^\n]+)", content)
    return match.group(1).strip() if match else None


def text_processing(text: Any) -> str:
    if isinstance(text, list):
        return " ".join(text_processing(item) for item in text)
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    import hashlib

    return prefix + hashlib.md5(content.encode()).hexdigest()


def entity_node_key(surface: Any) -> str:
    return compute_mdhash_id(text_processing(surface), prefix="entity-")


if __name__ == "__main__":
    main()
