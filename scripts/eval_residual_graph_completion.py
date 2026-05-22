"""Evaluate residual dense-conditioned graph completion.

This is a diagnostic script, not a paper-ready pipeline.  It tests whether
dense-first retrieval can guide graph retrieval toward evidence that dense did
not already cover.

For each query:
  1. Take ColBERT dense top-k from saved routing rows.
  2. Build a query-side graph seed distribution.
     - hipporag_internal: reuse HippoRAG's fact reranker and phrase seeds.
     - cached_graph: seed cached HippoRAG top passages as a no-LLM diagnostic.
  3. Add bridge entity seeds extracted from triples in dense top-k passages.
  4. Compute dense-neighborhood PPR seeded on dense top-k passage nodes.
  5. Rank graph complements by:

         score(p) = P_query_bridge(p) - lambda * P_dense(p)

     excluding dense top-k passages.

The main comparison is missed-gold recovery in the correction slots:
  residual_graph_completion vs cached HippoRAG complement.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for candidate in (PROJECT_ROOT, REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hipporag.utils.misc_utils import compute_mdhash_id, text_processing


DATASET_MAP = {
    "hotpot": ("hipporag_cache/hotpot_shared_500", "results/study_hotpot_hipporag_colbert_500"),
    "2wiki": ("hipporag_cache/2wiki_shared_500", "results/study_2wiki_hipporag_colbert_500"),
    "nq": ("hipporag_cache/nq_shared_500", "results/study_nq_hipporag_colbert_500"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual graph completion from saved runs.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--cache-dirs", nargs="*", default=[])
    parser.add_argument("--dense-k", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--bridge-share", type=float, default=0.35)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--seed-source",
        choices=["hipporag_internal", "cached_graph"],
        default="hipporag_internal",
        help=(
            "hipporag_internal reuses true HippoRAG fact seeds. cached_graph is a no-LLM "
            "sanity check that seeds cached graph top passages."
        ),
    )
    parser.add_argument(
        "--include-hippo-passage-weights",
        action="store_true",
        help="When using hipporag_internal, include HippoRAG's internal dense passage weights in the query seed.",
    )
    parser.add_argument("--hipporag-llm-model", default="gpt-4.1")
    parser.add_argument("--hipporag-embedding-model", default="text-embedding-3-small")
    parser.add_argument("--hipporag-llm-base-url", default=os.getenv("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument(
        "--hipporag-embedding-base-url",
        default=os.getenv("HIPPORAG_EMBEDDING_BASE_URL"),
    )
    parser.add_argument("--output", default="results/residual_graph_completion_eval.json")
    return parser.parse_args()


class GraphContext:
    def __init__(self, cache_dir: Path) -> None:
        graph_path = cache_dir / "gpt-4.1_text-embedding-3-small" / "graph.pickle"
        openie_path = cache_dir / "openie_results_ner_gpt-4.1.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing HippoRAG graph pickle: {graph_path}")
        if not openie_path.exists():
            raise FileNotFoundError(f"Missing OpenIE cache: {openie_path}")

        with graph_path.open("rb") as handle:
            self.g = pickle.load(handle)
        with openie_path.open("r", encoding="utf-8") as handle:
            self.openie_docs = json.load(handle).get("docs", [])

        self.n = int(self.g.vcount())
        self.node_key_to_vid: dict[str, int] = {}
        self.vid_to_node_key: dict[int, str] = {}
        self.passage_to_vid: dict[str, int] = {}
        self.vid_to_passage: dict[int, str] = {}
        self.chunk_vid_to_triple_entity_vids: dict[int, list[int]] = {}

        for vertex in self.g.vs:
            key = vertex["name"] if "name" in vertex.attributes() else vertex["hash_id"]
            key = str(key)
            self.node_key_to_vid[key] = int(vertex.index)
            self.vid_to_node_key[int(vertex.index)] = key
            content = str(vertex["content"]) if "content" in vertex.attributes() else ""
            if key.startswith("chunk-"):
                passage_id = extract_passage_id(content)
                if passage_id:
                    self.passage_to_vid[passage_id] = int(vertex.index)
                    self.vid_to_passage[int(vertex.index)] = passage_id

        for doc in self.openie_docs:
            chunk_key = str(doc.get("idx", ""))
            chunk_vid = self.node_key_to_vid.get(chunk_key)
            if chunk_vid is None:
                continue
            entity_vids: list[int] = []
            for triple in doc.get("extracted_triples", []):
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                for surface in (triple[0], triple[2]):
                    entity_key = entity_node_key(surface)
                    entity_vid = self.node_key_to_vid.get(entity_key)
                    if entity_vid is not None:
                        entity_vids.append(entity_vid)
            self.chunk_vid_to_triple_entity_vids[chunk_vid] = sorted(set(entity_vids))

    def pagerank(self, reset: np.ndarray, damping: float) -> np.ndarray:
        reset = clean_reset(reset)
        if reset.sum() <= 0:
            return np.zeros(self.n, dtype=np.float64)
        scores = self.g.personalized_pagerank(
            reset=(reset / reset.sum()).tolist(),
            damping=damping,
            weights="weight",
            directed=False,
        )
        return np.asarray(scores, dtype=np.float64)


class HippoInternalSeeder:
    def __init__(
        self,
        cache_dir: Path,
        ctx: GraphContext,
        llm_model: str,
        embedding_model: str,
        llm_base_url: str | None,
        embedding_base_url: str | None,
        include_passage_weights: bool,
    ) -> None:
        from hipporag import HippoRAG
        from hipporag.prompts.linking import get_query_instruction
        from hipporag.utils.misc_utils import min_max_normalize

        self.ctx = ctx
        self.include_passage_weights = include_passage_weights
        self.min_max_normalize = min_max_normalize
        self.get_query_instruction = get_query_instruction
        self.hippo = HippoRAG(
            save_dir=str(cache_dir),
            llm_model_name=llm_model,
            embedding_model_name=embedding_model,
            llm_base_url=llm_base_url or None,
            embedding_base_url=embedding_base_url or None,
        )
        self.hippo.prepare_retrieval_objects()

    def reset_for_query(self, query: str) -> tuple[np.ndarray, dict[str, Any]]:
        hippo = self.hippo
        hippo.get_query_embeddings([query])
        fact_scores = hippo.get_fact_scores(query)
        top_fact_indices, top_facts, _ = hippo.rerank_facts(query, fact_scores)

        reset = np.zeros(self.ctx.n, dtype=np.float64)
        linked_entity_count = 0
        for rank, fact in enumerate(top_facts):
            if len(fact) != 3:
                continue
            fact_score = fact_scores[top_fact_indices[rank]] if np.ndim(fact_scores) > 0 else float(fact_scores)
            for surface in (fact[0], fact[2]):
                key = entity_node_key(surface)
                vid = self.ctx.node_key_to_vid.get(key)
                if vid is None:
                    continue
                degree_docs = len(getattr(hippo, "ent_node_to_chunk_ids", {}).get(key, set()))
                weight = float(fact_score)
                if degree_docs > 0:
                    weight /= degree_docs
                reset[vid] += weight
                linked_entity_count += 1

        if self.include_passage_weights:
            query_embedding = hippo.query_to_embedding["passage"].get(query)
            if query_embedding is None:
                query_embedding = hippo.embedding_model.batch_encode(
                    query,
                    instruction=self.get_query_instruction("query_to_passage"),
                    norm=True,
                )
            doc_scores = np.dot(hippo.passage_embeddings, query_embedding.T)
            doc_scores = np.squeeze(doc_scores) if np.ndim(doc_scores) == 2 else doc_scores
            doc_scores = self.min_max_normalize(doc_scores)
            for local_doc_id, score in enumerate(doc_scores.tolist()):
                passage_key = hippo.passage_node_keys[local_doc_id]
                vid = self.ctx.node_key_to_vid.get(passage_key)
                if vid is not None:
                    reset[vid] += float(score) * float(hippo.global_config.passage_node_weight)

        return reset, {
            "top_fact_count": len(top_facts),
            "linked_entity_seed_count": linked_entity_count,
            "nonzero_seed_count": int(np.count_nonzero(reset)),
        }


def main() -> None:
    args = parse_args()
    pairs = resolve_dataset_pairs(args)
    reports = []
    for dataset_name, cache_dir, result_dir in pairs:
        reports.append(evaluate_dataset(dataset_name, cache_dir, result_dir, args))

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)

    print_summary(reports, args)
    print(f"\nFull report: {output}")


def evaluate_dataset(dataset_name: str, cache_dir: Path, result_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    print(f"\n=== {dataset_name}: {result_dir.name} ===", flush=True)
    start = time.time()
    ctx = GraphContext(cache_dir)
    print(
        f"graph loaded: {ctx.n} nodes, {len(ctx.passage_to_vid)} passages "
        f"[{time.time() - start:.1f}s]",
        flush=True,
    )

    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_by_id = {
        row["id"]: row
        for row in load_jsonl(result_dir / "retrieval_results.jsonl")
    }
    if args.max_queries:
        rows = rows[: args.max_queries]

    seeder = None
    if args.seed_source == "hipporag_internal":
        seeder = HippoInternalSeeder(
            cache_dir=cache_dir,
            ctx=ctx,
            llm_model=args.hipporag_llm_model,
            embedding_model=args.hipporag_embedding_model,
            llm_base_url=args.hipporag_llm_base_url,
            embedding_base_url=args.hipporag_embedding_base_url,
            include_passage_weights=bool(args.include_hippo_passage_weights),
        )

    aggregators = make_aggregators(args.lambdas, args.budgets)
    seed_stats: dict[str, list[float]] = defaultdict(list)
    bucket_counts: Counter[str] = Counter()
    errors = 0

    for index, row in enumerate(rows, start=1):
        try:
            gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
            if not gold:
                continue
            dense_ids = list(row.get("dense_ids") or [])[: args.dense_k]
            graph_ids = list(row.get("graph_ids") or [])
            dense_set = set(dense_ids)
            missed = gold - dense_set
            bucket = classify_query_bucket(row, retrieval_by_id.get(row["id"], {}), missed, args.dense_k)
            bucket_counts[bucket] += 1

            query_reset, query_meta = build_query_reset(row, graph_ids, ctx, seeder, args)
            bridge_reset = build_bridge_reset(ctx, dense_ids)
            dense_reset = build_dense_reset(ctx, dense_ids, row.get("dense_scores") or [])

            seed_stats["query_nonzero"].append(float(np.count_nonzero(query_reset)))
            seed_stats["bridge_nonzero"].append(float(np.count_nonzero(bridge_reset)))
            seed_stats["dense_nonzero"].append(float(np.count_nonzero(dense_reset)))
            for key, value in query_meta.items():
                if isinstance(value, (int, float)):
                    seed_stats[key].append(float(value))

            query_bridge_reset = combine_query_bridge_reset(
                query_reset=query_reset,
                bridge_reset=bridge_reset,
                bridge_share=float(args.bridge_share),
            )
            pr_query_bridge = ctx.pagerank(query_bridge_reset, damping=float(args.damping))
            pr_dense = ctx.pagerank(dense_reset, damping=float(args.damping))

            hippo_complement = [pid for pid in graph_ids if pid not in dense_set]
            oracle_complement = list(missed)

            for budget in args.budgets:
                evaluate_candidate_list(
                    aggregators["hipporag_cached"][budget],
                    row,
                    gold,
                    missed,
                    bucket,
                    dense_ids,
                    hippo_complement[:budget],
                    budget,
                    args.dense_k,
                )
                evaluate_candidate_list(
                    aggregators["oracle"][budget],
                    row,
                    gold,
                    missed,
                    bucket,
                    dense_ids,
                    oracle_complement[:budget],
                    budget,
                    args.dense_k,
                )

            for lam in args.lambdas:
                residual_scores = pr_query_bridge - float(lam) * pr_dense
                complement = rank_passage_complements(
                    ctx=ctx,
                    scores=residual_scores,
                    exclude=dense_set,
                    top_k=max(args.budgets),
                )
                method_name = residual_method_name(lam)
                for budget in args.budgets:
                    evaluate_candidate_list(
                        aggregators[method_name][budget],
                        row,
                        gold,
                        missed,
                        bucket,
                        dense_ids,
                        [pid for pid, _ in complement[:budget]],
                        budget,
                        args.dense_k,
                    )
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  [warn] {row.get('id')}: {exc}", flush=True)

        if index % 50 == 0:
            print(f"  progress {index}/{len(rows)} [{time.time() - start:.1f}s]", flush=True)

    report = {
        "dataset": dataset_name,
        "cache_dir": str(cache_dir),
        "result_dir": str(result_dir),
        "seed_source": args.seed_source,
        "include_hippo_passage_weights": bool(args.include_hippo_passage_weights),
        "dense_k": args.dense_k,
        "budgets": args.budgets,
        "lambdas": args.lambdas,
        "bridge_share": args.bridge_share,
        "damping": args.damping,
        "num_rows": len(rows),
        "errors": errors,
        "bucket_counts": dict(bucket_counts),
        "seed_coverage": summarize_seed_stats(seed_stats),
        "methods": summarize_aggregators(aggregators),
    }
    print(f"done {dataset_name} [{time.time() - start:.1f}s]", flush=True)
    return report


def build_query_reset(
    row: dict[str, Any],
    graph_ids: list[str],
    ctx: GraphContext,
    seeder: HippoInternalSeeder | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.seed_source == "hipporag_internal":
        if seeder is None:
            raise RuntimeError("hipporag_internal seed source requires a seeder.")
        return seeder.reset_for_query(str(row["question"]))

    reset = np.zeros(ctx.n, dtype=np.float64)
    for rank, passage_id in enumerate(graph_ids[: args.dense_k], start=1):
        vid = ctx.passage_to_vid.get(passage_id)
        if vid is not None:
            reset[vid] += 1.0 / rank
    return reset, {
        "cached_graph_seed_count": min(len(graph_ids), args.dense_k),
        "nonzero_seed_count": int(np.count_nonzero(reset)),
    }


def build_bridge_reset(ctx: GraphContext, dense_ids: list[str]) -> np.ndarray:
    reset = np.zeros(ctx.n, dtype=np.float64)
    for passage_id in dense_ids:
        chunk_vid = ctx.passage_to_vid.get(passage_id)
        if chunk_vid is None:
            continue
        for entity_vid in ctx.chunk_vid_to_triple_entity_vids.get(chunk_vid, []):
            reset[entity_vid] += 1.0
    return clean_reset(reset)


def build_dense_reset(ctx: GraphContext, dense_ids: list[str], dense_scores: list[float]) -> np.ndarray:
    reset = np.zeros(ctx.n, dtype=np.float64)
    positive_scores = normalize_positive(dense_scores[: len(dense_ids)])
    for index, passage_id in enumerate(dense_ids):
        vid = ctx.passage_to_vid.get(passage_id)
        if vid is None:
            continue
        if index < len(positive_scores):
            reset[vid] += positive_scores[index]
        else:
            reset[vid] += 1.0
    return clean_reset(reset)


def combine_query_bridge_reset(query_reset: np.ndarray, bridge_reset: np.ndarray, bridge_share: float) -> np.ndarray:
    query = clean_reset(query_reset)
    bridge = clean_reset(bridge_reset)
    query_sum = query.sum()
    bridge_sum = bridge.sum()
    if query_sum <= 0 and bridge_sum <= 0:
        return np.zeros_like(query)
    if query_sum <= 0:
        return bridge / bridge_sum
    if bridge_sum <= 0:
        return query / query_sum
    bridge_share = min(max(float(bridge_share), 0.0), 1.0)
    return (1.0 - bridge_share) * (query / query_sum) + bridge_share * (bridge / bridge_sum)


def rank_passage_complements(
    ctx: GraphContext,
    scores: np.ndarray,
    exclude: set[str],
    top_k: int,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for vid, passage_id in ctx.vid_to_passage.items():
        if passage_id in exclude:
            continue
        ranked.append((passage_id, float(scores[vid])))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


def evaluate_candidate_list(
    agg: dict[str, Any],
    row: dict[str, Any],
    gold: set[str],
    missed: set[str],
    bucket: str,
    dense_ids: list[str],
    complement_ids: list[str],
    budget: int,
    dense_k: int,
) -> None:
    final_ids = dedupe_preserve_order(dense_ids[: max(0, dense_k - budget)] + complement_ids)[:dense_k]
    final_recall = len(set(final_ids) & gold) / max(1, len(gold))
    has_missed = bool(missed)
    missed_recovery = len(set(complement_ids) & missed) / len(missed) if has_missed else None
    complement_hit = 1.0 if set(complement_ids) & missed else 0.0

    strata = ["ALL", f"gap_{bucket}"]
    label = row.get("label")
    if label is None:
        strata.append("label_tie_or_invalid")
    else:
        strata.append(f"label_{int(label)}")
    question_type = str(row.get("question_type", "unknown")).replace(" ", "_")
    strata.append(f"type_{question_type}")

    for stratum in strata:
        agg[stratum]["n"] += 1
        agg[stratum]["final_recall@5_sum"] += final_recall
        agg[stratum]["avg_complement_size_sum"] += len(complement_ids)
        if missed_recovery is not None:
            agg[stratum]["missed_n"] += 1
            agg[stratum]["missed_recovery_sum"] += missed_recovery
            agg[stratum]["complement_hit_sum"] += complement_hit


def classify_query_bucket(
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    missed: set[str],
    dense_k: int,
) -> str:
    if not missed:
        return "none"
    dense_passages = retrieval_row.get("retrieval", {}).get("dense", [])[:dense_k]
    dense_text = normalize_space(
        " ".join((p.get("title") or "") + " " + (p.get("text") or "") for p in dense_passages)
    )
    question = normalize_space(str(row.get("question", "")))
    gap_types = set()
    for gold_id in missed:
        gold_norm = normalize_space(gold_id)
        if gold_norm and gold_norm in question:
            gap_types.add("C")
        elif gold_norm and gold_norm in dense_text:
            gap_types.add("A")
        else:
            gap_types.add("B")
    if gap_types == {"A"}:
        return "A_bridge_visible"
    if gap_types == {"B"}:
        return "B_hop1_miss"
    if gap_types == {"C"}:
        return "C_query_entity"
    return "mixed"


def make_aggregators(lambdas: list[float], budgets: list[int]) -> dict[str, dict[int, dict[str, Any]]]:
    methods = ["hipporag_cached", "oracle"] + [residual_method_name(lam) for lam in lambdas]
    return {
        method: {
            budget: defaultdict(lambda: {
                "n": 0,
                "missed_n": 0,
                "final_recall@5_sum": 0.0,
                "missed_recovery_sum": 0.0,
                "complement_hit_sum": 0.0,
                "avg_complement_size_sum": 0.0,
            })
            for budget in budgets
        }
        for method in methods
    }


def summarize_aggregators(aggregators: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, by_budget in aggregators.items():
        output[method] = {}
        for budget, by_stratum in by_budget.items():
            output[method][f"B={budget}"] = {}
            for stratum, values in by_stratum.items():
                n = int(values["n"])
                missed_n = int(values["missed_n"])
                output[method][f"B={budget}"][stratum] = {
                    "n": n,
                    "missed_n": missed_n,
                    "final_recall@5": round(values["final_recall@5_sum"] / n, 4) if n else None,
                    "missed_recovery": (
                        round(values["missed_recovery_sum"] / missed_n, 4) if missed_n else None
                    ),
                    "complement_hit_rate": (
                        round(values["complement_hit_sum"] / missed_n, 4) if missed_n else None
                    ),
                    "avg_complement_size": round(values["avg_complement_size_sum"] / n, 4) if n else None,
                }
    return output


def summarize_seed_stats(seed_stats: dict[str, list[float]]) -> dict[str, Any]:
    output = {}
    for key, values in seed_stats.items():
        if not values:
            continue
        output[key] = {
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "zero_rate": round(float(np.mean([value == 0 for value in values])), 4),
        }
    return output


def print_summary(reports: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not reports:
        return
    selected_budget = min(args.budgets, key=lambda value: abs(value - 3))
    print(f"\n=== Summary: ALL, B={selected_budget} ===")
    for report in reports:
        methods = report["methods"]
        base = methods["hipporag_cached"][f"B={selected_budget}"]["ALL"]
        print(
            f"{report['dataset']}: Hippo complement finalR={base['final_recall@5']} "
            f"missedRec={base['missed_recovery']}"
        )
        for lam in args.lambdas:
            method = residual_method_name(lam)
            row = methods[method][f"B={selected_budget}"]["ALL"]
            print(
                f"  {method}: finalR={row['final_recall@5']} "
                f"missedRec={row['missed_recovery']} hit={row['complement_hit_rate']}"
            )


def resolve_dataset_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.result_dirs or args.cache_dirs:
        if len(args.result_dirs) != len(args.cache_dirs):
            raise ValueError("--result-dirs and --cache-dirs must have the same length.")
        return [
            (Path(result_dir).name, resolve_project_path(cache_dir), resolve_project_path(result_dir))
            for cache_dir, result_dir in zip(args.cache_dirs, args.result_dirs)
        ]
    pairs = []
    for dataset in args.datasets:
        cache_raw, result_raw = DATASET_MAP[dataset]
        pairs.append((dataset, resolve_project_path(cache_raw), resolve_project_path(result_raw)))
    return pairs


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
    if not match:
        return None
    return match.group(1).strip()


def entity_node_key(surface: Any) -> str:
    processed = text_processing(surface)
    return compute_mdhash_id(processed, prefix="entity-")


def clean_reset(reset: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(reset) | np.isinf(reset) | (reset < 0), 0.0, reset).astype(np.float64)


def normalize_positive(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    arr = clean_reset(arr - np.min(arr))
    if arr.sum() <= 0:
        return [1.0 / len(values)] * len(values)
    return (arr / arr.sum()).tolist()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def residual_method_name(lam: float) -> str:
    return f"residual_lambda={lam:g}"


if __name__ == "__main__":
    main()
