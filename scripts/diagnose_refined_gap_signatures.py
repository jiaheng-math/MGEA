"""Diagnose refined dense-miss gap signatures.

The earlier A/B/C buckets are useful but coarse:

  A: missed gold title appears in dense top-k text
  B: missed gold title appears neither in query nor dense top-k text
  C: missed gold title appears in the query

This script keeps those legacy buckets for continuity, but decomposes each
dense miss into more actionable signatures:

  - question / dense-title / dense-text visibility
  - dense and graph rank bands for the missed gold passage
  - optional graph reachability from dense-exposed bridge entity seeds
  - per-query mixed-signature combinations

It is a diagnostic script. It does not call LLMs and can run from saved routing
and retrieval outputs. Add --with-graph-reachability to load HippoRAG's graph
and compute bounded shortest-path distances from dense bridge seeds to missed
gold passage nodes.

Action taxonomy:

  bridge_exposed_reachable
      Dense evidence exposes the missed title/entity and dense bridge seeds can
      reach its graph passage node.
  bridge_exposed_unreachable
      Dense evidence exposes the missed title/entity but bounded graph
      reachability fails.
  bridge_exposed_surface
      Dense evidence exposes the missed title/entity, but reachability was not
      computed.
  query_anchored_graph_solved / unanchored_graph_solved
      The missed gold is already in graph top-5. Default HippoRAG completion is
      usually the right operator.
  query_anchored_graph_recoverable / unanchored_graph_recoverable
      The missed gold is in graph ranks 6-20. This is a graph budget or rerank
      issue, not a bridge-expansion issue.
  graph_deep_recoverable
      The missed gold is in graph ranks 21-100.
  graph_absent_or_alias
      The missed gold is absent from the saved graph ranking, or the passage id
      does not align with graph nodes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict, deque
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


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
            raise FileNotFoundError(f"Missing HippoRAG graph pickle: {graph_path}")
        if not openie_path.exists():
            raise FileNotFoundError(f"Missing OpenIE cache: {openie_path}")

        with graph_path.open("rb") as handle:
            self.g = pickle.load(handle)
        with openie_path.open("r", encoding="utf-8") as handle:
            self.openie_docs = json.load(handle).get("docs", [])

        self.n = int(self.g.vcount())
        self.node_key_to_vid: dict[str, int] = {}
        self.passage_to_vid: dict[str, int] = {}
        self.chunk_vid_to_triple_entity_vids: dict[int, list[int]] = {}

        for vertex in self.g.vs:
            key = vertex["name"] if "name" in vertex.attributes() else vertex["hash_id"]
            key = str(key)
            self.node_key_to_vid[key] = int(vertex.index)
            content = str(vertex["content"]) if "content" in vertex.attributes() else ""
            if key.startswith("chunk-"):
                passage_id = extract_passage_id(content)
                if passage_id:
                    self.passage_to_vid[passage_id] = int(vertex.index)

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
                    entity_vid = self.node_key_to_vid.get(entity_node_key(surface))
                    if entity_vid is not None:
                        entity_vids.append(entity_vid)
            self.chunk_vid_to_triple_entity_vids[chunk_vid] = sorted(set(entity_vids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose refined gap signatures from saved runs.")
    parser.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    parser.add_argument("--result-dirs", nargs="*", default=[])
    parser.add_argument("--cache-dirs", nargs="*", default=[])
    parser.add_argument("--dense-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--with-graph-reachability", action="store_true")
    parser.add_argument("--max-bridge-distance", type=int, default=3)
    parser.add_argument("--output", default="results/refined_gap_signatures.json")
    parser.add_argument("--per-missed-output", default="results/refined_gap_signatures_per_missed.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    per_missed_rows: list[dict[str, Any]] = []

    for dataset, cache_dir, result_dir in resolve_dataset_pairs(args):
        report, rows = diagnose_dataset(dataset, cache_dir, result_dir, args)
        reports.append(report)
        per_missed_rows.extend(rows)

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    per_missed_output = resolve_project_path(args.per_missed_output)
    per_missed_output.parent.mkdir(parents=True, exist_ok=True)
    with per_missed_output.open("w", encoding="utf-8") as handle:
        for row in per_missed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_summary(reports)
    print(f"\nFull report: {output}")
    print(f"Per-missed rows: {per_missed_output}")


def diagnose_dataset(
    dataset: str,
    cache_dir: Path,
    result_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(result_dir / "routing_rows.jsonl")
    retrieval_by_id = {row["id"]: row for row in load_jsonl(result_dir / "retrieval_results.jsonl")}
    if args.max_queries:
        rows = rows[: args.max_queries]

    ctx = None
    adjacency = None
    if args.with_graph_reachability:
        ctx = GraphContext(cache_dir)
        adjacency = ctx.g.get_adjlist(mode="all")

    old_bucket_counts: Counter[str] = Counter()
    per_missed_class_counts: Counter[str] = Counter()
    per_query_class_counts: Counter[str] = Counter()
    old_x_action: dict[str, Counter[str]] = defaultdict(Counter)
    old_x_graph_band: dict[str, Counter[str]] = defaultdict(Counter)
    old_x_dense_band: dict[str, Counter[str]] = defaultdict(Counter)
    old_x_visibility: dict[str, Counter[str]] = defaultdict(Counter)
    mixed_combo_counts: Counter[str] = Counter()
    bridge_distance_counts: Counter[str] = Counter()
    per_missed_rows: list[dict[str, Any]] = []

    for row in rows:
        qid = row["id"]
        gold = set(row.get("gold_passage_ids") or row.get("gold_titles") or [])
        dense_ids = list(row.get("dense_ids") or [])
        graph_ids = list(row.get("graph_ids") or [])
        dense_top = dense_ids[: args.dense_k]
        missed = sorted(gold - set(dense_top))

        retrieval_row = retrieval_by_id.get(qid, {})
        legacy_bucket = classify_query_bucket(row, retrieval_row, set(missed), args.dense_k)
        old_bucket_counts[legacy_bucket] += 1
        if not missed:
            per_query_class_counts["none"] += 1
            continue

        dense_passages = retrieval_row.get("retrieval", {}).get("dense", [])[: args.dense_k]
        dense_title_blob = norm(" ".join(p.get("title") or "" for p in dense_passages))
        dense_text_blob = norm(" ".join(p.get("text") or "" for p in dense_passages))
        question = norm(row.get("question") or "")
        dense_rank = rank_map(dense_ids)
        graph_rank = rank_map(graph_ids)

        bridge_distances: dict[str, int | None] = {}
        if ctx is not None and adjacency is not None:
            bridge_distances = bridge_seed_distances(
                ctx=ctx,
                adjacency=adjacency,
                dense_ids=dense_top,
                targets=missed,
                max_distance=args.max_bridge_distance,
            )

        per_query_actions = []
        per_query_signatures = []
        for gold_id in missed:
            signature = missed_signature(
                gold_id=gold_id,
                question=question,
                dense_title_blob=dense_title_blob,
                dense_text_blob=dense_text_blob,
                dense_rank=dense_rank.get(gold_id),
                graph_rank=graph_rank.get(gold_id),
                bridge_distance=bridge_distances.get(gold_id),
                with_graph_reachability=args.with_graph_reachability,
                max_bridge_distance=args.max_bridge_distance,
            )
            action_class = action_class_for_signature(signature)
            per_query_actions.append(action_class)
            per_query_signatures.append(signature["compact_signature"])

            per_missed_class_counts[action_class] += 1
            old_x_action[legacy_bucket][action_class] += 1
            old_x_graph_band[legacy_bucket][signature["graph_band"]] += 1
            old_x_dense_band[legacy_bucket][signature["dense_band"]] += 1
            old_x_visibility[legacy_bucket][signature["visibility"]] += 1
            bridge_distance_counts[signature["bridge_distance_band"]] += 1

            per_missed_rows.append(
                {
                    "dataset": dataset,
                    "id": qid,
                    "question": row.get("question"),
                    "gold_id": gold_id,
                    "legacy_bucket": legacy_bucket,
                    "action_class": action_class,
                    **signature,
                }
            )

        action_set = sorted(set(per_query_actions))
        if len(action_set) == 1:
            per_query_class = action_set[0]
        else:
            per_query_class = "multi_evidence_conflict"
            mixed_combo_counts[" + ".join(action_set)] += 1
        per_query_class_counts[per_query_class] += 1

        if legacy_bucket == "mixed":
            mixed_combo_counts["legacy:" + " + ".join(sorted(set(per_query_signatures)))] += 1

    report = {
        "dataset": dataset,
        "n": len(rows),
        "dense_k": args.dense_k,
        "with_graph_reachability": bool(args.with_graph_reachability),
        "max_bridge_distance": args.max_bridge_distance,
        "old_bucket_counts": dict(old_bucket_counts),
        "per_missed_action_counts": dict(per_missed_class_counts),
        "per_query_action_counts": dict(per_query_class_counts),
        "old_bucket_x_action": nested_counter_to_dict(old_x_action),
        "old_bucket_x_graph_band": nested_counter_to_dict(old_x_graph_band),
        "old_bucket_x_dense_band": nested_counter_to_dict(old_x_dense_band),
        "old_bucket_x_visibility": nested_counter_to_dict(old_x_visibility),
        "bridge_distance_counts": dict(bridge_distance_counts),
        "mixed_combo_top": dict(mixed_combo_counts.most_common(30)),
    }
    return report, per_missed_rows


def missed_signature(
    gold_id: str,
    question: str,
    dense_title_blob: str,
    dense_text_blob: str,
    dense_rank: int | None,
    graph_rank: int | None,
    bridge_distance: int | None,
    with_graph_reachability: bool,
    max_bridge_distance: int,
) -> dict[str, Any]:
    gold_norm = norm(gold_id)
    q_visible = bool(gold_norm and gold_norm in question)
    dense_title_visible = bool(gold_norm and gold_norm in dense_title_blob)
    dense_text_visible = bool(gold_norm and gold_norm in dense_text_blob)
    dense_band = rank_band(dense_rank)
    graph_band = rank_band(graph_rank)

    if not with_graph_reachability:
        bridge_distance_band = "not_computed"
        bridge_reachable = None
    elif bridge_distance is None:
        bridge_distance_band = f">{max_bridge_distance}_or_unreachable"
        bridge_reachable = False
    else:
        bridge_distance_band = str(bridge_distance)
        bridge_reachable = True

    visibility_parts = []
    if q_visible:
        visibility_parts.append("Q")
    if dense_title_visible:
        visibility_parts.append("D_title")
    if dense_text_visible:
        visibility_parts.append("D_text")
    visibility = "+".join(visibility_parts) if visibility_parts else "none"

    compact = f"vis={visibility}|g={graph_band}|d={dense_band}|bd={bridge_distance_band}"
    return {
        "q_visible": q_visible,
        "dense_title_visible": dense_title_visible,
        "dense_text_visible": dense_text_visible,
        "visibility": visibility,
        "dense_rank": dense_rank,
        "dense_band": dense_band,
        "graph_rank": graph_rank,
        "graph_band": graph_band,
        "bridge_distance": bridge_distance,
        "bridge_distance_band": bridge_distance_band,
        "bridge_reachable": bridge_reachable,
        "compact_signature": compact,
    }


def action_class_for_signature(signature: dict[str, Any]) -> str:
    dense_text_visible = bool(signature["dense_text_visible"] or signature["dense_title_visible"])
    q_visible = bool(signature["q_visible"])
    graph_rank = signature["graph_rank"]
    dense_rank = signature["dense_rank"]
    bridge_reachable = signature["bridge_reachable"]

    if dense_text_visible:
        if bridge_reachable is True:
            return "bridge_exposed_reachable"
        if bridge_reachable is False:
            return "bridge_exposed_unreachable"
        return "bridge_exposed_surface"
    if graph_rank is not None and graph_rank <= 5:
        return "query_anchored_graph_solved" if q_visible else "unanchored_graph_solved"
    if graph_rank is not None and graph_rank <= 20:
        return "query_anchored_graph_recoverable" if q_visible else "unanchored_graph_recoverable"
    if graph_rank is not None and graph_rank <= 100:
        return "graph_deep_recoverable"
    if dense_rank is not None and dense_rank > 5 and dense_rank <= 20:
        return "dense_near_miss"
    if graph_rank is None:
        return "graph_absent_or_alias"
    return "graph_rank_gt100"


def bridge_seed_distances(
    ctx: GraphContext,
    adjacency: list[list[int]],
    dense_ids: list[str],
    targets: list[str],
    max_distance: int,
) -> dict[str, int | None]:
    bridge_reset = build_bridge_reset(ctx, dense_ids)
    sources = [int(idx) for idx in np.flatnonzero(bridge_reset)]
    target_vid_to_pid = {
        ctx.passage_to_vid[pid]: pid
        for pid in targets
        if pid in ctx.passage_to_vid
    }
    out: dict[str, int | None] = {pid: None for pid in targets}
    if not sources or not target_vid_to_pid:
        return out

    remaining = set(target_vid_to_pid)
    seen = set(sources)
    queue: deque[tuple[int, int]] = deque((source, 0) for source in sources)
    while queue and remaining:
        vid, dist = queue.popleft()
        if vid in remaining:
            out[target_vid_to_pid[vid]] = dist
            remaining.remove(vid)
            continue
        if dist >= max_distance:
            continue
        for nxt in adjacency[vid]:
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return out


def build_bridge_reset(ctx: GraphContext, dense_ids: list[str]) -> np.ndarray:
    reset = np.zeros(ctx.n, dtype=np.float64)
    for passage_id in dense_ids:
        chunk_vid = ctx.passage_to_vid.get(passage_id)
        if chunk_vid is None:
            continue
        for entity_vid in ctx.chunk_vid_to_triple_entity_vids.get(chunk_vid, []):
            reset[entity_vid] += 1.0
    return np.where(np.isnan(reset) | np.isinf(reset) | (reset < 0), 0.0, reset).astype(np.float64)


def classify_query_bucket(
    row: dict[str, Any],
    retrieval_row: dict[str, Any],
    missed: set[str],
    dense_k: int,
) -> str:
    if not missed:
        return "none"
    dense_passages = retrieval_row.get("retrieval", {}).get("dense", [])[:dense_k]
    dense_text = norm(
        " ".join((p.get("title") or "") + " " + (p.get("text") or "") for p in dense_passages)
    )
    question = norm(str(row.get("question", "")))
    gap_types = set()
    for gold_id in missed:
        gold_norm = norm(gold_id)
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


def rank_map(ids: list[str]) -> dict[str, int]:
    out = {}
    for rank, pid in enumerate(ids, start=1):
        out.setdefault(pid, rank)
    return out


def rank_band(rank: int | None) -> str:
    if rank is None:
        return "absent"
    if rank <= 5:
        return "top5"
    if rank <= 20:
        return "6-20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    return ">100"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def text_processing(text: Any) -> str:
    if isinstance(text, list):
        return " ".join(text_processing(item) for item in text)
    if not isinstance(text, str):
        text = str(text)
    return re.sub("[^A-Za-z0-9 ]", " ", text.lower()).strip()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def entity_node_key(surface: Any) -> str:
    return compute_mdhash_id(text_processing(surface), prefix="entity-")


def extract_passage_id(content: str) -> str | None:
    match = re.search(r"PASSAGE_ID::([^\n]+)", content)
    if not match:
        return None
    return match.group(1).strip()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nested_counter_to_dict(data: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(counter) for key, counter in sorted(data.items())}


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


def print_summary(reports: list[dict[str, Any]]) -> None:
    for report in reports:
        print("\n" + "=" * 88)
        print(f"DATASET {report['dataset']} n={report['n']} dense_k={report['dense_k']}")
        print(f"legacy buckets: {report['old_bucket_counts']}")
        print(f"per-query action: {report['per_query_action_counts']}")
        print(f"per-missed action: {report['per_missed_action_counts']}")
        print("\nold bucket x action")
        for bucket, counts in report["old_bucket_x_action"].items():
            print(f"  {bucket}: {counts}")
        print("\nold bucket x graph rank band")
        for bucket, counts in report["old_bucket_x_graph_band"].items():
            print(f"  {bucket}: {counts}")
        print("\nold bucket x visibility")
        for bucket, counts in report["old_bucket_x_visibility"].items():
            print(f"  {bucket}: {counts}")
        if report["with_graph_reachability"]:
            print(f"\nbridge distance: {report['bridge_distance_counts']}")
        print("\ntop mixed/action combos")
        for combo, count in report["mixed_combo_top"].items():
            print(f"  {count:>4}  {combo}")


if __name__ == "__main__":
    main()
