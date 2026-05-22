"""Dense-Bridge PPR: correction via dense-conditioned personalized PageRank.

Algorithm:
  1. Given a query and its dense top-5 chunks
  2. Extract bridges: entities mentioned in dense top-5 chunks minus entities
     already in the query (these are the "newly revealed" intermediate nodes)
  3. Build a seed distribution over entity nodes: query entities + bridges
  4. Set teleport mass on dense-top-5 chunk nodes to 0 (walker never restarts
     there; forced to discover NEW chunks)
  5. Run personalized PR on the graph
  6. Return top-B chunks not already in dense_top_5
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

# -------------------- shared utilities --------------------

def norm_entity(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def entity_hash(s: str) -> str:
    return "entity-" + hashlib.md5(norm_entity(s).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# -------------------- graph & openie indices --------------------

class GraphContext:
    def __init__(self, cache_dir: Path):
        with (cache_dir / "gpt-4.1_text-embedding-3-small" / "graph.pickle").open("rb") as f:
            self.g = pickle.load(f)
        with (cache_dir / "openie_results_ner_gpt-4.1.json").open() as f:
            openie = json.load(f)

        # hash_id -> vertex index
        self.hash_to_vid: dict[str, int] = {v["hash_id"]: i for i, v in enumerate(self.g.vs)}
        # passage_id -> chunk_vid
        self.passage_to_vid: dict[str, int] = {}
        for v in self.g.vs:
            if v["hash_id"].startswith("chunk-"):
                m = re.search(r"PASSAGE_ID::([^\n]+)", str(v["content"]))
                if m:
                    self.passage_to_vid[m.group(1).strip()] = v.index
        # chunk_vid -> passage_id
        self.vid_to_passage: dict[int, str] = {v: p for p, v in self.passage_to_vid.items()}
        # chunk_vid -> entity_vids (from openie)
        self.chunk_vid_to_ent_vids: dict[int, list[int]] = {}
        for doc in openie["docs"]:
            cvid = self.hash_to_vid.get(doc["idx"])
            if cvid is None:
                continue
            ent_vids: list[int] = []
            for surface in doc.get("extracted_entities", []):
                evid = self.hash_to_vid.get(entity_hash(surface))
                if evid is not None:
                    ent_vids.append(evid)
            self.chunk_vid_to_ent_vids[cvid] = ent_vids

        self.n = self.g.vcount()
        # cache chunk_vid set for filtering
        self.chunk_vid_set = set(self.vid_to_passage)


# -------------------- Dense-Bridge PPR --------------------

def extract_bridges(
    ctx: GraphContext,
    dense_passage_ids: list[str],
    query_entity_vids: set[int],
) -> dict[int, float]:
    """Bridges: entities that appear in dense_top_k chunks, minus query entities.
    Weight by number of dense chunks mentioning the entity (higher = more certain bridge).
    """
    counts: dict[int, int] = {}
    for pid in dense_passage_ids:
        cvid = ctx.passage_to_vid.get(pid)
        if cvid is None:
            continue
        for evid in ctx.chunk_vid_to_ent_vids.get(cvid, []):
            if evid in query_entity_vids:
                continue
            counts[evid] = counts.get(evid, 0) + 1
    # Return per-entity weight = count (to be normalized later)
    return {vid: float(c) for vid, c in counts.items()}


def extract_query_entity_vids(ctx: GraphContext, query_entity_strings: list[str]) -> set[int]:
    out: set[int] = set()
    for s in query_entity_strings:
        evid = ctx.hash_to_vid.get(entity_hash(s))
        if evid is not None:
            out.add(evid)
    return out


def dense_bridge_ppr(
    ctx: GraphContext,
    query_entity_vids: set[int],
    bridge_weights: dict[int, float],
    dense_passage_ids: list[str],
    damping: float = 0.5,
    bridge_share: float = 0.5,
    mask_mode: str = "teleport",
) -> np.ndarray:
    """Run personalized PR with residual teleport distribution.

    mask_mode:
      - "none":     no mask; seeds = query_entities + bridges
      - "teleport": dense chunks get 0 teleport mass (original Dense-Bridge)
      - "graph":    remove dense chunk nodes from the graph entirely before PPR
                     (walker cannot pass through them, so mass flows only in
                     non-dense regions)
    """
    reset = np.zeros(ctx.n, dtype=np.float64)

    # Query entity mass: (1 - bridge_share), uniform
    if query_entity_vids:
        per_q = (1.0 - bridge_share) / len(query_entity_vids)
        for vid in query_entity_vids:
            reset[vid] += per_q
    # Bridge mass: bridge_share, weighted by dense-chunk co-occurrence count
    if bridge_weights:
        total = sum(bridge_weights.values())
        for vid, w in bridge_weights.items():
            reset[vid] += bridge_share * (w / total)

    # Fallback: no bridges → all mass on query entities
    if reset.sum() == 0 and query_entity_vids:
        for vid in query_entity_vids:
            reset[vid] = 1.0 / len(query_entity_vids)
    if reset.sum() == 0:
        return np.zeros(ctx.n)

    # Teleport mask: zero out dense chunk nodes in reset vector
    if mask_mode == "teleport":
        for pid in dense_passage_ids:
            cvid = ctx.passage_to_vid.get(pid)
            if cvid is not None:
                reset[cvid] = 0.0
        if reset.sum() == 0:  # if everything was masked (shouldn't happen)
            for vid in query_entity_vids:
                reset[vid] = 1.0 / max(1, len(query_entity_vids))

    reset = reset / reset.sum()

    if mask_mode == "graph":
        # Delete dense chunk nodes from graph, re-index reset vector
        dense_vids = [ctx.passage_to_vid[p] for p in dense_passage_ids
                      if p in ctx.passage_to_vid]
        keep_mask = np.ones(ctx.n, dtype=bool)
        keep_mask[dense_vids] = False
        sub = ctx.g.subgraph([i for i in range(ctx.n) if keep_mask[i]])
        # Build mapping from original vid to subgraph vid
        old_to_new = {}
        new_vid = 0
        for i in range(ctx.n):
            if keep_mask[i]:
                old_to_new[i] = new_vid
                new_vid += 1
        sub_reset = np.zeros(sub.vcount(), dtype=np.float64)
        for i, m in enumerate(reset):
            if m > 0 and i in old_to_new:
                sub_reset[old_to_new[i]] += m
        if sub_reset.sum() == 0:
            return np.zeros(ctx.n)
        sub_reset = sub_reset / sub_reset.sum()
        pr_sub = np.asarray(sub.personalized_pagerank(
            reset=sub_reset.tolist(), damping=damping,
            weights="weight", directed=False,
        ))
        # Map back to full-graph indexing
        pr = np.zeros(ctx.n, dtype=np.float64)
        for old_i, new_i in old_to_new.items():
            pr[old_i] = pr_sub[new_i]
        return pr

    # Default path: "none" or "teleport"
    pr = ctx.g.personalized_pagerank(
        reset=reset.tolist(),
        damping=damping,
        weights="weight",
        directed=False,
    )
    return np.asarray(pr)


def rank_correction_chunks(
    ctx: GraphContext,
    pr: np.ndarray,
    exclude_passage_ids: set[str],
    top_b: int,
) -> list[tuple[str, float]]:
    """Return top-B passage ids (not in exclude) ranked by PR score."""
    scored: list[tuple[str, float]] = []
    for cvid, pid in ctx.vid_to_passage.items():
        if pid in exclude_passage_ids:
            continue
        scored.append((pid, float(pr[cvid])))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_b]


def standard_ppr(
    ctx: GraphContext,
    query_entity_vids: set[int],
    damping: float = 0.5,
) -> np.ndarray:
    """Baseline: PPR seeded only on query entities, no dense masking.
    This mirrors what standard HippoRAG does conceptually (sans its own
    seed linking via embedding). Used as our in-script apples-to-apples
    baseline so Dense-Bridge PPR's gain is purely from the teleport change.
    """
    if not query_entity_vids:
        return np.zeros(ctx.n)
    reset = np.zeros(ctx.n, dtype=np.float64)
    per_q = 1.0 / len(query_entity_vids)
    for vid in query_entity_vids:
        reset[vid] = per_q
    pr = ctx.g.personalized_pagerank(
        reset=reset.tolist(),
        damping=damping,
        weights="weight",
        directed=False,
    )
    return np.asarray(pr)


def dense_seeded_ppr(
    ctx: GraphContext,
    dense_passage_ids: list[str],
    damping: float = 0.5,
) -> np.ndarray:
    """PPR seeded uniformly on dense top-k chunk nodes.

    Represents the neighborhood that dense retrieval has already covered.
    Subtracting this from r_Q gives a 'residual' signal that downweights
    chunks dense already implicitly captured.
    """
    dense_vids = [
        ctx.passage_to_vid[p] for p in dense_passage_ids if p in ctx.passage_to_vid
    ]
    if not dense_vids:
        return np.zeros(ctx.n)
    reset = np.zeros(ctx.n, dtype=np.float64)
    per = 1.0 / len(dense_vids)
    for vid in dense_vids:
        reset[vid] = per
    pr = ctx.g.personalized_pagerank(
        reset=reset.tolist(),
        damping=damping,
        weights="weight",
        directed=False,
    )
    return np.asarray(pr)


def residual_score(
    pr_query: np.ndarray,
    pr_dense: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Ranking signal r_Q - lam * r_D.

    Not a probability distribution. Still monotone-safe for top-k ranking
    via rank_correction_chunks (which only sorts, never normalizes).
    """
    return pr_query - lam * pr_dense


# -------------------- entrypoint for one query --------------------

def correct_one_query(
    ctx: GraphContext,
    query_entity_strings: list[str],
    dense_passage_ids: list[str],
    top_b: int = 3,
    damping: float = 0.5,
    bridge_share: float = 0.5,
) -> dict[str, Any]:
    qvids = extract_query_entity_vids(ctx, query_entity_strings)
    bridges = extract_bridges(ctx, dense_passage_ids, qvids)
    pr = dense_bridge_ppr(
        ctx=ctx,
        query_entity_vids=qvids,
        bridge_weights=bridges,
        dense_passage_ids=dense_passage_ids,
        damping=damping,
        bridge_share=bridge_share,
    )
    corrections = rank_correction_chunks(
        ctx=ctx,
        pr=pr,
        exclude_passage_ids=set(dense_passage_ids),
        top_b=top_b,
    )
    return {
        "correction_passage_ids": [p for p, _ in corrections],
        "correction_scores": [s for _, s in corrections],
        "num_query_entities": len(qvids),
        "num_bridges": len(bridges),
    }
