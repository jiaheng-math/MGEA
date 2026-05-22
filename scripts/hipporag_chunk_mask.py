"""HippoRAG variant that suppresses dense-covered chunk stopping points.

Design:
  HippoRAG's PPR reset distribution = phrase_weights + passage_weights.
  - phrase_weights: set on entity nodes via dense-linked facts (the 'bridge'
    signal we want to KEEP, since dense-exposed bridge entities are the best
    hint toward missed gold).
  - passage_weights: set on passage nodes via dense top-k scores.

  To operationalize "suppress chunk stopping points dense already covered,
  without touching bridge entities", we zero out passage_weights entries for
  any passage_id in the caller-provided dense mask, then let HippoRAG run its
  normal PPR on the modified seed.

  Phrase/entity seeds are untouched, so HippoRAG's native dense-aware fact
  linking is preserved (that was the strongest ingredient in the earlier
  smoke test).

Exports:
  HippoRAGMasked.retrieve_with_mask(queries, dense_ids_per_query, num_to_retrieve)
"""
from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

from hipporag import HippoRAG
from hipporag.utils.misc_utils import compute_mdhash_id, min_max_normalize


PASSAGE_ID_RE = re.compile(r"PASSAGE_ID::([^\n]+)")


class HippoRAGMasked(HippoRAG):
    """HippoRAG subclass supporting dense-chunk masking on the PPR seed."""

    def _build_pilot_id_index(self) -> None:
        """Build pilot-id indices (cached on instance):
          _pilot_id_to_vidx:       pilot passage_id -> graph vertex idx
          _pilot_id_to_chunk_key:  pilot passage_id -> hipporag chunk hash-key
        """
        if getattr(self, "_pilot_id_to_vidx", None) is not None:
            return
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        vidx_map: dict[str, int] = {}
        key_map: dict[str, str] = {}
        for key in self.passage_node_keys:
            row = self.chunk_embedding_store.get_row(key)
            content = row["content"] if isinstance(row, dict) else row
            m = PASSAGE_ID_RE.search(str(content))
            if m:
                pid = m.group(1).strip()
                vidx = self.node_name_to_vertex_idx.get(key)
                if vidx is not None:
                    vidx_map[pid] = vidx
                    key_map[pid] = key
        self._pilot_id_to_vidx = vidx_map
        self._pilot_id_to_chunk_key = key_map

    def graph_search_with_fact_entities(  # type: ignore[override]
        self,
        query: str,
        link_top_k: int,
        query_fact_scores: np.ndarray,
        top_k_facts,
        top_k_fact_indices,
        passage_node_weight: float = 0.05,
        dense_mask_passage_ids: Optional[set[str]] = None,
        chunk_mask: bool = False,
        entity_mask_mode: str = "none",
        entity_mask_threshold: float = 1.0,
    ):
        """Replicates the superclass body, with optional seed masking.

        Masking options (both driven by `dense_mask_passage_ids`, typically
        dense top-5):

          chunk_mask: zero passage_weights entries for those pilot passage_ids.
            (Shown to be ineffective because passage_node_weight=0.05 already
            makes passage contribution marginal vs phrase_weights.)

          entity_mask_mode:
            - "none":       do not mask any phrase_weights.
            - "saturation": for each entity endpoint of a reranked fact, compute
                saturated_frac = |entity.chunks ∩ dense_chunks| / |entity.chunks|.
                If saturated_frac >= entity_mask_threshold, zero its
                phrase_weights entry. Intuition: this entity's local
                neighborhood is already fully covered by dense, so anchoring
                PPR here wastes mass — suppress it and force mass onto
                under-covered bridge entities.
        """
        linking_score_map: dict = {}
        phrase_scores: dict = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        # Pre-compute dense chunk-key set for entity saturation check
        dense_chunk_keys: set[str] = set()
        if dense_mask_passage_ids and entity_mask_mode != "none":
            self._build_pilot_id_index()
            for pid in dense_mask_passage_ids:
                k = self._pilot_id_to_chunk_key.get(pid)
                if k is not None:
                    dense_chunk_keys.add(k)

        # --- phrase weights from reranked facts (unchanged ordering; mask below) ---
        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = (
                query_fact_scores[top_k_fact_indices[rank]]
                if query_fact_scores.ndim > 0 else query_fact_scores
            )
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score
                    ent_chunks = self.ent_node_to_chunk_ids.get(phrase_key, set())
                    if len(ent_chunks) > 0:
                        phrase_weights[phrase_id] /= len(ent_chunks)
                phrase_scores.setdefault(phrase, []).append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(
                link_top_k, phrase_weights, linking_score_map
            )

        # --- entity-saturation mask: applied AFTER get_top_k_weights so we
        # don't violate its nonzero-count assertion. Zero out phrase_weights
        # entries for entities whose chunk-neighborhood is already saturated
        # by dense top-k. linking_score_map is kept (display-only) to avoid
        # touching the superclass's internal accounting. ---
        if entity_mask_mode == "saturation" and dense_chunk_keys:
            for phrase_key, ent_chunks in self.ent_node_to_chunk_ids.items():
                if not ent_chunks:
                    continue
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
                if phrase_id is None or phrase_weights[phrase_id] == 0.0:
                    continue
                sat = len(ent_chunks & dense_chunk_keys) / len(ent_chunks)
                if sat >= entity_mask_threshold:
                    phrase_weights[phrase_id] = 0.0

        # --- passage weights from dense scoring, with optional chunk-mask ---
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        mask_vidxs: set[int] = set()
        if dense_mask_passage_ids and chunk_mask:
            self._build_pilot_id_index()
            for pid in dense_mask_passage_ids:
                v = self._pilot_id_to_vidx.get(pid)
                if v is not None:
                    mask_vidxs.add(v)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            if passage_node_id in mask_vidxs:
                # chunk-mask: suppress dense-covered passage nodes from the seed
                continue
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            row = self.chunk_embedding_store.get_row(passage_node_key)
            passage_node_text = row["content"] if isinstance(row, dict) else row
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        node_weights = phrase_weights + passage_weights

        if len(linking_score_map) > 30:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
            )

        # Guard: if everything was masked (e.g. tiny corpus), fall back to no mask
        if sum(node_weights) <= 0:
            # defensive: rebuild passage_weights without mask, keep phrase_weights
            for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
                passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = (
                    normalized_dpr_sorted_scores[i] * passage_node_weight
                )
            node_weights = phrase_weights + passage_weights

        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights, damping=self.global_config.damping
        )
        self.ppr_time += (time.time() - ppr_start)

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def retrieve_with_mask(
        self,
        queries: list[str],
        dense_ids_per_query: list[list[str]],
        num_to_retrieve: Optional[int] = None,
        exclude_dense_from_output: bool = True,
        chunk_mask: bool = False,
        entity_mask_mode: str = "none",
        entity_mask_threshold: float = 1.0,
    ) -> list[list[tuple[str, float]]]:
        """Run HippoRAG retrieval with per-query dense-chunk masking on the PPR seed.

        Args:
          queries: list of query strings.
          dense_ids_per_query: aligned with queries; each inner list is the pilot
            passage_id set that dense retrieval already returned (typically
            dense top-5). Entities mentioned by these chunks are NOT suppressed
            (only the chunk nodes themselves are).
          num_to_retrieve: per-query output length after optional dense filtering.
          exclude_dense_from_output: drop dense_ids from the final ranking so
            the correction budget is purely "new" chunks.

        Returns:
          list of per-query rankings: [(pilot_passage_id, ppr_score), ...]
        """
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self._build_pilot_id_index()
        self.get_query_embeddings(queries)

        results: list[list[tuple[str, float]]] = []
        for q_idx, query in tqdm(
            enumerate(queries), desc="Retrieving (masked)", total=len(queries)
        ):
            dense_ids = set(dense_ids_per_query[q_idx] or [])

            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, _ = self.rerank_facts(query, query_fact_scores)

            if len(top_k_facts) == 0:
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight,
                    dense_mask_passage_ids=dense_ids,
                    chunk_mask=chunk_mask,
                    entity_mask_mode=entity_mask_mode,
                    entity_mask_threshold=entity_mask_threshold,
                )

            # Convert internal doc ids -> pilot passage ids; optionally drop dense
            ranked: list[tuple[str, float]] = []
            for idx, score in zip(sorted_doc_ids.tolist(), sorted_doc_scores.tolist()):
                key = self.passage_node_keys[idx]
                row = self.chunk_embedding_store.get_row(key)
                content = row["content"] if isinstance(row, dict) else row
                m = PASSAGE_ID_RE.search(str(content))
                if not m:
                    continue
                pid = m.group(1).strip()
                if exclude_dense_from_output and pid in dense_ids:
                    continue
                ranked.append((pid, float(score)))
                if len(ranked) >= num_to_retrieve:
                    break
            results.append(ranked)

        return results
