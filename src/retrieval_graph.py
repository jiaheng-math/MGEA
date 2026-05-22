from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import log
from pathlib import Path
from typing import Any, Sequence

from src.dataset import Passage
from src.retrieval_graph_bm25 import BM25GraphRetriever
from src.retrieval_graph_hipporag import HippoRAGGraphRetriever
from src.retrieval_graph_lightrag import LightRAGGraphRetriever
from src.utils import RetrievedPassage, extract_entities


class GraphRetriever:
    def __init__(self, corpus: Sequence[Passage], nlp) -> None:
        self.corpus = list(corpus)
        self.nlp = nlp
        self.passage_entities: dict[str, set[str]] = {}
        self.entity_to_passage_ids: dict[str, set[str]] = defaultdict(set)
        self.passage_lookup = {passage.id: passage for passage in self.corpus}
        self.corpus_size = len(self.corpus)
        self._build_index()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedPassage]:
        query_entity_list = list(dict.fromkeys(extract_entities(query, self.nlp)))
        query_entities = set(query_entity_list)
        seeded_passages = {
            passage_id
            for entity in query_entities
            for passage_id in self.entity_to_passage_ids.get(entity, set())
        }
        expanded_entity_weights = self._expanded_entity_weights(query_entities, seeded_passages)

        scored: list[tuple[float, str]] = []
        for passage in self.corpus:
            entities = self.passage_entities.get(passage.id, set())
            overlap = self._weighted_overlap(query_entities, entities)
            pair_bonus = self._pair_bonus(query_entity_list, entities)
            bridge_bonus = self._bridge_bonus(passage.id, entities, seeded_passages)
            expansion_bonus = self._expansion_bonus(entities, expanded_entity_weights, query_entities)
            final_score = float(overlap) + 0.3 * float(pair_bonus) + 0.2 * float(bridge_bonus) + 0.15 * float(expansion_bonus)
            scored.append((final_score, passage.id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        top_passages = scored[: min(top_k, len(scored))]

        results: list[RetrievedPassage] = []
        for score, passage_id in top_passages:
            passage = self.passage_lookup[passage_id]
            results.append(
                RetrievedPassage(
                    id=passage.id,
                    text=passage.text,
                    title=passage.title,
                    source_doc_id=passage.source_doc_id,
                    score=score,
                )
            )
        return results

    def _build_index(self) -> None:
        for passage in self.corpus:
            entities = set(extract_entities(self._format_text(passage), self.nlp))
            self.passage_entities[passage.id] = entities
            for entity in entities:
                self.entity_to_passage_ids[entity].add(passage.id)

    def _format_text(self, passage: Passage) -> str:
        if passage.title:
            return f"{passage.title}. {passage.text}"
        return passage.text

    def _bridge_bonus(self, passage_id: str, entities: set[str], seeded_passages: set[str]) -> int:
        if not entities or not seeded_passages:
            return 0

        neighbors = seeded_passages - {passage_id}
        if not neighbors:
            return 0

        for entity in entities:
            linked_passages = self.entity_to_passage_ids.get(entity, set())
            if linked_passages & neighbors:
                return 1
        return 0

    def _weighted_overlap(self, query_entities: set[str], passage_entities: set[str]) -> float:
        overlap = query_entities & passage_entities
        return sum(self._entity_weight(entity) for entity in overlap)

    def _pair_bonus(self, query_entities: list[str], passage_entities: set[str]) -> int:
        if len(query_entities) < 2:
            return 0
        return sum(1 for left, right in combinations(query_entities, 2) if left in passage_entities and right in passage_entities)

    def _entity_weight(self, entity: str) -> float:
        document_frequency = len(self.entity_to_passage_ids.get(entity, set()))
        if document_frequency == 0:
            return 0.0
        return log((1.0 + self.corpus_size) / (1.0 + document_frequency)) + 1.0

    def _expanded_entity_weights(self, query_entities: set[str], seeded_passages: set[str]) -> dict[str, float]:
        expanded_weights: dict[str, float] = {}
        for passage_id in seeded_passages:
            entities = self.passage_entities.get(passage_id, set())
            for entity in entities:
                if entity in query_entities:
                    continue
                expanded_weights[entity] = max(expanded_weights.get(entity, 0.0), self._entity_weight(entity))
        return expanded_weights

    def _expansion_bonus(
        self,
        passage_entities: set[str],
        expanded_entity_weights: dict[str, float],
        query_entities: set[str],
    ) -> float:
        bonus = 0.0
        for entity in passage_entities:
            if entity in query_entities:
                continue
            bonus += expanded_entity_weights.get(entity, 0.0)
        return bonus


def build_graph_retriever(corpus: Sequence[Passage], nlp, config: dict[str, Any], project_root: Path):
    backend = str(config.get("graph_backend", "simplified")).lower()
    if backend == "simplified":
        return GraphRetriever(corpus=corpus, nlp=nlp)
    if backend == "bm25":
        return BM25GraphRetriever(corpus=corpus)
    if backend == "hipporag":
        save_dir = Path(str(config.get("hipporag_save_dir", "hipporag_cache/default")))
        if not save_dir.is_absolute():
            save_dir = project_root / save_dir
        return HippoRAGGraphRetriever(
            corpus=corpus,
            save_dir=str(save_dir),
            llm_model_name=str(config.get("hipporag_llm_model", "gpt-4.1")),
            embedding_model_name=str(config.get("hipporag_embedding_model", "text-embedding-3-small")),
            llm_base_url=str(config.get("hipporag_llm_base_url", "")) or None,
            embedding_base_url=str(config.get("hipporag_embedding_base_url", "")) or None,
            rebuild_index=bool(config.get("hipporag_rebuild_index", False)),
        )
    if backend == "lightrag":
        working_dir = Path(str(config.get("lightrag_working_dir", "lightrag_cache/default")))
        if not working_dir.is_absolute():
            working_dir = project_root / working_dir
        return LightRAGGraphRetriever(
            corpus=corpus,
            working_dir=str(working_dir),
            llm_model_name=str(config.get("lightrag_llm_model", "gpt-4.1-mini")),
            embedding_model_name=str(config.get("lightrag_embedding_model", "text-embedding-3-small")),
            llm_base_url=str(config.get("lightrag_llm_base_url", "")) or None,
            embedding_base_url=str(config.get("lightrag_embedding_base_url", "")) or None,
            llm_api_key=str(config.get("lightrag_llm_api_key", "")) or None,
            embedding_api_key=str(config.get("lightrag_embedding_api_key", "")) or None,
            embedding_dim=(
                int(config["lightrag_embedding_dim"]) if config.get("lightrag_embedding_dim") is not None else None
            ),
            query_mode=str(config.get("lightrag_query_mode", "hybrid")),
            rebuild_index=bool(config.get("lightrag_rebuild_index", False)),
        )
    raise ValueError(f"Unsupported graph_backend: {backend}")
