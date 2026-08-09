from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .schema import Example, Passage, normalize_key


CONTENT_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were",
    "are", "which", "who", "what", "when", "where", "how", "into", "than",
    "its", "their", "his", "her", "has", "had", "have", "did", "does",
}
WH_TYPES = ("who", "what", "when", "where", "which", "how", "other")
COMPARISON_PATTERN = re.compile(
    r"\b(more than|less than|older than|before|after|than)\b", re.IGNORECASE
)
CONJUNCTION_PATTERN = re.compile(
    r"\b(and|or|but|while|whereas|although)\b", re.IGNORECASE
)
RELATION_PATTERN = re.compile(
    r"\b(and|or|both|between|before|after)\b", re.IGNORECASE
)


TRIAGE_FEATURE_NAMES = (
    "query_length_tokens",
    "query_entity_count",
    "conjunction_count",
    "has_comparison_cue",
    *(f"wh_type_{name}" for name in WH_TYPES),
    "dense_top1_score",
    "dense_top1_top2_gap",
    "dense_top5_score_std",
    "dense_entity_coverage_ratio",
    "dense_unique_doc_count",
)

SLOT_QUERY_FEATURE_NAMES = (
    "dense_graph_top5_overlap",
    "graph_top5_not_in_dense_top5",
    "dense_top1_score",
    "dense_top1_top2_gap",
    "dense_top5_score_std",
    "graph_top1_score",
    "graph_top1_top2_gap",
    "graph_top5_score_std",
    "dense_entity_coverage_ratio",
    "dense_unique_doc_count",
    "query_length_tokens",
    "query_entity_count",
    "conjunction_count",
    "has_comparison_cue",
    "query_regex_token_count",
    "query_comma_count",
    "query_relation_cue",
)

SLOT_LOCAL_FEATURE_NAMES = (
    "graph_rank",
    "reciprocal_graph_rank",
    "distance_from_base",
    "graph_score",
    "graph_score_z",
    "previous_graph_score_drop",
    "next_graph_score_drop",
    "score_vs_rank5",
    "score_vs_base_mean",
    "appears_in_dense",
    "appears_in_dense_top5",
    "appears_in_dense_top20",
    "reciprocal_dense_rank",
    "dense_rank",
    "dense_score",
    "dense_score_z",
    "candidate_length",
    "lexical_novelty_vs_base",
    "jaccard_vs_base",
    "query_term_coverage",
    "query_candidate_jaccard",
)

SLOT_FEATURE_NAMES = SLOT_QUERY_FEATURE_NAMES + SLOT_LOCAL_FEATURE_NAMES


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalize_key(text))
        if len(token) > 2 and token not in CONTENT_STOPWORDS
    ]


def passage_terms(passage: Passage) -> set[str]:
    return set(content_tokens(f"{passage.title or passage.id} {passage.text}"))


def query_terms(example: Example) -> set[str]:
    return set(content_tokens(example.question))


def query_entities(example: Example) -> tuple[str, ...]:
    if example.query_entities:
        return example.query_entities
    candidates = re.findall(
        r"\b(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3})\b", example.question
    )
    first_word = example.question.split(maxsplit=1)[0].casefold() if example.question else ""
    return tuple(
        normalize_key(candidate)
        for candidate in candidates
        if normalize_key(candidate) not in WH_TYPES and normalize_key(candidate) != first_word
    )


def dense_probe_features(example: Example, base_k: int = 5) -> list[float]:
    dense = example.dense[:base_k]
    scores = [passage.score for passage in dense]
    entities = query_entities(example)
    retrieved_text = normalize_key(" ".join(passage.text for passage in dense))
    covered = sum(1 for entity in entities if entity and entity in retrieved_text)
    unique_docs = {
        passage.source_doc_id or passage.id
        for passage in dense
    }
    return [
        score_at(scores, 0),
        score_at(scores, 0) - score_at(scores, 1),
        float(np.std(scores)) if scores else 0.0,
        covered / len(entities) if entities else 0.0,
        float(len(unique_docs)),
    ]


def triage_features(example: Example, base_k: int = 5) -> list[float]:
    lowered = normalize_key(example.question)
    wh_type = "other"
    for candidate in WH_TYPES[:-1]:
        if re.match(rf"^{candidate}\b", lowered):
            wh_type = candidate
            break
    entities = query_entities(example)
    query_values = [
        float(len(example.question.split())),
        float(len(entities)),
        float(len(CONJUNCTION_PATTERN.findall(example.question))),
        float(bool(COMPARISON_PATTERN.search(example.question))),
        *(float(candidate == wh_type) for candidate in WH_TYPES),
    ]
    return query_values + dense_probe_features(example, base_k)


def slot_query_features(example: Example, base_k: int = 5) -> list[float]:
    dense_ids = [passage.id for passage in example.dense]
    graph_ids = [passage.id for passage in example.graph]
    dense_scores = [passage.score for passage in example.dense]
    graph_scores = [passage.score for passage in example.graph]
    dense_top = set(dense_ids[:base_k])
    graph_top = set(graph_ids[:base_k])
    probe = dense_probe_features(example, base_k)
    entities = query_entities(example)
    return [
        len(dense_top & graph_top) / max(1, base_k),
        float(len([pid for pid in graph_ids[:base_k] if pid not in dense_top])),
        score_at(dense_scores, 0),
        score_at(dense_scores, 0) - score_at(dense_scores, 1),
        float(np.std(dense_scores[:base_k])) if dense_scores else 0.0,
        score_at(graph_scores, 0),
        score_at(graph_scores, 0) - score_at(graph_scores, 1),
        float(np.std(graph_scores[:base_k])) if graph_scores else 0.0,
        probe[3],
        probe[4],
        float(len(example.question.split())),
        float(len(entities)),
        float(len(CONJUNCTION_PATTERN.findall(example.question))),
        float(bool(COMPARISON_PATTERN.search(example.question))),
        float(len(re.findall(r"\w+", example.question))),
        float(example.question.count(",")),
        float(bool(RELATION_PATTERN.search(example.question))),
    ]


def slot_local_features(
    example: Example,
    passage: Passage,
    *,
    base_k: int = 5,
    max_k: int = 20,
) -> list[float]:
    dense_rank = {candidate.id: index + 1 for index, candidate in enumerate(example.dense)}
    graph_rank = {candidate.id: index + 1 for index, candidate in enumerate(example.graph)}
    rank = graph_rank.get(passage.id, 999)
    graph_index = rank - 1 if rank != 999 else 999
    candidate_dense_rank = dense_rank.get(passage.id)
    dense_index = candidate_dense_rank - 1 if candidate_dense_rank is not None else 999
    graph_scores = [candidate.score for candidate in example.graph]
    dense_scores = [candidate.score for candidate in example.dense]
    graph_score = score_at(graph_scores, graph_index)
    dense_score = score_at(dense_scores, dense_index)
    candidate_terms = passage_terms(passage)
    base_terms: set[str] = set()
    for base_passage in example.graph[:base_k]:
        base_terms.update(passage_terms(base_passage))
    question_terms = query_terms(example)
    return [
        float(rank),
        reciprocal_rank(rank),
        float(rank - base_k),
        graph_score,
        zscore_at(graph_scores, graph_index),
        score_at(graph_scores, graph_index - 1) - graph_score if graph_index > 0 else 0.0,
        graph_score - score_at(graph_scores, graph_index + 1),
        graph_score - score_at(graph_scores, base_k - 1),
        graph_score - (float(np.mean(graph_scores[:base_k])) if graph_scores[:base_k] else 0.0),
        float(candidate_dense_rank is not None),
        float(candidate_dense_rank is not None and candidate_dense_rank <= base_k),
        float(candidate_dense_rank is not None and candidate_dense_rank <= max_k),
        reciprocal_rank(candidate_dense_rank),
        float(candidate_dense_rank if candidate_dense_rank is not None else 999),
        dense_score,
        zscore_at(dense_scores, dense_index),
        len(candidate_terms) / 100.0,
        len(candidate_terms - base_terms) / max(1, len(candidate_terms)),
        jaccard(candidate_terms, base_terms),
        len(candidate_terms & question_terms) / max(1, len(question_terms)),
        jaccard(candidate_terms, question_terms),
    ]


def slot_features(
    example: Example,
    passage: Passage,
    *,
    base_k: int = 5,
    max_k: int = 20,
) -> list[float]:
    return slot_query_features(example, base_k) + slot_local_features(
        example, passage, base_k=base_k, max_k=max_k
    )


def score_at(scores: Sequence[float], index: int) -> float:
    return float(scores[index]) if 0 <= index < len(scores) else 0.0


def zscore_at(scores: Sequence[float], index: int) -> float:
    if not (0 <= index < len(scores)) or not scores:
        return 0.0
    std = float(np.std(scores))
    if std <= 1e-12:
        return 0.0
    return (float(scores[index]) - float(np.mean(scores))) / std


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / float(rank)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
