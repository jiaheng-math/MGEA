from __future__ import annotations

import re

import numpy as np

from src.utils import RetrievedPassage, extract_entities, normalize_text

WH_TYPES = ["who", "what", "when", "where", "which", "how", "other"]
COMPARISON_PATTERN = re.compile(
    r"\b(more than|less than|older than|before|after|than)\b",
    flags=re.IGNORECASE,
)
CONJUNCTION_PATTERN = re.compile(r"\b(and|or|but|while|whereas|although)\b", flags=re.IGNORECASE)


def query_feature_names() -> list[str]:
    return [
        "query_length_tokens",
        "query_entity_count",
        "conjunction_count",
        "has_comparison_cue",
        *[f"wh_type_{name}" for name in WH_TYPES],
    ]


def probe_feature_names() -> list[str]:
    return [
        "dense_top1_score",
        "dense_top1_top2_gap",
        "dense_topk_score_std",
        "dense_entity_coverage_ratio",
        "dense_unique_doc_count",
    ]


def extract_query_features(question: str, nlp) -> tuple[dict[str, float], list[str]]:
    query_entities = extract_entities(question, nlp)
    wh_type = _detect_wh_type(question)
    token_count = len(question.split())
    conjunction_count = len(CONJUNCTION_PATTERN.findall(question))
    has_comparison_cue = float(bool(COMPARISON_PATTERN.search(question)))

    features: dict[str, float] = {
        "query_length_tokens": float(token_count),
        "query_entity_count": float(len(query_entities)),
        "conjunction_count": float(conjunction_count),
        "has_comparison_cue": has_comparison_cue,
    }
    for candidate in WH_TYPES:
        features[f"wh_type_{candidate}"] = float(candidate == wh_type)

    return features, query_entities


def extract_probe_features(query_entities: list[str], dense_results: list[RetrievedPassage]) -> dict[str, float]:
    scores = [result.score for result in dense_results]
    top1_score = float(scores[0]) if scores else 0.0
    top2_score = float(scores[1]) if len(scores) > 1 else 0.0
    top1_top2_gap = top1_score - top2_score
    score_std = float(np.std(scores)) if scores else 0.0

    retrieved_text = " ".join(result.text for result in dense_results)
    retrieved_text_norm = normalize_text(retrieved_text)
    if query_entities:
        covered = sum(1 for entity in query_entities if entity in retrieved_text_norm)
        coverage_ratio = covered / len(query_entities)
    else:
        coverage_ratio = 0.0

    unique_doc_ids = {
        result.source_doc_id if result.source_doc_id is not None else result.id for result in dense_results
    }

    return {
        "dense_top1_score": float(top1_score),
        "dense_top1_top2_gap": float(top1_top2_gap),
        "dense_topk_score_std": float(score_std),
        "dense_entity_coverage_ratio": float(coverage_ratio),
        "dense_unique_doc_count": float(len(unique_doc_ids)),
    }


def _detect_wh_type(question: str) -> str:
    lowered = normalize_text(question)
    for wh_type in WH_TYPES[:-1]:
        if re.match(rf"^{wh_type}\b", lowered):
            return wh_type
    return "other"
