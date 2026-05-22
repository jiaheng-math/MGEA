from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class RetrievedPassage:
    id: str
    text: str
    score: float
    title: str | None = None
    source_doc_id: str | None = None


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must parse to a mapping: {path}")
    return payload


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def answer_in_passages(answer: str, passages: list[RetrievedPassage]) -> float:
    if not answer.strip():
        return 0.0
    answer_norm = normalize_text(answer)
    for passage in passages:
        if answer_norm in normalize_text(passage.text):
            return 1.0
    return 0.0


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required. Install dependencies from environment.yml.") from exc

    try:
        return spacy.load(model_name)
    except OSError:
        warnings.warn(
            "spaCy model 'en_core_web_sm' is not available. Falling back to a blank English pipeline "
            "with heuristic entity extraction."
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def extract_entities(text: str, nlp) -> list[str]:
    doc = nlp(text)
    entities: list[str] = []
    seen: set[str] = set()
    for ent in doc.ents:
        normalized = normalize_text(ent.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(normalized)
    if entities:
        return entities

    heuristic_entities = _heuristic_entities(text)
    for entity in heuristic_entities:
        if entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
    return entities


def _heuristic_entities(text: str) -> list[str]:
    pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b|\b\d{4}\b")
    blocked = {"who", "what", "when", "where", "which", "how", "did", "was", "is", "the"}
    output: list[str] = []
    seen: set[str] = set()
    for match in pattern.findall(text):
        normalized = normalize_text(match)
        if not normalized or normalized in blocked or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output
