from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def normalize_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    score: float = 0.0
    title: str | None = None
    source_doc_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Passage":
        passage_id = value.get("id")
        if passage_id is None:
            passage_id = value.get("source_doc_id") or value.get("title")
        if passage_id is None:
            raise ValueError("Every retrieved passage needs an id, source_doc_id, or title.")
        return cls(
            id=str(passage_id),
            text=str(value.get("text") or value.get("content") or ""),
            score=float(value.get("score", 0.0) or 0.0),
            title=str(value["title"]) if value.get("title") is not None else None,
            source_doc_id=(
                str(value["source_doc_id"])
                if value.get("source_doc_id") is not None
                else None
            ),
        )

    @property
    def evidence_keys(self) -> frozenset[str]:
        return frozenset(
            key
            for key in (
                normalize_key(self.id),
                normalize_key(self.source_doc_id),
                normalize_key(self.title),
            )
            if key
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_doc_id": self.source_doc_id,
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    dense: tuple[Passage, ...]
    graph: tuple[Passage, ...]
    gold_keys: frozenset[str] = frozenset()
    query_entities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Example":
        retrieval = value.get("retrieval")
        retrieval = retrieval if isinstance(retrieval, Mapping) else value
        dense_raw = retrieval.get("dense") or value.get("dense_passages") or []
        graph_raw = retrieval.get("graph") or value.get("graph_passages") or []
        gold_values = list(value.get("gold_passage_ids") or [])
        gold_values.extend(value.get("gold_titles") or [])
        gold_values.extend(value.get("gold_evidence") or [])
        entities = value.get("query_entities") or []
        return cls(
            id=str(value.get("id") or value.get("qid") or ""),
            question=str(value.get("question") or ""),
            dense=tuple(Passage.from_mapping(item) for item in dense_raw),
            graph=tuple(Passage.from_mapping(item) for item in graph_raw),
            gold_keys=frozenset(normalize_key(item) for item in gold_values if normalize_key(item)),
            query_entities=tuple(normalize_key(item) for item in entities if normalize_key(item)),
        )

    def covered_gold(self, passages: Sequence[Passage]) -> frozenset[str]:
        covered: set[str] = set()
        for passage in passages:
            covered.update(self.gold_keys & passage.evidence_keys)
        return frozenset(covered)

    def recall(self, passages: Sequence[Passage]) -> float:
        if not self.gold_keys:
            return 0.0
        return len(self.covered_gold(passages)) / len(self.gold_keys)


def load_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = Example.from_mapping(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
            if not example.id:
                raise ValueError(f"Missing example id at line {line_number}.")
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {path}.")
    return examples


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
