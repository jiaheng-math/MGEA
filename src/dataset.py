from __future__ import annotations

import json
import random
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str | None = None
    source_doc_id: str | None = None


@dataclass(frozen=True)
class QASample:
    id: str
    question: str
    answer: str
    answer_aliases: list[str]
    passages: list[Passage]
    gold_passage_ids: list[str]
    gold_titles: list[str]
    dataset_name: str = "unknown"
    workload: str = "unknown"
    question_type: str = "unknown"


def load_dataset(dataset_path: str, subset_size: int | None, random_seed: int) -> list[QASample]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw_records = _load_raw_records(path)
    if subset_size is not None and subset_size > 0 and subset_size < len(raw_records):
        rng = random.Random(random_seed)
        raw_records = rng.sample(raw_records, subset_size)

    inferred_dataset_name = _infer_dataset_name(path)
    samples = [_normalize_record(record, idx, inferred_dataset_name) for idx, record in enumerate(raw_records)]
    samples = [sample for sample in samples if sample.passages]
    if not samples:
        raise ValueError("No valid samples were loaded from the dataset.")
    return samples


def load_shared_corpus(corpus_path: str) -> list[Passage]:
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Shared corpus file not found: {path}")

    raw_records = _load_raw_records(path)
    corpus: list[Passage] = []
    seen_doc_ids: set[str] = set()

    for idx, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"Shared corpus record {idx} is not a JSON object.")

        text = str(record.get("text", record.get("passage", record.get("content", "")))).strip()
        title = record.get("title")
        title = str(title).strip() if title is not None else None
        if not text:
            continue

        source_doc_id = str(record.get("source_doc_id", title or record.get("id", f"doc-{idx}")))
        passage = Passage(
            id=str(record.get("id", source_doc_id)),
            text=text,
            title=title,
            source_doc_id=source_doc_id,
        )
        canonical_id = _canonical_doc_id(passage)
        if canonical_id in seen_doc_ids:
            continue
        seen_doc_ids.add(canonical_id)
        corpus.append(
            Passage(
                id=canonical_id,
                text=passage.text,
                title=passage.title,
                source_doc_id=canonical_id,
            )
        )

    if not corpus:
        raise ValueError("No valid passages were loaded from the shared corpus file.")
    return corpus


def build_corpus(samples: list[QASample]) -> list[Passage]:
    corpus: list[Passage] = []
    seen_doc_ids: set[str] = set()
    for sample in samples:
        for passage in sample.passages:
            canonical_id = _canonical_doc_id(passage)
            if canonical_id in seen_doc_ids:
                continue
            seen_doc_ids.add(canonical_id)
            corpus.append(
                Passage(
                    id=canonical_id,
                    text=passage.text,
                    title=passage.title,
                    source_doc_id=canonical_id,
                )
            )
    return corpus


def _load_raw_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        raise ValueError("JSON dataset must be a list or contain a top-level 'data' list.")

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _normalize_record(record: dict[str, Any], fallback_idx: int, inferred_dataset_name: str) -> QASample:
    sample_id = str(record.get("id", f"sample-{fallback_idx}"))
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    answer_aliases = _extract_answer_aliases(record, answer)
    if not answer and answer_aliases:
        answer = answer_aliases[0]
    passages = _extract_passages(sample_id, record)
    gold_passage_ids, gold_titles = _extract_gold_targets(record, passages)
    dataset_name = _extract_dataset_name(record, inferred_dataset_name)
    question_type = _extract_question_type(record)
    workload = _extract_workload(record, dataset_name)

    if not question:
        raise ValueError(f"Sample {sample_id} is missing a question.")

    return QASample(
        id=sample_id,
        question=question,
        answer=answer,
        answer_aliases=answer_aliases,
        passages=passages,
        gold_passage_ids=gold_passage_ids,
        gold_titles=gold_titles,
        dataset_name=dataset_name,
        workload=workload,
        question_type=question_type,
    )


def _extract_answer_aliases(record: dict[str, Any], answer: str) -> list[str]:
    aliases: list[str] = []
    for key in ("gold_answers", "possible_answers", "answers", "aliases", "answer_aliases", "o_aliases"):
        aliases.extend(_coerce_answer_list(record.get(key)))
    aliases.extend(_coerce_answer_list(record.get("obj")))
    aliases.extend(_coerce_answer_list(answer))
    return _dedupe([alias.strip() for alias in aliases if alias.strip()])


def _coerce_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        output: list[str] = []
        for item in value:
            output.extend(_coerce_answer_list(item))
        return output
    if isinstance(value, tuple):
        return _coerce_answer_list(list(value))
    if isinstance(value, dict):
        output: list[str] = []
        for key in ("text", "answer", "value", "name"):
            if key in value:
                output.extend(_coerce_answer_list(value[key]))
        return output

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            return _coerce_answer_list(parsed)
    return [text]


def _extract_passages(sample_id: str, record: dict[str, Any]) -> list[Passage]:
    passages: list[Passage] = []
    raw_passages = record.get("passages")
    raw_context = record.get("context")

    if isinstance(raw_context, list) and raw_context and all(isinstance(item, list) and len(item) == 2 for item in raw_context):
        for idx, item in enumerate(raw_context):
            title = str(item[0]).strip() if item[0] is not None else None
            body = item[1]
            if isinstance(body, list):
                text = " ".join(str(sentence).strip() for sentence in body if str(sentence).strip())
            else:
                text = str(body).strip()
            if not text:
                continue
            passage_id = f"{sample_id}::p{idx}"
            source_doc_id = title or passage_id
            passages.append(Passage(id=passage_id, text=text, title=title, source_doc_id=source_doc_id))
        return passages

    candidate_list = raw_passages if raw_passages is not None else raw_context
    if not isinstance(candidate_list, list):
        raise ValueError(f"Sample {sample_id} must contain a list under 'context' or 'passages'.")

    for idx, item in enumerate(candidate_list):
        if isinstance(item, str):
            text = item.strip()
            title = None
            passage_id = f"{sample_id}::p{idx}"
        elif isinstance(item, dict):
            text = str(item.get("text", item.get("passage", item.get("content", "")))).strip()
            title = item.get("title")
            title = str(title).strip() if title is not None else None
            passage_id = str(item.get("id", f"{sample_id}::p{idx}"))
        else:
            continue

        if not text:
            continue
        source_doc_id = title or passage_id
        passages.append(Passage(id=passage_id, text=text, title=title, source_doc_id=source_doc_id))

    return passages


def _extract_gold_targets(record: dict[str, Any], passages: list[Passage]) -> tuple[list[str], list[str]]:
    gold_ids: list[str] = []
    gold_titles: list[str] = []

    raw_gold_passages = record.get("gold_passages") or record.get("supporting_passages")
    if isinstance(raw_gold_passages, list):
        for item in raw_gold_passages:
            if isinstance(item, str):
                gold_ids.append(item)
            elif isinstance(item, dict):
                if item.get("id") is not None:
                    gold_ids.append(str(item["id"]))
                elif item.get("title") is not None:
                    gold_titles.append(str(item["title"]))

    supporting_facts = record.get("supporting_facts")
    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            if isinstance(fact, list) and fact:
                gold_titles.append(str(fact[0]))
            elif isinstance(fact, dict) and fact.get("title") is not None:
                gold_titles.append(str(fact["title"]))

    title_to_ids = {}
    local_id_to_canonical: dict[str, str] = {}
    for passage in passages:
        local_id_to_canonical[passage.id] = _canonical_doc_id(passage)
        if passage.title:
            title_to_ids.setdefault(passage.title, []).append(_canonical_doc_id(passage))

    expanded_gold_ids = [
        local_id_to_canonical.get(gold_id, gold_id)
        for gold_id in gold_ids
    ]
    for title in gold_titles:
        expanded_gold_ids.extend(title_to_ids.get(title, []))

    return _dedupe(expanded_gold_ids), _dedupe(gold_titles)


def _canonical_doc_id(passage: Passage) -> str:
    if passage.source_doc_id:
        return passage.source_doc_id
    if passage.title:
        return passage.title
    return passage.id


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if "hotpot" in stem:
        return "hotpotqa"
    if "2wiki" in stem:
        return "2wikimultihopqa"
    if "popqa" in stem:
        return "popqa"
    if "nq" in stem or "natural_questions" in stem:
        return "nq"
    if "mixed" in stem:
        return "mixed"
    return "unknown"


def _extract_dataset_name(record: dict[str, Any], fallback: str) -> str:
    for key in ("dataset_name", "dataset", "source_dataset"):
        value = record.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    return fallback


def _extract_question_type(record: dict[str, Any]) -> str:
    for key in ("question_type", "type"):
        value = record.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    return "unknown"


def _extract_workload(record: dict[str, Any], dataset_name: str) -> str:
    for key in ("workload", "hop_type"):
        value = record.get(key)
        if value is not None:
            text = str(value).strip().lower()
            if text:
                return text
    if dataset_name in {"hotpotqa", "2wikimultihopqa"}:
        return "multi-hop"
    if dataset_name in {"popqa", "nq"}:
        return "single-hop"
    return "unknown"
