from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from pathlib import Path
from typing import Any


HTML_TAG_RE = re.compile(r"^<[^>]+>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Natural Questions single-hop benchmark subset with real long-answer "
            "Wikipedia passages as retrieval evidence."
        )
    )
    parser.add_argument("--input", required=True, help="Input NQ JSON/JSONL/JSONL.GZ file.")
    parser.add_argument("--queries-output", required=True, help="Output sampled query JSON file.")
    parser.add_argument("--corpus-output", required=True, help="Output shared corpus JSON file.")
    parser.add_argument("--subset-size", type=int, default=500, help="Number of valid NQ queries to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-passage-tokens",
        type=int,
        default=384,
        help=(
            "Maximum number of NQ document tokens kept for each long-answer passage. "
            "The window is centered on the first short answer when available."
        ),
    )
    parser.add_argument(
        "--allow-no-short-answer",
        action="store_true",
        help="Keep records with a long answer but no short answer by using the object/long answer as answer text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    records = load_records(input_path)
    normalized = []
    skipped = 0
    for raw_index, record in enumerate(records):
        row = normalize_record(
            record,
            raw_index,
            allow_no_short_answer=args.allow_no_short_answer,
            max_passage_tokens=args.max_passage_tokens,
        )
        if row is None:
            skipped += 1
            continue
        normalized.append(row)

    if len(normalized) < args.subset_size:
        raise ValueError(
            f"Only {len(normalized)} valid NQ rows found after filtering; requested {args.subset_size}. "
            "Use a larger input split or pass --allow-no-short-answer if appropriate."
        )

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(normalized)), args.subset_size))
    subset = [normalized[index] for index in selected_indices]
    corpus = build_shared_corpus(subset)

    queries_output = Path(args.queries_output)
    corpus_output = Path(args.corpus_output)
    queries_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    queries_output.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    corpus_output.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Prepared NQ single-hop shared-corpus benchmark")
    print(f"Input file: {input_path}")
    print(f"Raw records: {len(records)}")
    print(f"Valid records: {len(normalized)}")
    print(f"Skipped records: {skipped}")
    print(f"Subset size: {len(subset)}")
    print(f"Shared corpus docs: {len(corpus)}")
    print(f"Queries output: {queries_output}")
    print(f"Corpus output: {corpus_output}")
    print(f"Seed: {args.seed}")
    print(f"Max passage tokens: {args.max_passage_tokens}")


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "examples", "records"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise ValueError("JSON input must be a list or contain data/examples/records list.")

    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_record(
    record: dict[str, Any],
    raw_index: int,
    *,
    allow_no_short_answer: bool,
    max_passage_tokens: int,
) -> dict[str, Any] | None:
    sample_id = str(record.get("id", record.get("example_id", f"nq-{raw_index}")))
    question = str(record.get("question", record.get("question_text", ""))).strip()
    if not question:
        return None

    simplified = normalize_simplified_record(record, sample_id, question, max_passage_tokens=max_passage_tokens)
    if simplified is not None:
        return simplified

    token_texts, html_mask = extract_document_tokens(record)
    if not token_texts:
        return None

    title = str(record.get("document_title", record.get("title", sample_id))).strip() or sample_id
    annotation = choose_annotation(record)
    if annotation is None:
        return None

    long_answer = annotation.get("long_answer") if isinstance(annotation.get("long_answer"), dict) else {}
    start_token = int(long_answer.get("start_token", -1))
    end_token = int(long_answer.get("end_token", -1))
    if start_token < 0 or end_token <= start_token:
        return None

    gold_answers = extract_short_answers(annotation, token_texts, html_mask)
    if not gold_answers and allow_no_short_answer:
        preview_start, preview_end = crop_window(start_token, end_token, None, None, max_passage_tokens)
        preview_text = detokenize_tokens(token_texts[preview_start:preview_end], html_mask[preview_start:preview_end])
        gold_answers = [preview_text] if preview_text else []
    if not gold_answers:
        return None

    answer_span = first_short_answer_span(annotation)
    answer_start, answer_end = answer_span if answer_span is not None else (None, None)
    window_start, window_end = crop_window(start_token, end_token, answer_start, answer_end, max_passage_tokens)
    passage_text = detokenize_tokens(token_texts[window_start:window_end], html_mask[window_start:window_end])
    if not passage_text:
        return None

    source_doc_id = canonical_source_doc_id(title, sample_id)
    return make_query_record(
        sample_id=sample_id,
        question=question,
        answer=gold_answers[0],
        gold_answers=gold_answers,
        title=title,
        text=passage_text,
        source_doc_id=source_doc_id,
    )


def normalize_simplified_record(
    record: dict[str, Any],
    sample_id: str,
    question: str,
    *,
    max_passage_tokens: int,
) -> dict[str, Any] | None:
    passage = first_simplified_passage(record)
    if passage is None:
        return None

    answers = coerce_answer_list(record.get("gold_answers"))
    answers.extend(coerce_answer_list(record.get("answers")))
    answers.extend(coerce_answer_list(record.get("answer")))
    if not answers:
        return None

    title = str(passage.get("title") or record.get("document_title") or sample_id).strip()
    text = truncate_words(
        str(passage.get("text") or passage.get("passage") or passage.get("content") or "").strip(),
        max_passage_tokens,
    )
    if not text:
        return None
    source_doc_id = str(passage.get("source_doc_id") or passage.get("id") or canonical_source_doc_id(title, sample_id))
    return make_query_record(
        sample_id=sample_id,
        question=question,
        answer=answers[0],
        gold_answers=dedupe(answers),
        title=title,
        text=text,
        source_doc_id=source_doc_id,
    )


def first_simplified_passage(record: dict[str, Any]) -> dict[str, Any] | None:
    passages = record.get("passages")
    if isinstance(passages, list):
        for item in passages:
            parsed = parse_passage_item(item, "passage")
            if parsed is not None:
                return parsed

    context = record.get("context")
    if isinstance(context, list):
        for item in context:
            parsed = parse_passage_item(item, "context")
            if parsed is not None:
                return parsed
    if isinstance(context, str) and context.strip():
        return {"title": str(record.get("document_title", record.get("title", ""))).strip(), "text": context.strip()}
    return None


def parse_passage_item(item: Any, fallback_title: str) -> dict[str, Any] | None:
    if isinstance(item, dict):
        text = str(item.get("text", item.get("passage", item.get("content", "")))).strip()
        if text:
            return dict(item)
    if isinstance(item, list) and len(item) == 2:
        title = str(item[0]).strip() if item[0] is not None else fallback_title
        body = item[1]
        text = " ".join(str(part).strip() for part in body if str(part).strip()) if isinstance(body, list) else str(body).strip()
        if text:
            return {"title": title, "text": text}
    if isinstance(item, str) and item.strip():
        return {"title": fallback_title, "text": item.strip()}
    return None


def choose_annotation(record: dict[str, Any]) -> dict[str, Any] | None:
    annotations = record.get("annotations")
    if not isinstance(annotations, list):
        return None
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        long_answer = annotation.get("long_answer")
        if isinstance(long_answer, dict) and int(long_answer.get("start_token", -1)) >= 0:
            if extract_yes_no_answer(annotation) or annotation.get("short_answers"):
                return annotation
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        long_answer = annotation.get("long_answer")
        if isinstance(long_answer, dict) and int(long_answer.get("start_token", -1)) >= 0:
            return annotation
    return None


def extract_document_tokens(record: dict[str, Any]) -> tuple[list[str], list[bool]]:
    document_tokens = record.get("document_tokens")
    if isinstance(document_tokens, list):
        tokens = []
        html_mask = []
        for item in document_tokens:
            if isinstance(item, dict):
                token = str(item.get("token", "")).strip()
                is_html = bool(item.get("html_token", False)) or HTML_TAG_RE.match(token) is not None
            else:
                token = str(item).strip()
                is_html = HTML_TAG_RE.match(token) is not None
            tokens.append(token)
            html_mask.append(is_html)
        return tokens, html_mask

    document_text = str(record.get("document_text", "")).strip()
    if document_text:
        tokens = document_text.split()
        return tokens, [HTML_TAG_RE.match(token) is not None for token in tokens]
    return [], []


def extract_short_answers(annotation: dict[str, Any], token_texts: list[str], html_mask: list[bool]) -> list[str]:
    yes_no = extract_yes_no_answer(annotation)
    if yes_no:
        return [yes_no]

    answers: list[str] = []
    short_answers = annotation.get("short_answers")
    if isinstance(short_answers, list):
        for answer in short_answers:
            if not isinstance(answer, dict):
                continue
            start_token = int(answer.get("start_token", -1))
            end_token = int(answer.get("end_token", -1))
            if start_token < 0 or end_token <= start_token:
                continue
            text = detokenize_tokens(token_texts[start_token:end_token], html_mask[start_token:end_token])
            if text:
                answers.append(text)
    return dedupe(answers)


def first_short_answer_span(annotation: dict[str, Any]) -> tuple[int, int] | None:
    short_answers = annotation.get("short_answers")
    if not isinstance(short_answers, list):
        return None
    for answer in short_answers:
        if not isinstance(answer, dict):
            continue
        start_token = int(answer.get("start_token", -1))
        end_token = int(answer.get("end_token", -1))
        if start_token >= 0 and end_token > start_token:
            return start_token, end_token
    return None


def crop_window(
    passage_start: int,
    passage_end: int,
    answer_start: int | None,
    answer_end: int | None,
    max_tokens: int,
) -> tuple[int, int]:
    if max_tokens <= 0 or passage_end - passage_start <= max_tokens:
        return passage_start, passage_end

    if answer_start is None or answer_end is None:
        return passage_start, passage_start + max_tokens

    center = (answer_start + answer_end) // 2
    window_start = max(passage_start, center - max_tokens // 2)
    window_end = min(passage_end, window_start + max_tokens)
    window_start = max(passage_start, window_end - max_tokens)
    return window_start, window_end


def extract_yes_no_answer(annotation: dict[str, Any]) -> str | None:
    value = str(annotation.get("yes_no_answer", "NONE")).strip().lower()
    if value in {"yes", "no"}:
        return value
    return None


def detokenize_tokens(tokens: list[str], html_mask: list[bool]) -> str:
    clean_tokens = [
        token
        for token, is_html in zip(tokens, html_mask)
        if token and not is_html and not HTML_TAG_RE.match(token)
    ]
    text = " ".join(clean_tokens)
    text = text.replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+'s\b", "'s", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_query_record(
    *,
    sample_id: str,
    question: str,
    answer: str,
    gold_answers: list[str],
    title: str,
    text: str,
    source_doc_id: str,
) -> dict[str, Any]:
    passage = {
        "id": source_doc_id,
        "title": title,
        "text": text,
        "source_doc_id": source_doc_id,
    }
    return {
        "id": sample_id,
        "question": question,
        "answer": answer,
        "gold_answers": dedupe(gold_answers),
        "dataset_name": "nq",
        "workload": "single-hop",
        "question_type": "single-hop",
        "passages": [passage],
        "gold_passages": [{"id": source_doc_id, "title": title}],
    }


def build_shared_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for row in rows:
        for passage in row.get("passages", []):
            source_doc_id = str(passage["source_doc_id"])
            existing = docs.get(source_doc_id)
            if existing is None:
                docs[source_doc_id] = dict(passage)
                continue
            text = str(passage.get("text", "")).strip()
            if text and text not in existing["text"]:
                existing["text"] = f"{existing['text']} {text}".strip()
    return list(docs.values())


def canonical_source_doc_id(title: str, sample_id: str) -> str:
    title = title.strip()
    return title if title else sample_id


def coerce_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        answers: list[str] = []
        for item in value:
            answers.extend(coerce_answer_list(item))
        return answers
    if isinstance(value, dict):
        answers: list[str] = []
        for key in ("text", "answer", "value"):
            if key in value:
                answers.extend(coerce_answer_list(value[key]))
        return answers
    text = str(value).strip()
    return [text] if text else []


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


if __name__ == "__main__":
    main()
