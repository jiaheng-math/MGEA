from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a sampled query file plus a shared retrieval corpus by merging all context/passages "
            "across the sampled queries."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL dataset file.")
    parser.add_argument("--queries-output", required=True, help="Output JSON file for sampled query records.")
    parser.add_argument("--corpus-output", required=True, help="Output JSON file for the shared corpus.")
    parser.add_argument("--subset-size", type=int, required=True, help="Number of sampled queries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--dataset-name",
        default="auto",
        help="Dataset name annotation. Use 'auto' to infer from the input filename.",
    )
    parser.add_argument(
        "--workload",
        default="auto",
        help="Workload annotation written into sampled query records: auto, single-hop, or multi-hop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    queries_output = Path(args.queries_output)
    corpus_output = Path(args.corpus_output)

    records = load_records(input_path)
    validate_subset_size(records, args.subset_size)
    subset = sample_records(records, args.subset_size, args.seed)
    dataset_name = infer_dataset_name(input_path) if args.dataset_name == "auto" else str(args.dataset_name).strip().lower()
    workload = infer_workload(dataset_name) if args.workload == "auto" else str(args.workload).strip().lower()

    annotated_subset = [annotate_record(record, dataset_name, workload) for record in subset]
    shared_corpus = build_shared_corpus(annotated_subset)

    queries_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    with queries_output.open("w", encoding="utf-8") as handle:
        json.dump(annotated_subset, handle, indent=2, ensure_ascii=False)
    with corpus_output.open("w", encoding="utf-8") as handle:
        json.dump(shared_corpus, handle, indent=2, ensure_ascii=False)

    print_summary(
        input_path=input_path,
        queries_output=queries_output,
        corpus_output=corpus_output,
        total_records=len(records),
        subset=annotated_subset,
        shared_corpus=shared_corpus,
        seed=args.seed,
        dataset_name=dataset_name,
        workload=workload,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError("Input dataset must be a JSON list, JSONL file, or a JSON object with top-level 'data'.")


def validate_subset_size(records: list[dict[str, Any]], subset_size: int) -> None:
    if subset_size <= 0:
        raise ValueError("subset-size must be positive.")
    if subset_size > len(records):
        raise ValueError(f"subset-size={subset_size} is larger than dataset size {len(records)}.")


def sample_records(records: list[dict[str, Any]], subset_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(records)), subset_size))
    return [records[idx] for idx in selected_indices]


def infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if "hotpot" in stem:
        return "hotpotqa"
    if "2wiki" in stem:
        return "2wikimultihopqa"
    if "popqa" in stem:
        return "popqa"
    if "nq" in stem or "natural_questions" in stem:
        return "nq"
    return "unknown"


def infer_workload(dataset_name: str) -> str:
    if dataset_name in {"hotpotqa", "2wikimultihopqa"}:
        return "multi-hop"
    if dataset_name in {"popqa", "nq"}:
        return "single-hop"
    return "unknown"


def annotate_record(record: dict[str, Any], dataset_name: str, workload: str) -> dict[str, Any]:
    output = dict(record)
    output.setdefault("dataset_name", dataset_name)
    output.setdefault("workload", workload)
    return output


def build_shared_corpus(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_docs: dict[str, dict[str, Any]] = {}

    for record in records:
        for doc in iterate_documents(record):
            source_doc_id = doc["source_doc_id"]
            existing = merged_docs.get(source_doc_id)
            if existing is None:
                merged_docs[source_doc_id] = dict(doc)
                continue

            existing_text = existing.get("text", "")
            new_text = doc.get("text", "")
            if new_text and new_text not in existing_text:
                existing["text"] = " ".join(part for part in [existing_text, new_text] if part).strip()
            if not existing.get("title") and doc.get("title"):
                existing["title"] = doc["title"]

    return list(merged_docs.values())


def iterate_documents(record: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    context = record.get("context")
    if isinstance(context, dict):
        for title, body in context.items():
            text = collapse_text(body)
            if text:
                documents.append(make_document(title=title, text=text))
    elif isinstance(context, list):
        for idx, item in enumerate(context):
            documents.extend(parse_document_item(item, idx))

    passages = record.get("passages")
    if isinstance(passages, list):
        for idx, item in enumerate(passages):
            documents.extend(parse_document_item(item, idx))

    return documents


def parse_document_item(item: Any, fallback_idx: int) -> list[dict[str, Any]]:
    if isinstance(item, list) and len(item) == 2:
        title = str(item[0]).strip() if item[0] is not None else f"doc-{fallback_idx}"
        text = collapse_text(item[1])
        return [make_document(title=title, text=text)] if text else []

    if isinstance(item, dict):
        title = item.get("title")
        title = str(title).strip() if title is not None else None
        source_doc_id = item.get("source_doc_id") or item.get("id") or title or f"doc-{fallback_idx}"
        text = collapse_text(item.get("text", item.get("passage", item.get("content", item.get("sentences", "")))))
        if text:
            return [make_document(title=title or str(source_doc_id), text=text, source_doc_id=str(source_doc_id))]
        return []

    if isinstance(item, str):
        text = item.strip()
        if text:
            title = f"doc-{fallback_idx}"
            return [make_document(title=title, text=text, source_doc_id=title)]
    return []


def collapse_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip())
    if isinstance(value, dict):
        if "sentences" in value:
            return collapse_text(value["sentences"])
        return collapse_text(list(value.values()))
    return str(value).strip()


def make_document(title: str, text: str, source_doc_id: str | None = None) -> dict[str, Any]:
    source_doc_id = source_doc_id or title
    return {
        "id": source_doc_id,
        "title": title,
        "text": text,
        "source_doc_id": source_doc_id,
    }


def print_summary(
    input_path: Path,
    queries_output: Path,
    corpus_output: Path,
    total_records: int,
    subset: list[dict[str, Any]],
    shared_corpus: list[dict[str, Any]],
    seed: int,
    dataset_name: str,
    workload: str,
) -> None:
    question_types = Counter(str(record.get("type", record.get("question_type", "unknown"))) for record in subset)
    avg_context_docs = sum(len(iterate_documents(record)) for record in subset) / max(1, len(subset))

    print("Prepared shared-corpus benchmark dataset")
    print(f"Input file: {input_path}")
    print(f"Queries output: {queries_output}")
    print(f"Corpus output: {corpus_output}")
    print(f"Dataset name: {dataset_name}")
    print(f"Workload: {workload}")
    print(f"Original size: {total_records}")
    print(f"Subset size: {len(subset)}")
    print(f"Shared corpus docs: {len(shared_corpus)}")
    print(f"Seed: {seed}")
    print(f"Average docs per query: {avg_context_docs:.2f}")
    print(f"Question type distribution: {dict(question_types)}")


if __name__ == "__main__":
    main()
