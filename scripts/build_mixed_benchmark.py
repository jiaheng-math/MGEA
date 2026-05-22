from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed benchmark by concatenating multiple prepared query files and merging "
            "their shared corpora."
        )
    )
    parser.add_argument("--queries", nargs="+", required=True, help="Prepared query JSON files.")
    parser.add_argument("--corpora", nargs="+", required=True, help="Prepared shared-corpus JSON files.")
    parser.add_argument("--queries-output", required=True, help="Output JSON file for mixed queries.")
    parser.add_argument("--corpus-output", required=True, help="Output JSON file for mixed shared corpus.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.queries) != len(args.corpora):
        raise ValueError("--queries and --corpora must have the same number of files.")

    mixed_queries: list[dict[str, Any]] = []
    mixed_corpus: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    merged_docs: dict[str, dict[str, Any]] = {}

    for query_path_str, corpus_path_str in zip(args.queries, args.corpora):
        query_path = Path(query_path_str)
        corpus_path = Path(corpus_path_str)
        query_records = load_json_list(query_path)
        corpus_records = load_json_list(corpus_path)

        for record in query_records:
            record = dict(record)
            dataset_name = str(record.get("dataset_name", infer_dataset_name(query_path))).strip().lower()
            workload = str(record.get("workload", infer_workload(dataset_name))).strip().lower()
            sample_id = str(record.get("id", f"{dataset_name}-{len(mixed_queries)}"))
            mixed_id = f"{dataset_name}::{sample_id}"
            if mixed_id in seen_query_ids:
                continue
            seen_query_ids.add(mixed_id)
            record["id"] = mixed_id
            record["dataset_name"] = dataset_name
            record["workload"] = workload
            mixed_queries.append(record)

        for doc in corpus_records:
            source_doc_id = str(doc.get("source_doc_id", doc.get("id", ""))).strip()
            title = str(doc.get("title", source_doc_id)).strip()
            if not source_doc_id:
                source_doc_id = title
            canonical_doc_id = f"{title}::{source_doc_id}" if title and source_doc_id != title else source_doc_id
            normalized_doc = {
                "id": canonical_doc_id,
                "title": title or canonical_doc_id,
                "text": str(doc.get("text", "")).strip(),
                "source_doc_id": canonical_doc_id,
            }
            existing = merged_docs.get(canonical_doc_id)
            if existing is None:
                merged_docs[canonical_doc_id] = normalized_doc
                continue
            if normalized_doc["text"] and normalized_doc["text"] not in existing["text"]:
                existing["text"] = " ".join(part for part in [existing["text"], normalized_doc["text"]] if part).strip()
            if not existing.get("title") and normalized_doc.get("title"):
                existing["title"] = normalized_doc["title"]

    mixed_corpus = list(merged_docs.values())

    queries_output = Path(args.queries_output)
    corpus_output = Path(args.corpus_output)
    queries_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    with queries_output.open("w", encoding="utf-8") as handle:
        json.dump(mixed_queries, handle, indent=2, ensure_ascii=False)
    with corpus_output.open("w", encoding="utf-8") as handle:
        json.dump(mixed_corpus, handle, indent=2, ensure_ascii=False)

    print("Built mixed benchmark")
    print(f"Queries output: {queries_output}")
    print(f"Corpus output: {corpus_output}")
    print(f"Total mixed queries: {len(mixed_queries)}")
    print(f"Total mixed corpus docs: {len(mixed_corpus)}")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a top-level JSON list: {path}")
    return payload


def infer_dataset_name(path: Path) -> str:
    stem = path.stem.lower()
    if "hotpot" in stem:
        return "hotpotqa"
    if "2wiki" in stem:
        return "2wikimultihopqa"
    if "popqa" in stem:
        return "popqa"
    if "nq" in stem:
        return "nq"
    return "unknown"


def infer_workload(dataset_name: str) -> str:
    if dataset_name in {"hotpotqa", "2wikimultihopqa"}:
        return "multi-hop"
    if dataset_name in {"popqa", "nq"}:
        return "single-hop"
    return "unknown"


if __name__ == "__main__":
    main()
