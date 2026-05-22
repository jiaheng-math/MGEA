from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/export Natural Questions dev split from Hugging Face into a compact JSONL.GZ "
            "file consumed by prepare_nq_shared_corpus.py."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default="google-research-datasets/natural_questions",
        help="Hugging Face dataset id.",
    )
    parser.add_argument("--config", default="dev", help="Hugging Face dataset config.")
    parser.add_argument("--split", default="validation", help="Hugging Face dataset split.")
    parser.add_argument("--output", default="data/nq_dev.jsonl.gz", help="Output JSONL.GZ path.")
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint mirror. Use empty string to keep the current environment.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap for quick local testing. Omit for the full dev split.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face streaming and materialize the split locally first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `python -m pip install datasets` "
            "or rerun the project environment setup."
        ) from exc

    dataset = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        streaming=not args.no_streaming,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(normalize_hf_row(row), ensure_ascii=False) + "\n")
            written += 1
            if args.max_records is not None and written >= args.max_records:
                break

    print("Downloaded/exported Natural Questions")
    print(f"Dataset: {args.dataset_name}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '')}")
    print(f"Output: {output_path}")
    print(f"Rows written: {written}")


def normalize_hf_row(row: dict[str, Any]) -> dict[str, Any]:
    document = row.get("document") if isinstance(row.get("document"), dict) else {}
    question = row.get("question") if isinstance(row.get("question"), dict) else {}
    return {
        "example_id": str(row.get("id", row.get("example_id", ""))),
        "question_text": str(question.get("text", row.get("question_text", row.get("question", "")))).strip(),
        "document_title": str(document.get("title", row.get("document_title", row.get("title", "")))).strip(),
        "document_url": str(document.get("url", row.get("document_url", ""))).strip(),
        "document_tokens": normalize_tokens(document.get("tokens", row.get("document_tokens", []))),
        "long_answer_candidates": normalize_candidates(row.get("long_answer_candidates", [])),
        "annotations": normalize_annotations(row.get("annotations", [])),
    }


def normalize_tokens(tokens: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in iter_sequence(tokens):
        if isinstance(item, dict):
            output.append(
                {
                    "token": str(item.get("token", "")),
                    "html_token": bool(item.get("is_html", item.get("html_token", False))),
                    "start_byte": int(item.get("start_byte", -1)),
                    "end_byte": int(item.get("end_byte", -1)),
                }
            )
        else:
            output.append(
                {
                    "token": str(item),
                    "html_token": False,
                    "start_byte": -1,
                    "end_byte": -1,
                }
            )
    return output


def normalize_candidates(candidates: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in iter_sequence(candidates):
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "start_token": int(item.get("start_token", -1)),
                "end_token": int(item.get("end_token", -1)),
                "start_byte": int(item.get("start_byte", -1)),
                "end_byte": int(item.get("end_byte", -1)),
                "top_level": bool(item.get("top_level", False)),
            }
        )
    return output


def normalize_annotations(annotations: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in iter_sequence(annotations):
        if not isinstance(item, dict):
            continue
        long_answer = item.get("long_answer")
        if not isinstance(long_answer, dict):
            long_answer = {}
        output.append(
            {
                "id": str(item.get("id", "")),
                "long_answer": {
                    "start_token": int(long_answer.get("start_token", -1)),
                    "end_token": int(long_answer.get("end_token", -1)),
                    "start_byte": int(long_answer.get("start_byte", -1)),
                    "end_byte": int(long_answer.get("end_byte", -1)),
                    "candidate_index": int(long_answer.get("candidate_index", -1)),
                },
                "short_answers": normalize_short_answers(item.get("short_answers", [])),
                "yes_no_answer": normalize_yes_no_answer(item.get("yes_no_answer", "NONE")),
            }
        )
    return output


def normalize_short_answers(short_answers: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in iter_sequence(short_answers):
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "start_token": int(item.get("start_token", -1)),
                "end_token": int(item.get("end_token", -1)),
                "start_byte": int(item.get("start_byte", -1)),
                "end_byte": int(item.get("end_byte", -1)),
                "text": str(item.get("text", "")),
            }
        )
    return output


def normalize_yes_no_answer(value: Any) -> str:
    if isinstance(value, int):
        return {0: "NO", 1: "YES", -1: "NONE"}.get(value, "NONE")
    text = str(value).strip()
    return text if text else "NONE"


def iter_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return value
    if isinstance(value, dict) and value and all(isinstance(item, list) for item in value.values()):
        keys = list(value)
        length = max(len(value[key]) for key in keys)
        return [
            {
                key: value[key][index] if index < len(value[key]) else None
                for key in keys
            }
            for index in range(length)
        ]
    return []


if __name__ == "__main__":
    main()
