from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated QA answers with exact match and token-level F1."
    )
    parser.add_argument("--input", required=True, help="Path to generations.jsonl.")
    parser.add_argument("--output", required=True, help="Path to write QA metrics JSON.")
    parser.add_argument(
        "--methods",
        default="auto",
        help="Comma-separated methods to evaluate, or 'auto' to use all methods present.",
    )
    parser.add_argument(
        "--per-sample-output",
        default=None,
        help="Optional JSONL path for per-sample EM/F1 records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    methods = infer_methods(rows) if args.methods == "auto" else parse_methods(args.methods)
    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    per_sample_rows: list[dict[str, Any]] = []

    for row in rows:
        gold_answers = extract_gold_answers(row)
        gold_answer = gold_answers[0] if gold_answers else ""
        method_payloads = row.get("methods", {})
        sample_payload: dict[str, Any] = {
            "id": row.get("id"),
            "question": row.get("question"),
            "dataset_name": row.get("dataset_name"),
            "workload": row.get("workload"),
            "question_type": row.get("question_type"),
            "gold_answer": gold_answer,
            "gold_answers": gold_answers,
            "methods": {},
        }

        for method in methods:
            payload = method_payloads.get(method, {})
            prediction = extract_prediction(payload.get("answer", ""))
            scores = score_prediction_against_gold_answers(prediction=prediction, gold_answers=gold_answers)
            record = {
                "id": row.get("id"),
                "prediction": prediction,
                "gold_answer": gold_answer,
                "gold_answers": gold_answers,
                "matched_gold_answer": scores["matched_gold_answer"],
                "exact_match": scores["exact_match"],
                "f1": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "has_error": bool(payload.get("error")),
                "finish_reason": payload.get("finish_reason"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
                "cache_hit": payload.get("cache_hit"),
                "top_k": payload.get("top_k"),
                "passage_ids": payload.get("passage_ids", []),
            }
            per_method[method].append(record)
            sample_payload["methods"][method] = record

        per_sample_rows.append(sample_payload)

    metrics = {
        "input": args.input,
        "num_samples": len(rows),
        "methods": {
            method: summarize_records(records)
            for method, records in per_method.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.per_sample_output:
        write_jsonl(Path(args.per_sample_output), per_sample_rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_methods(rows: list[dict[str, Any]]) -> list[str]:
    methods: set[str] = set()
    for row in rows:
        method_payloads = row.get("methods", {})
        if isinstance(method_payloads, dict):
            methods.update(str(method) for method in method_payloads)
    preferred_order = [
        "dense_only",
        "graph_only",
        "dense_graph_rrf",
        "query_only_router",
        "probe_only_router",
        "ours_combined_router",
        "random_router",
        "dense",
        "graph",
        "fusion",
    ]
    ordered = [method for method in preferred_order if method in methods]
    ordered.extend(sorted(methods - set(ordered)))
    return ordered


def parse_methods(value: str) -> list[str]:
    return [method.strip() for method in value.split(",") if method.strip()]


def extract_prediction(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = text.strip("` \n\t")
    if not text:
        return ""

    # Keep EM/F1 robust when the model ignores the short-answer instruction.
    answer_markers = [
        r"(?im)^answer\s*:\s*(.+)$",
        r"(?im)^final answer\s*:\s*(.+)$",
        r"(?im)^the answer is\s+(.+)$",
    ]
    for pattern in answer_markers:
        match = re.search(pattern, text)
        if match:
            text = match.group(1).strip()
            break

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    return first_line.strip().strip('"').strip("'").strip()


def extract_gold_answers(row: dict[str, Any]) -> list[str]:
    values = row.get("gold_answers")
    if isinstance(values, list):
        answers = [str(value).strip() for value in values if str(value).strip()]
    else:
        answers = []
    fallback = str(row.get("gold_answer", "")).strip()
    if fallback:
        answers.append(fallback)
    return dedupe(answers)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def score_prediction_against_gold_answers(prediction: str, gold_answers: list[str]) -> dict[str, Any]:
    if not gold_answers:
        scores = score_prediction(prediction=prediction, gold_answer="")
        scores["matched_gold_answer"] = ""
        return scores

    scored = [
        (gold_answer, score_prediction(prediction=prediction, gold_answer=gold_answer))
        for gold_answer in gold_answers
    ]
    matched_gold_answer, best_scores = max(
        scored,
        key=lambda item: (item[1]["exact_match"], item[1]["f1"], item[1]["recall"], item[1]["precision"]),
    )
    payload: dict[str, Any] = dict(best_scores)
    payload["matched_gold_answer"] = matched_gold_answer
    return payload


def score_prediction(prediction: str, gold_answer: str) -> dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    exact_match = float(normalized_prediction == normalized_gold)
    precision, recall, f1 = token_f1(normalized_prediction, normalized_gold)
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def normalize_answer(text: str) -> str:
    def lower(value: str) -> str:
        return value.lower()

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(text)))))


def token_f1(normalized_prediction: str, normalized_gold: str) -> tuple[float, float, float]:
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not prediction_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "num_samples": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    prompt_tokens = [record["prompt_tokens"] for record in records if record.get("prompt_tokens") is not None]
    completion_tokens = [
        record["completion_tokens"] for record in records if record.get("completion_tokens") is not None
    ]
    cache_hits = [record["cache_hit"] for record in records if record.get("cache_hit") is not None]

    return {
        "num_samples": count,
        "exact_match": mean(record["exact_match"] for record in records),
        "f1": mean(record["f1"] for record in records),
        "precision": mean(record["precision"] for record in records),
        "recall": mean(record["recall"] for record in records),
        "error_rate": mean(float(record["has_error"]) for record in records),
        "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": mean(completion_tokens) if completion_tokens else None,
        "cache_hit_rate": mean(float(value) for value in cache_hits) if cache_hits else None,
    }


def mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
