from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline parser-stability check for HippoRAG OpenIE/NER raw responses."
    )
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL cache file.")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "ner", "triples"],
        help="Which payload to parse. 'auto' infers from file path and payload shape.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of candidate raw responses to test.",
    )
    parser.add_argument(
        "--failures-output",
        default="",
        help="Optional path to write failure cases as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    payload = load_payload(input_path)
    candidates = collect_candidate_records(payload)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    task = infer_task(args.task, input_path)
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for row in candidates:
        raw = row["raw_response"]
        try:
            if task == "ner":
                parsed = extract_ner_from_response(raw)
            elif task == "triples":
                parsed = extract_triples_from_response(raw)
            else:
                parsed = extract_auto(raw)
            success_rows.append(
                {
                    "record_id": row["record_id"],
                    "path": row["path"],
                    "parsed_len": len(parsed) if isinstance(parsed, list) else None,
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "record_id": row["record_id"],
                    "path": row["path"],
                    "error": str(exc),
                    "raw_response": raw[:1000],
                }
            )

    summary = {
        "input": str(input_path),
        "task": task,
        "tested_candidates": len(candidates),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "success_rate": (len(success_rows) / len(candidates)) if candidates else 0.0,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failure_rows:
        print("\nFirst failure:")
        print(json.dumps(failure_rows[0], indent=2, ensure_ascii=False))

    if args.failures_output:
        output_path = Path(args.failures_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "failures": failure_rows,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )


def load_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_task(requested: str, input_path: Path) -> str:
    if requested != "auto":
        return requested
    lowered = input_path.name.lower()
    if "ner" in lowered:
        return "ner"
    if "triple" in lowered or "openie" in lowered:
        return "triples"
    return "auto"


def collect_candidate_records(payload: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def visit(value: Any, path: str, record_id: str | None) -> None:
        if isinstance(value, dict):
            local_record_id = record_id
            for key in ("chunk_id", "id", "record_id"):
                if key in value and value[key] is not None:
                    local_record_id = str(value[key])
                    break

            for key in ("response", "raw_response", "content", "message", "text"):
                candidate = value.get(key)
                if isinstance(candidate, str) and ("{" in candidate or "[" in candidate):
                    rows.append(
                        {
                            "record_id": local_record_id or path,
                            "path": f"{path}.{key}" if path else key,
                            "raw_response": candidate,
                        }
                    )
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key), local_record_id)
            return

        if isinstance(value, list):
            for idx, item in enumerate(value):
                visit(item, f"{path}[{idx}]" if path else f"[{idx}]", record_id)

    visit(payload, "", None)

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["record_id"], row["raw_response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def parse_json_payload(real_response: str):
    text = real_response.strip()
    candidates: list[str] = []

    if text:
        candidates.append(text)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    end_positions = [pos for pos in (text.rfind("}"), text.rfind("]")) if pos != -1]
    if start_positions and end_positions:
        start_pos = min(start_positions)
        end_pos = max(end_positions)
        if start_pos < end_pos:
            snippet = text[start_pos : end_pos + 1].strip()
            if snippet:
                candidates.append(snippet)

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        parsed = try_parse_candidate(candidate)
        if parsed is not None:
            return parsed

    if text.count("{") < text.count("}") and text.rstrip().endswith("}"):
        trimmed = text.rstrip()
        while trimmed.count("{") < trimmed.count("}"):
            trimmed = trimmed[:-1].rstrip()
        parsed = try_parse_candidate(trimmed)
        if parsed is not None:
            return parsed

    if text.lstrip().startswith("["):
        trimmed = text.strip()
        if trimmed.endswith(","):
            trimmed = trimmed[:-1].rstrip()
        while trimmed and not trimmed.endswith("]"):
            last_comma = trimmed.rfind(",")
            if last_comma == -1:
                break
            trimmed = trimmed[:last_comma].rstrip()
            if trimmed.endswith(","):
                trimmed = trimmed[:-1].rstrip()
        if trimmed and not trimmed.endswith("]"):
            trimmed = trimmed + "]"
        parsed = try_parse_candidate(trimmed)
        if parsed is not None:
            return parsed

    raise ValueError(f"Could not parse JSON from response: {real_response[:500]!r}")


def try_parse_candidate(candidate: str) -> dict[str, Any] | list[Any] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            payload = parser(candidate)
        except Exception:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def extract_ner_from_response(real_response: str) -> list[Any]:
    payload = parse_json_payload(real_response)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        entities = payload.get("named_entities", [])
        if isinstance(entities, list):
            return entities
    raise ValueError(f"Unexpected NER payload: {payload!r}")


def extract_triples_from_response(real_response: str) -> list[Any]:
    payload = parse_json_payload(real_response)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        triples = payload.get("triples", [])
        if isinstance(triples, list):
            return triples
    raise ValueError(f"Unexpected triples payload: {payload!r}")


def extract_auto(real_response: str) -> list[Any]:
    payload = parse_json_payload(real_response)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("named_entities", "triples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unexpected payload: {payload!r}")


if __name__ == "__main__":
    main()
