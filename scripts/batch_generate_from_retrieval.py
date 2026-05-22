from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided passages. "
    "Return only the final short answer, with no explanation. "
    "If the passages do not contain enough information, say unknown."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer generation from a saved retrieval_results.jsonl file."
    )
    parser.add_argument("--input", required=True, help="Path to retrieval_results.jsonl.")
    parser.add_argument("--output", required=True, help="Path to generation output JSONL.")
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional SQLite cache path for LLM input-output pairs. Defaults to <output>.sqlite",
    )
    parser.add_argument("--model", required=True, help="Generator model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable name for API key.")
    parser.add_argument(
        "--methods",
        default="dense,graph,fusion",
        help="Comma-separated retrieval methods to generate from, or 'auto' to use the input rows.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved passages per method to include.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional LLM sampling seed for OpenAI-compatible endpoints that support it.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="OpenAI client timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="OpenAI client retry count.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens for each generation.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt text.")
    parser.add_argument(
        "--prompt-template-file",
        default=None,
        help=(
            "Optional text file for the user prompt template. Available placeholders: "
            "{question}, {method}, {top_k}, {context}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache_path) if args.cache_path else output_path.with_suffix(output_path.suffix + ".sqlite")
    rows = load_jsonl(input_path)
    methods = (
        infer_methods(rows)
        if args.methods == "auto"
        else [method.strip() for method in args.methods.split(",") if method.strip()]
    )
    prompt_template = load_prompt_template(args.prompt_template_file)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key environment variable: {args.api_key_env}")

    from openai import OpenAI

    client_kwargs = {
        "api_key": api_key,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_cache(cache_path)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            generations: dict[str, Any] = {}
            retrieval = row.get("retrieval", {})
            for method in methods:
                passages = list(retrieval.get(method, []))[: args.top_k]
                if not passages:
                    generations[method] = {
                        "error": f"No retrieval passages available for method '{method}'.",
                    }
                    continue

                effective_top_k = len(passages)
                prompt = prompt_template.format(
                    question=row["question"],
                    method=method,
                    top_k=effective_top_k,
                    context=format_context(passages),
                )
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                cached = load_cached_generation(
                    cache_path=cache_path,
                    base_url=args.base_url,
                    model=args.model,
                    temperature=args.temperature,
                    seed=args.seed,
                    max_completion_tokens=args.max_completion_tokens,
                    messages=messages,
                )
                if cached is None:
                    try:
                        request_kwargs = {
                            "model": args.model,
                            "temperature": args.temperature,
                            "max_completion_tokens": args.max_completion_tokens,
                            "messages": messages,
                        }
                        if args.seed is not None:
                            request_kwargs["seed"] = args.seed
                        response = client.chat.completions.create(
                            **request_kwargs,
                        )
                        cached = {
                            "answer": response.choices[0].message.content,
                            "finish_reason": response.choices[0].finish_reason,
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                            "completion_tokens": response.usage.completion_tokens if response.usage else None,
                            "cache_hit": False,
                        }
                        save_cached_generation(
                            cache_path=cache_path,
                            base_url=args.base_url,
                            model=args.model,
                            temperature=args.temperature,
                            seed=args.seed,
                            max_completion_tokens=args.max_completion_tokens,
                            messages=messages,
                            payload=cached,
                        )
                    except Exception as exc:
                        cached = error_generation_payload(exc)
                        print(
                            (
                                "[generation-error] "
                                f"id={row.get('id')} method={method} "
                                f"type={cached['error_type']} error={cached['error']}"
                            ),
                            flush=True,
                        )
                else:
                    cached = dict(cached)
                    cached["cache_hit"] = True

                generations[method] = {
                    "answer": cached.get("answer", ""),
                    "finish_reason": cached.get("finish_reason"),
                    "prompt_tokens": cached.get("prompt_tokens"),
                    "completion_tokens": cached.get("completion_tokens"),
                    "cache_hit": cached.get("cache_hit", False),
                    "error": cached.get("error"),
                    "error_type": cached.get("error_type"),
                    "top_k": effective_top_k,
                    "passage_ids": [passage.get("id") for passage in passages],
                }

            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "gold_answer": row.get("gold_answer", row.get("answer")),
                        "gold_answers": row.get("gold_answers", [row.get("gold_answer", row.get("answer"))]),
                        "dataset_name": row.get("dataset_name"),
                        "workload": row.get("workload"),
                        "question_type": row.get("question_type"),
                        "main_table_decisions": row.get("main_table_decisions"),
                        "main_table_methods": row.get("main_table_methods"),
                        "methods": generations,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def error_generation_payload(exc: Exception) -> dict[str, Any]:
    return {
        "answer": "",
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cache_hit": False,
        "error": truncate_error(str(exc)),
        "error_type": type(exc).__name__,
    }


def truncate_error(message: str, limit: int = 1000) -> str:
    message = " ".join(str(message).split())
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def infer_methods(rows: list[dict[str, Any]]) -> list[str]:
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
    for row in rows:
        row_methods = row.get("main_table_methods")
        if isinstance(row_methods, list) and row_methods:
            return [str(method) for method in row_methods if str(method).strip()]

    methods: set[str] = set()
    for row in rows:
        retrieval = row.get("retrieval", {})
        if isinstance(retrieval, dict):
            methods.update(str(method) for method in retrieval)

    ordered = [method for method in preferred_order if method in methods]
    ordered.extend(sorted(methods - set(ordered)))
    return ordered


def initialize_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def compute_cache_key(
    *,
    base_url: str | None,
    model: str,
    temperature: float,
    seed: int | None,
    max_completion_tokens: int,
    messages: list[dict[str, str]],
) -> str:
    payload = {
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_completion_tokens": max_completion_tokens,
        "messages": messages,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_generation(
    *,
    cache_path: Path,
    base_url: str | None,
    model: str,
    temperature: float,
    seed: int | None,
    max_completion_tokens: int,
    messages: list[dict[str, str]],
) -> dict[str, Any] | None:
    cache_key = compute_cache_key(
        base_url=base_url,
        model=model,
        temperature=temperature,
        seed=seed,
        max_completion_tokens=max_completion_tokens,
        messages=messages,
    )
    with sqlite3.connect(cache_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT response_json FROM generation_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def save_cached_generation(
    *,
    cache_path: Path,
    base_url: str | None,
    model: str,
    temperature: float,
    seed: int | None,
    max_completion_tokens: int,
    messages: list[dict[str, str]],
    payload: dict[str, Any],
) -> None:
    cache_key = compute_cache_key(
        base_url=base_url,
        model=model,
        temperature=temperature,
        seed=seed,
        max_completion_tokens=max_completion_tokens,
        messages=messages,
    )
    with sqlite3.connect(cache_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO generation_cache (cache_key, response_json) VALUES (?, ?)",
            (cache_key, json.dumps(payload, ensure_ascii=False)),
        )
        connection.commit()


def load_prompt_template(path: str | None) -> str:
    if path is None:
        return (
            "Question: {question}\n"
            "Use the following top-{top_k} passages.\n\n"
            "{context}\n\n"
            "Answer with only the final short answer:"
        )
    return Path(path).read_text(encoding="utf-8")


def format_context(passages: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for index, passage in enumerate(passages, start=1):
        title = passage.get("title") or "(untitled)"
        source_doc_id = passage.get("source_doc_id") or ""
        prefix = f"[{index}] title={title}"
        if source_doc_id:
            prefix += f" source_doc_id={source_doc_id}"
        formatted.append(f"{prefix}\n{passage.get('text', '')}")
    return "\n\n".join(formatted)


if __name__ == "__main__":
    main()
