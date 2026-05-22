"""Gap-conditioned query rewriting for the correction subset (label=1).

Pipeline position:
  [1st-pass dense + HippoRAG]  ->  routing_rows.jsonl + retrieval_results.jsonl
  [THIS SCRIPT]                 ->  rewritten_queries.jsonl   (one rewrite per label=1 query)
  [2nd-pass dense on rewrite]  ->  you re-run your existing colbert/dense script with the
                                    rewritten queries; merge into final ranking elsewhere.

Design (borrowed from IRCoT + Q-PRM, stripped down):
  * IRCoT  -> use already-retrieved context (dense top-k + graph top-k bridge candidates)
              as grounding for the next sub-query.  We do ONE rewrite step, not a full loop;
              empirically most of the benefit is in the first interleave.
  * Q-PRM  -> prompt-level atomic decomposition: ask the LLM to (a) classify the gap type
              (B = missing bridge, C = missing query-entity fact), (b) emit a single
              rewritten query aimed exactly at the missing hop.
  * Not borrowed: IRCoT's CoT answer generation (we want retrieval, not answers);
                  Q-PRM's MCTS-trained PRM (prompt-only is enough for the pilot).

What we write per query: rewritten_query (string), gap_type (A/B/C/unclear),
llm_rationale (for error analysis). You feed rewritten_query into your existing
dense retrieval pipeline.

Usage:
  export OPENAI_API_KEY=...
  python3 scripts/gap_conditioned_rewrite.py \\
      --retrieval-file results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \\
      --routing-file   results/study_hotpot_hipporag_colbert_500/routing_rows.jsonl \\
      --out            results/study_hotpot_hipporag_colbert_500/rewritten_queries.jsonl \\
      --base-url       https://<your-proxy>/v1 \\
      --model          gpt-4.1 \\
      --top-k          5

Notes:
  * Only label=1 rows are rewritten.  label=0 queries are passed through verbatim
    (rewritten_query = question) so downstream merging is simple.
  * Caching by (question, dense_titles, graph_titles, model) so retries / 2-round
    experiments don't re-charge the API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You rewrite multi-hop questions to target exactly the retrieval gap that a first-"
    "pass dense retriever missed. You will be given the question, the titles+snippets "
    "the dense retriever already returned, and the titles a graph retriever surfaced as "
    "bridge candidates. Your output is a JSON object with keys gap_type, rationale, "
    "rewritten_query. Be concise; do not answer the question."
)

USER_TEMPLATE = """Question:
{question}

Dense top-{top_k} (already retrieved):
{dense_block}

Graph bridge candidates (titles surfaced by a graph retriever but NOT already in dense):
{graph_block}

Task:
1. Decide the gap type:
   - "C" (query_entity_miss): the question mentions an entity whose facts are absent
     from the dense context. The entity itself is in the question; we need a chunk
     describing its relevant attribute.
   - "B" (bridge_miss): the question's answer requires a bridge entity that is not
     visible in the dense context but is plausibly among the graph candidates above.
   - "unclear": not enough evidence above to decide.
2. Write ONE rewritten_query focused on the missing hop:
   - If gap_type == "C": rewrite to emphasize the question's named entity plus the
     specific attribute the question asks about (e.g. "When did <entity> <event>?").
   - If gap_type == "B": pick the MOST plausible bridge candidate from the graph list
     and rewrite as a lookup against that candidate's likely relevant attribute
     (e.g. if the question is "Director of X is based in what city?" and graph lists
     "Jane Doe" as a candidate director, rewrite as "Where is Jane Doe based?").
   - If gap_type == "unclear": repeat the original question.

Respond with ONLY a JSON object. No prose outside the JSON. Schema:
{{"gap_type": "C"|"B"|"unclear", "rationale": "<=1 sentence", "rewritten_query": "<one sentence>"}}
"""


# ---------- helpers ----------

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def format_block(items: list[tuple[str, str]]) -> str:
    if not items:
        return "(none)"
    lines = []
    for i, (title, snippet) in enumerate(items, 1):
        snippet = (snippet or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        lines.append(f"{i}. {title}\n   {snippet}")
    return "\n".join(lines)


def format_titles(titles: list[str]) -> str:
    if not titles:
        return "(none)"
    return "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))


def build_cache_key(model: str, question: str, dense_titles: list[str],
                    graph_titles: list[str]) -> str:
    payload = json.dumps(
        {"m": model, "q": question, "d": dense_titles, "g": graph_titles},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rewrite (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    row = conn.execute("SELECT value FROM rewrite WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(conn: sqlite3.Connection, key: str, value: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rewrite (key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False)),
    )
    conn.commit()


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(text: str) -> dict:
    """GPT-4.1 is usually well-behaved, but strip stray prose just in case."""
    if not text:
        return {}
    m = JSON_BLOCK_RE.search(text)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-file", required=True, type=Path)
    p.add_argument("--routing-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"),
                   help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL when set.")
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="Dry run: process first N label=1 queries.")
    args = p.parse_args()

    cache_path = args.cache_path or args.out.with_suffix(args.out.suffix + ".sqlite")
    cache = init_cache(cache_path)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"Missing env var {args.api_key_env}")

    from openai import OpenAI
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    routing = load_jsonl(args.routing_file)
    retrieval = {r["id"]: r for r in load_jsonl(args.retrieval_file)}

    label1_rows = [r for r in routing if r.get("label") == 1]
    if args.limit:
        label1_rows = label1_rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_cache = n_api = n_parse_fail = 0
    gap_counts: dict[str, int] = {}

    with args.out.open("w", encoding="utf-8") as out_f:
        # Pass through label=0 rows verbatim first so downstream can join on id
        id_to_label1 = {r["id"] for r in label1_rows}
        for r in routing:
            qid = r["id"]
            if qid in id_to_label1:
                continue
            out_f.write(json.dumps({
                "id": qid,
                "question": r["question"],
                "rewritten_query": r["question"],
                "gap_type": "skip_label0",
                "rationale": "",
                "cache_hit": False,
            }, ensure_ascii=False) + "\n")

        for row in tqdm(label1_rows, desc="Rewriting label=1"):
            qid = row["id"]
            question = row["question"]
            retr = retrieval.get(qid, {}).get("retrieval", {})
            dense = (retr.get("dense") or [])[: args.top_k]
            graph = (retr.get("graph") or [])[: args.top_k]

            dense_titles = [p.get("title") or p.get("id") or "" for p in dense]
            dense_items = [(p.get("title") or p.get("id") or "", p.get("text") or "")
                           for p in dense]
            dense_title_set = {t for t in dense_titles if t}
            graph_bridge_titles = [
                p.get("title") or p.get("id") or "" for p in graph
                if (p.get("title") or p.get("id") or "") not in dense_title_set
            ][: args.top_k]

            key = build_cache_key(args.model, question, dense_titles, graph_bridge_titles)
            cached = cache_get(cache, key)
            if cached is not None:
                n_cache += 1
                payload = cached
                cache_hit = True
            else:
                user_msg = USER_TEMPLATE.format(
                    question=question,
                    top_k=args.top_k,
                    dense_block=format_block(dense_items),
                    graph_block=format_titles(graph_bridge_titles),
                )
                for attempt in range(args.max_retries + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=args.model,
                            temperature=args.temperature,
                            max_completion_tokens=args.max_tokens,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_msg},
                            ],
                            response_format={"type": "json_object"},
                        )
                        break
                    except Exception as e:
                        if attempt == args.max_retries:
                            raise
                        time.sleep(min(2 ** attempt, 30))
                raw = resp.choices[0].message.content or ""
                parsed = parse_llm_json(raw)
                if not parsed.get("rewritten_query"):
                    n_parse_fail += 1
                    parsed = {
                        "gap_type": "unclear",
                        "rationale": "parse_failed",
                        "rewritten_query": question,
                    }
                payload = {
                    "gap_type": parsed.get("gap_type") or "unclear",
                    "rationale": parsed.get("rationale") or "",
                    "rewritten_query": parsed.get("rewritten_query") or question,
                    "raw_response": raw,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                }
                cache_put(cache, key, payload)
                n_api += 1
                cache_hit = False

            gap_counts[payload["gap_type"]] = gap_counts.get(payload["gap_type"], 0) + 1
            out_f.write(json.dumps({
                "id": qid,
                "question": question,
                "rewritten_query": payload["rewritten_query"],
                "gap_type": payload["gap_type"],
                "rationale": payload["rationale"],
                "dense_titles": dense_titles,
                "graph_bridge_titles": graph_bridge_titles,
                "cache_hit": cache_hit,
            }, ensure_ascii=False) + "\n")

    print(
        f"done. label1={len(label1_rows)} api_calls={n_api} cache_hits={n_cache} "
        f"parse_fail={n_parse_fail} gap_dist={gap_counts}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
