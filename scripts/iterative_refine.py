"""Iterative retrieval refinement 


  1. skip A_router           -> we already have a probe-aware router upstream
                                 (routing_rows.jsonl label=1/0).
  2. skip A_decompose (round 1) -> round-1 evidence is seeded from the existing
                                 dense top-5 + graph top-5 in retrieval_results.jsonl
                                 (product of our first-pass dense + HippoRAG PPR).
                                 LLM only enters from round 2 onward.
  3. skip A_filter           -> ColBERT already produces a ranked list; no extra
                                 LLM-based filter agent.

What remains is the paper's core: SEA + AQR inside a T-round loop.

  Algorithm (per label=1 query):
    E_agg  <- dense_top5 UNION graph_top5  (seed; round 1 surrogate)
    Q_prev <- {original_question}
    for t in 2..T:
        (summary, is_sufficient)  <-  SEA(question, E_agg)   # gap checklist + audit
        if is_sufficient: break
        sub_q  <-  AQR(question, summary, Q_prev)             # one targeted sub-query
        if jaccard(sub_q, any q in Q_prev) > 0.85: break      # progress stall
        Q_prev <- Q_prev + {sub_q}
        D      <-  ColBERT(sub_q, top-k)
        E_agg  <-  merge(E_agg, D)   # dedup by chunk_id, keep MAX score (IRCoT pattern)
    return E_agg ranked by max-score

label=0 queries pass through untouched (router said no correction needed).

Output JSONL: one row per query.
  {id, question, iterations: [{sub_query, is_sufficient, gaps, confirmed,
                                retrieved_ids, round}], final_ranking: [...],
   n_llm_calls, stopped_reason, gap_type_hint}

Downstream: a small merge script (write separately) injects
  retrieval["iter_refine_T{T}_top{K}"] = final_ranking[:K]
into retrieval_results.jsonl so the existing QA eval works unchanged.

Usage:
  export OPENAI_API_KEY=...
  python3 scripts/iterative_refine.py \\
      --config         configs/study_hotpot_hipporag_colbert_500.yaml \\
      --retrieval-file results/study_hotpot_hipporag_colbert_500/retrieval_results.jsonl \\
      --routing-file   results/study_hotpot_hipporag_colbert_500/routing_rows.jsonl \\
      --out            results/study_hotpot_hipporag_colbert_500/iterative_refine.jsonl \\
      --max-iter 3 --per-round-k 10 --final-k 5
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import load_shared_corpus
from src.retrieval_dense_colbert import ColBERTRetriever


# =====================================================================
# Prompts
# =====================================================================

SEA_SYSTEM = (
    "You are a Structured Evidence Assessment agent. Given a question and the "
    "currently collected evidence passages, you decompose the question into a "
    "checklist of required findings, audit the evidence, and decide whether the "
    "evidence is sufficient to answer faithfully. You NEVER answer the question. "
    "You output only a JSON object."
)

SEA_USER_TEMPLATE = """Question:
{question}

Current evidence (titles + snippets):
{evidence_block}

Previously confirmed findings (from earlier rounds — TREAT AS ALREADY CONFIRMED;
do NOT list them again as gaps unless the new evidence explicitly contradicts them):
{prior_confirmed_block}

Task:
1. Decompose the question into up to 4 atomic required findings (short noun phrases).
   A "required finding" is a factual component the answer depends on.
2. For each finding, decide whether it is CONFIRMED or REMAINS as a gap. A finding is
   CONFIRMED if it is:
     (a) directly stated in the current evidence, OR
     (b) listed in the previously-confirmed findings above, OR
     (c) TRIVIALLY DERIVABLE from the confirmed atoms — e.g., a yes/no comparison of
         two already-confirmed nationalities; "who is older" when both birth dates are
         confirmed; a simple arithmetic or lookup over confirmed values.
   If the atomic components needed to answer are all confirmed, the higher-level
   comparison/classification is ALSO confirmed — do NOT list it as a separate gap.
3. Your confirmed_findings output MUST include every item from the previously-confirmed
   list above (restate them verbatim) PLUS any new findings confirmed this round.
4. Classify each remaining gap as either:
   - "B" (bridge_miss): a hop-1 bridge entity / linking fact is missing
   - "C" (query_entity_miss): a direct attribute of an entity named in the question
     is missing
   - "other": neither clearly B nor C
5. Set is_sufficient = "Yes" only if NO required findings remain as gaps.
6. Bias toward is_sufficient="Yes" when in doubt: retrieving more is costly, and
   flagging a derivable fact as a gap pulls in noisy chunks that degrade the reader.

Respond with ONLY a JSON object, schema:
{{
  "confirmed_findings": ["<finding>", ...],
  "remaining_gaps": [{{"finding": "<finding>", "gap_type": "B"|"C"|"other"}}, ...],
  "is_sufficient": "Yes"|"No"
}}
"""

AQR_SYSTEM = (
    "You are an Adaptive Query Refinement agent. Given a question, a list of "
    "already-confirmed findings, a list of remaining gaps (each with a gap_type), "
    "and the queries tried so far, you write ONE new retrieval query targeting "
    "the single most important remaining gap. The query must differ materially "
    "from all prior queries. You output only a JSON object."
)

AQR_USER_TEMPLATE = """Question:
{question}

Confirmed findings (do NOT re-search these):
{confirmed_block}

Remaining gaps (rank the most important one first):
{gaps_block}

Prior queries already tried (avoid paraphrases):
{prior_block}

Write ONE retrieval query aimed at the top remaining gap:
  - gap_type "C": emphasize the question's named entity + the missing attribute,
    e.g. "When did <entity> <event>?"
  - gap_type "B": use the most plausible bridge entity (inferable from question +
    confirmed findings) and look up its relevant attribute,
    e.g. "Where is <bridge_entity> based?"
  - gap_type "other": write a keyword-rich lookup for the missing finding.

Respond with ONLY a JSON object, schema:
{{"target_gap": "<short copy of the chosen gap>",
  "gap_type": "B"|"C"|"other",
  "new_query": "<one sentence>"}}
"""


# =====================================================================
# Helpers
# =====================================================================

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def parse_llm_json(text: str) -> dict:
    if not text:
        return {}
    m = JSON_BLOCK_RE.search(text)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def chunk_id(p: dict) -> str:
    return p.get("id") or p.get("source_doc_id") or ""


def format_evidence(items: list[dict], max_len: int = 240) -> str:
    if not items:
        return "(none)"
    lines = []
    for i, p in enumerate(items, 1):
        title = p.get("title") or p.get("id") or ""
        text = (p.get("text") or "").strip().replace("\n", " ")
        if len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        lines.append(f"{i}. {title}\n   {text}")
    return "\n".join(lines)


def format_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {s}" for s in items)


def format_gaps(gaps: list[dict]) -> str:
    if not gaps:
        return "(none)"
    return "\n".join(
        f"- [{g.get('gap_type', 'other')}] {g.get('finding', '')}" for g in gaps
    )


def jaccard(a: str, b: str) -> float:
    A = set(re.findall(r"\w+", (a or "").lower()))
    B = set(re.findall(r"\w+", (b or "").lower()))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# =====================================================================
# Cache (sqlite, same pattern as gap_conditioned_rewrite.py)
# =====================================================================

def init_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    row = conn.execute("SELECT value FROM llm WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(conn: sqlite3.Connection, key: str, value: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO llm (key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False)),
    )
    conn.commit()


def cache_key(agent: str, model: str, payload: dict) -> str:
    blob = json.dumps({"a": agent, "m": model, **payload},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# =====================================================================
# LLM wrapper
# =====================================================================

def llm_json(client, model: str, system: str, user: str,
             temperature: float, max_tokens: int,
             max_retries: int = 5) -> tuple[dict, str, dict]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            break
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raw = resp.choices[0].message.content or ""
    parsed = parse_llm_json(raw)
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    return parsed, raw, usage


# =====================================================================
# ColBERT retrieval (reuse existing index)
# =====================================================================

def build_retriever(config: dict, project_root: Path) -> ColBERTRetriever:
    shared_corpus_path = config.get("shared_corpus_path")
    if not shared_corpus_path:
        raise ValueError("config.shared_corpus_path is required")
    corpus = load_shared_corpus(str(resolve_path(project_root, str(shared_corpus_path))))
    colbert_root = str(resolve_path(project_root, config["colbert_root"]))
    return ColBERTRetriever(
        corpus=corpus,
        root=colbert_root,
        experiment_name=config["colbert_experiment_name"],
        index_name=config["colbert_index_name"],
        checkpoint=config.get("colbert_checkpoint", "colbert-ir/colbertv2.0"),
        nbits=config.get("colbert_nbits", 2),
        partitions=config.get("colbert_partitions"),
        doc_maxlen=config.get("colbert_doc_maxlen", 220),
        query_maxlen=config.get("colbert_query_maxlen", 64),
        kmeans_niters=config.get("colbert_kmeans_niters", 4),
        rebuild_index=False,
    )


def retrieve_as_dicts(retriever: ColBERTRetriever, query: str, k: int) -> list[dict]:
    try:
        hits = retriever.retrieve(query, k)
    except Exception as e:
        print(f"[warn] retrieve failed for query={query[:80]!r}: {e}", file=sys.stderr)
        return []
    out = []
    for rank, h in enumerate(hits):
        try:
            d = asdict(h)
        except TypeError:
            d = {"id": h.id, "title": getattr(h, "title", None),
                 "text": getattr(h, "text", None),
                 "score": float(getattr(h, "score", 0.0)),
                 "source_doc_id": getattr(h, "source_doc_id", None)}
        d["rank"] = rank
        out.append(d)
    return out


# =====================================================================
# Per-query iterative loop
# =====================================================================

def run_query(
    question: str,
    seed_dense: list[dict],
    seed_graph: list[dict],
    retriever: ColBERTRetriever,
    client,
    cache: sqlite3.Connection,
    model: str,
    temperature: float,
    max_tokens: int,
    max_iter: int,
    per_round_k: int,
    sea_evidence_k: int,
    jaccard_thresh: float,
) -> dict:
    # E_agg: chunk_id -> {passage dict, best_score, first_round}
    e_agg: dict[str, dict] = {}

    def merge(items: list[dict], round_idx: int) -> None:
        for p in items:
            cid = chunk_id(p)
            if not cid:
                continue
            score = float(p.get("score") or 0.0)
            cur = e_agg.get(cid)
            if cur is None:
                e_agg[cid] = {"passage": p, "score": score, "round": round_idx}
            elif score > cur["score"]:
                cur["passage"] = p
                cur["score"] = score

    # Round 1 seed: existing dense + graph union (FAIR-RAG's decompose+retrieve
    # is replaced by our already-materialised hybrid retrieval).
    merge(seed_dense, round_idx=0)
    merge(seed_graph, round_idx=0)

    prior_queries = [question]
    prior_confirmed: list[str] = []   # monotonic accumulation across rounds (see patch below)
    trace: list[dict] = []
    n_llm_calls = 0
    stopped = "max_iter"

    for t in range(1, max_iter + 1):
        # Current evidence view for SEA (top-N by score).
        evidence_sorted = sorted(e_agg.values(), key=lambda x: -x["score"])
        evidence_view = [x["passage"] for x in evidence_sorted[:sea_evidence_k]]
        ev_titles = [p.get("title") or p.get("id") or "" for p in evidence_view]

        # ---- SEA ----
        # Pass prior_confirmed into SEA so earlier-confirmed findings do not
        # reappear as gaps in later rounds (fixes drift observed in sample-10
        # where R2 confirmed "Olive Branch is a suburb of Memphis" but R3
        # re-listed "specific suburb" as a gap).
        sea_payload = {
            "q": question,
            "ev_titles": ev_titles,
            "prior_confirmed": prior_confirmed,
            "round": t,
            "prompt_version": "sea_v2",   # tighter prompt: derivable-facts-not-gaps
        }
        sea_key = cache_key("sea", model, sea_payload)
        sea_cached = cache_get(cache, sea_key)
        if sea_cached is not None:
            sea_parsed = sea_cached["parsed"]
        else:
            sea_user = SEA_USER_TEMPLATE.format(
                question=question,
                evidence_block=format_evidence(evidence_view),
                prior_confirmed_block=format_list(prior_confirmed),
            )
            sea_parsed, sea_raw, sea_usage = llm_json(
                client, model, SEA_SYSTEM, sea_user, temperature, max_tokens
            )
            cache_put(cache, sea_key, {
                "parsed": sea_parsed, "raw": sea_raw, "usage": sea_usage,
            })
            n_llm_calls += 1

        confirmed = sea_parsed.get("confirmed_findings") or []
        gaps = sea_parsed.get("remaining_gaps") or []
        is_sufficient = (sea_parsed.get("is_sufficient") or "").strip().lower() == "yes"

        # Accumulate confirmed findings monotonically for the next round's SEA.
        # Dedup by lowercased string to avoid near-duplicate inflation.
        seen_lc = {s.lower() for s in prior_confirmed}
        for c in confirmed:
            if isinstance(c, str) and c.lower() not in seen_lc:
                prior_confirmed.append(c)
                seen_lc.add(c.lower())

        round_trace = {
            "round": t,
            "confirmed_findings": confirmed,
            "remaining_gaps": gaps,
            "is_sufficient": is_sufficient,
        }

        if is_sufficient or not gaps:
            trace.append(round_trace)
            stopped = "sufficient"
            break

        # ---- AQR ----
        aqr_payload = {
            "q": question,
            "confirmed": confirmed,
            "gaps": gaps,
            "prior": prior_queries,
            "round": t,
            "prompt_version": "aqr_v1",
        }
        aqr_key = cache_key("aqr", model, aqr_payload)
        aqr_cached = cache_get(cache, aqr_key)
        if aqr_cached is not None:
            aqr_parsed = aqr_cached["parsed"]
        else:
            aqr_user = AQR_USER_TEMPLATE.format(
                question=question,
                confirmed_block=format_list(confirmed),
                gaps_block=format_gaps(gaps),
                prior_block=format_list(prior_queries),
            )
            aqr_parsed, aqr_raw, aqr_usage = llm_json(
                client, model, AQR_SYSTEM, aqr_user, temperature, max_tokens
            )
            cache_put(cache, aqr_key, {
                "parsed": aqr_parsed, "raw": aqr_raw, "usage": aqr_usage,
            })
            n_llm_calls += 1

        new_q = (aqr_parsed.get("new_query") or "").strip()
        gap_type = aqr_parsed.get("gap_type") or "other"

        if not new_q:
            round_trace["sub_query"] = ""
            round_trace["stop_reason"] = "aqr_empty"
            trace.append(round_trace)
            stopped = "aqr_empty"
            break

        max_prior_jac = max(jaccard(new_q, q) for q in prior_queries)
        round_trace["sub_query"] = new_q
        round_trace["sub_query_gap_type"] = gap_type
        round_trace["jaccard_max_prior"] = round(max_prior_jac, 3)

        if max_prior_jac > jaccard_thresh:
            round_trace["stop_reason"] = "jaccard_stall"
            trace.append(round_trace)
            stopped = "jaccard_stall"
            break

        prior_queries.append(new_q)

        # ---- Retrieve + merge ----
        hits = retrieve_as_dicts(retriever, new_q, per_round_k)
        merge(hits, round_idx=t)
        round_trace["retrieved_ids"] = [chunk_id(p) for p in hits]
        trace.append(round_trace)

    # Final ranking
    final_sorted = sorted(e_agg.values(), key=lambda x: -x["score"])
    final_passages = [x["passage"] for x in final_sorted]

    return {
        "iterations": trace,
        "prior_queries": prior_queries,
        "final_ranking": final_passages,
        "n_llm_calls": n_llm_calls,
        "stopped_reason": stopped,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--retrieval-file", required=True, type=Path)
    ap.add_argument("--routing-file", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-path", type=Path, default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=3,
                    help="Max LLM-driven rounds AFTER the seed round (FAIR-RAG Algorithm 1 T=3).")
    ap.add_argument("--per-round-k", type=int, default=10,
                    help="ColBERT top-k per AQR sub-query.")
    ap.add_argument("--sea-evidence-k", type=int, default=10,
                    help="How many top passages from E_agg to show the SEA agent.")
    ap.add_argument("--seed-dense-k", type=int, default=5)
    ap.add_argument("--seed-graph-k", type=int, default=5)
    ap.add_argument("--jaccard-thresh", type=float, default=0.85,
                    help="Stop if a new sub-query has Jaccard > this against any prior query.")
    ap.add_argument("--final-k", type=int, default=20,
                    help="Keep top-K passages in final_ranking per output row.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Dry run: process first N label=1 queries.")
    ap.add_argument("--only-label1", action="store_true", default=True,
                    help="Only iterate on label=1; label=0 rows pass through with seed only.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    with args.config.open() as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"Missing env var {args.api_key_env}")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    cache_path = args.cache_path or args.out.with_suffix(args.out.suffix + ".sqlite")
    cache = init_cache(cache_path)

    routing = load_jsonl(args.routing_file)
    retrieval = {r["id"]: r for r in load_jsonl(args.retrieval_file)}

    # Lazy retriever: only build ColBERT if we actually have label=1 work to do
    # (avoids a multi-minute index load on a cache-fully-hit dev re-run).
    retriever: Optional[ColBERTRetriever] = None
    def get_retriever() -> ColBERTRetriever:
        nonlocal retriever
        if retriever is None:
            retriever = build_retriever(config, project_root)
        return retriever

    label1 = [r for r in routing if r.get("label") == 1]
    if args.limit:
        label1 = label1[: args.limit]
    label1_ids = {r["id"] for r in label1}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_llm_calls = 0
    stop_counts: dict[str, int] = {}

    with args.out.open("w", encoding="utf-8") as out_f:
        # Pass-through label=0 (seed only, no iterations) so downstream can join on id.
        for r in routing:
            qid = r["id"]
            if qid in label1_ids:
                continue
            retr = retrieval.get(qid, {}).get("retrieval", {})
            seed = (retr.get("dense") or [])[: args.seed_dense_k]
            out_f.write(json.dumps({
                "id": qid,
                "question": r["question"],
                "label": 0,
                "iterations": [],
                "prior_queries": [r["question"]],
                "final_ranking": seed[: args.final_k],
                "n_llm_calls": 0,
                "stopped_reason": "skip_label0",
            }, ensure_ascii=False) + "\n")

        for row in tqdm(label1, desc="Iter-refine label=1"):
            qid = row["id"]
            question = row["question"]
            retr = retrieval.get(qid, {}).get("retrieval", {})
            seed_dense = (retr.get("dense") or [])[: args.seed_dense_k]
            seed_graph = (retr.get("graph") or [])[: args.seed_graph_k]

            # Ensure retriever is built before first real query.
            _ = get_retriever()

            result = run_query(
                question=question,
                seed_dense=seed_dense,
                seed_graph=seed_graph,
                retriever=retriever,
                client=client,
                cache=cache,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_iter=args.max_iter,
                per_round_k=args.per_round_k,
                sea_evidence_k=args.sea_evidence_k,
                jaccard_thresh=args.jaccard_thresh,
            )

            total_llm_calls += result["n_llm_calls"]
            stop_counts[result["stopped_reason"]] = stop_counts.get(
                result["stopped_reason"], 0) + 1

            out_f.write(json.dumps({
                "id": qid,
                "question": question,
                "label": 1,
                "iterations": result["iterations"],
                "prior_queries": result["prior_queries"],
                "final_ranking": result["final_ranking"][: args.final_k],
                "n_llm_calls": result["n_llm_calls"],
                "stopped_reason": result["stopped_reason"],
            }, ensure_ascii=False) + "\n")

    print(
        f"done. label1={len(label1)} total_llm_calls={total_llm_calls} "
        f"avg_llm_per_q={total_llm_calls / max(1, len(label1)):.2f} "
        f"stop_reasons={stop_counts}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
