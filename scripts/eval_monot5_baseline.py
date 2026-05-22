"""MonoT5 (slot allocator) and graph_top7 baselines at AvgK=7.

Two **independent** baselines for the marginal slot allocation table:

  * ``graph_top7``  --  the graph retriever's own top-7 from G20(q).
    A trivial fixed-budget baseline that sits between ``graph_top5`` and
    ``graph_top8`` in the same table. The reviewer-facing significance
    test pairs every other method against this one.

  * ``monot5_slot_graph_5_to_20_cap5_avg7`` --  apples-to-apples MonoT5
    slot allocator. Base context is fixed to G5(q); MonoT5 (a self-contained
    relevance cross-encoder) scores tail candidates in graph ranks 6..20;
    the same global slot budget as ``score_slot_*`` and ``slot_text_rerank_*``
    is applied (``total_slots = round(N * (avg_k - base_k))``, per-query
    cap 5, ties broken by qid/rank/pid). This means MonoT5 picks ~2 tail
    slots per query on average, **never** displacing a G5 passage. Same
    setup as the slot model, only the scorer changes.

The narrative this enables in the slot QA table is a clean spectrum:

    graph-score-slot  (graph retriever's own scores -- no semantic reranker)
        |
    monot5-slot       (self-contained query-passage relevance reranker)
        |
    text-only slot    (supervised self-contained reranker over text features)
        |
    MGEA              (conditional marginal evidence gain given G5(q))

Workflow (single GPT-4.1 reader pass, 1000 examples, one table):

    # 1. Score MonoT5 over the G20 candidate pool and merge the two new
    #    methods into the existing slot generation input.
    python scripts/eval_monot5_baseline.py \\
      --datasets hotpot 2wiki \\
      --deep-retrieval-files \\
        results/study_hotpot_hipporag_colbert_500/retrieval_results_deep20.jsonl \\
        results/study_2wiki_hipporag_colbert_500/retrieval_results_deep20.jsonl \\
      --merge-into results/marginal_slot_allocation_ablation_generation_input.jsonl \\
      --merged-output results/monot5_baseline/merged_generation_input.jsonl \\
      --score-cache results/monot5_baseline/scores_cache.jsonl \\
      --output-dir results/monot5_baseline

    # 2. One reader sweep covers graph_top5/7/8/10, slot variants, MonoT5,
    #    MGEA -- everything in the same table.
    python scripts/run_slot_reader_sweep.py \\
      --generation-input results/monot5_baseline/merged_generation_input.jsonl \\
      --models gpt-4.1 \\
      --output-dir results/monot5_baseline/reader_sweep \\
      --baseline graph_top7 \\
      --compare graph_top5,graph_top8,graph_top10,monot5_slot_graph_5_to_20_cap5_avg7,slot_graph_5_to_20_cap5_avg7,score_slot_graph_5_to_20_cap5_avg7,random_slot_graph_5_to_20_cap5_avg7,slot_text_rerank_graph_5_to_20_cap5_avg7

    # 3. Reviewer-facing McNemar (EM) + 5k bootstrap (F1) vs graph_top7.
    python scripts/sig_test_qa.py \\
      --input results/monot5_baseline/reader_sweep/qa_per_sample_gpt-4.1.jsonl \\
      --baseline graph_top7 \\
      --compare graph_top5,graph_top8,graph_top10,monot5_slot_graph_5_to_20_cap5_avg7,slot_graph_5_to_20_cap5_avg7,score_slot_graph_5_to_20_cap5_avg7 \\
      --n-boot 5000 --seed 42

If you do not have an existing slot generation input handy, drop
``--merge-into`` / ``--merged-output``; the script will still write a
standalone ``monot5_generation_input.jsonl`` containing
``graph_top5``, ``graph_top7``, and ``monot5_slot_graph_5_to_20_cap5_avg7``.

Use ``--skip-monot5`` to emit only ``graph_top5`` and ``graph_top7``
(no GPU, no model download).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_MAP = {
    "hotpot": "results/study_hotpot_hipporag_colbert_500",
    "2wiki": "results/study_2wiki_hipporag_colbert_500",
    "nq": "results/study_nq_hipporag_colbert_500",
}


def monot5_slot_method_name(base_k: int, max_k: int, cap: int, target_avg_k: float) -> str:
    """Mirror ``score_slot_method_name`` / ``slot_text_rerank_*`` naming so
    downstream tooling (sig_test_qa.py, summarize_slot_ablation.py) treats
    this method like the other slot baselines."""
    avg_token = int(target_avg_k) if float(target_avg_k).is_integer() else target_avg_k
    return f"monot5_slot_graph_{base_k}_to_{max_k}_cap{cap}_avg{avg_token}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MonoT5 slot allocator (over ranks 6..20) and graph_top7 baselines.",
    )
    p.add_argument("--datasets", nargs="+", default=["hotpot", "2wiki"])
    p.add_argument(
        "--deep-retrieval-files", nargs="+", required=True,
        help="One JSONL per dataset (same order). Must contain retrieval.graph "
             "with at least --slot-max-k items per row.",
    )
    p.add_argument(
        "--routing-row-files", nargs="*", default=[],
        help="Optional routing_rows.jsonl per dataset. Defaults to "
             "<DATASET_MAP[d]>/routing_rows.jsonl.",
    )
    # Slot-allocation budget (must match the existing slot baselines).
    p.add_argument("--base-k", type=int, default=5,
                   help="Size of the fixed base context G_base(q). MonoT5 never displaces these.")
    p.add_argument("--slot-max-k", type=int, default=20,
                   help="Tail candidates come from graph ranks base_k+1 .. slot_max_k.")
    p.add_argument("--slot-per-query-cap", type=int, default=5,
                   help="Maximum number of tail slots a single query can receive.")
    p.add_argument("--slot-target-avg-k", type=float, default=7.0,
                   help="Average context length budget. Total tail slots = N*(avg_k - base_k).")
    # Independent fixed-top baseline.
    p.add_argument("--graph-top-k", type=int, default=7,
                   help="K for the independent graph_topK baseline (just G20[:K]).")
    p.add_argument("--include-graph-top5", action="store_true", default=True,
                   help="Pass graph_top5 through as a reference baseline (default on).")
    p.add_argument("--no-graph-top5", dest="include_graph_top5", action="store_false")
    # MonoT5 model.
    p.add_argument("--monot5-model", default="castorini/monot5-base-msmarco")
    p.add_argument("--monot5-batch-size", type=int, default=64)
    p.add_argument("--monot5-max-length", type=int, default=512)
    p.add_argument("--device", default=None,
                   help="cuda / cuda:0 / cpu. Defaults to cuda when available.")
    # IO.
    p.add_argument(
        "--score-cache", default="results/monot5_baseline/scores_cache.jsonl",
        help="Append-only JSONL cache of {qid, pid, score, model}. Reread on rerun to skip work.",
    )
    p.add_argument("--output-dir", default="results/monot5_baseline")
    p.add_argument(
        "--merge-into", default=None,
        help="Optional existing generation-input JSONL (e.g. the slot allocation "
             "ablation input). The new methods are added by id and the merged "
             "JSONL is written to --merged-output. graph_top5/8/10 and slot/MGEA "
             "methods already in the file are preserved.",
    )
    p.add_argument(
        "--merged-output", default=None,
        help="Where to write the merged generation input. Required when --merge-into is set.",
    )
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--skip-monot5", action="store_true",
                   help="Only emit graph_top5/graph_top7; skip MonoT5 entirely (no GPU, no download).")
    p.add_argument("--dry-run-no-model", action="store_true",
                   help="Skip MonoT5 loading; assume cache covers every (qid, pid) pair (debug only).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# MonoT5 scorer
# --------------------------------------------------------------------------- #
class MonoT5Scorer:
    """MonoT5 pointwise reranker: score = logit(true) - logit(false) on
    the first decoder step of (q, p) prompts. Standard Castorini prompt:
    ``Query: ... Document: ... Relevant:``.
    """

    def __init__(self, model_name: str, device: str | None, max_length: int) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[monot5] loading {model_name} on {device} ...", flush=True)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        self.true_id = self.tokenizer("true").input_ids[0]
        self.false_id = self.tokenizer("false").input_ids[0]
        self.decoder_start = self.model.config.decoder_start_token_id

    @staticmethod
    def format_prompt(question: str, passage_text: str, title: str | None = None) -> str:
        title = (title or "").strip()
        body = (passage_text or "").strip()
        if title and body and not body.lower().startswith(title.lower()):
            document = f"{title}. {body}"
        else:
            document = body or title
        return f"Query: {question} Document: {document} Relevant:"

    def score_batch(self, prompts: list[str]) -> list[float]:
        torch = self.torch
        if not prompts:
            return []
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)
        decoder_input_ids = torch.full(
            (input_ids.size(0), 1),
            self.decoder_start,
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
        logits = out.logits[:, 0, :]
        true_l = logits[:, self.true_id]
        false_l = logits[:, self.false_id]
        return (true_l - false_l).detach().float().cpu().tolist()


# --------------------------------------------------------------------------- #
# Score cache
# --------------------------------------------------------------------------- #
def load_score_cache(path: Path, model_name: str) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("model") != model_name:
                continue
            out[(str(r["qid"]), str(r["pid"]))] = float(r["score"])
    return out


def collect_pairs_to_score(
    examples: list[dict[str, Any]],
    base_k: int,
    slot_max_k: int,
    cache: dict[tuple[str, str], float],
) -> list[tuple[str, str, str, str, str]]:
    """Return (qid, pid, question, title, text) for tail candidates that need
    scoring. We only score ranks base_k .. slot_max_k because that is what
    MonoT5 needs in this slot-allocation setup."""
    todo: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ex in examples:
        qid = ex["id"]
        question = ex["question"]
        for pid in ex["graph_ids"][base_k:slot_max_k]:
            key = (qid, pid)
            if key in cache or key in seen:
                continue
            seen.add(key)
            passage = ex["passage_lookup"].get(pid)
            if not passage:
                continue
            todo.append(
                (qid, pid, question, str(passage.get("title") or pid), str(passage.get("text") or ""))
            )
    return todo


def run_monot5_scoring(
    todo: list[tuple[str, str, str, str, str]],
    scorer: MonoT5Scorer,
    cache_path: Path,
    model_name: str,
    batch_size: int,
) -> dict[tuple[str, str], float]:
    new_scores: dict[tuple[str, str], float] = {}
    if not todo:
        return new_scores
    print(f"[monot5] scoring {len(todo)} (q, p) tail pairs in batches of {batch_size} ...", flush=True)
    started = time.time()
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        prompts = [
            MonoT5Scorer.format_prompt(q, t, ttl) for (_qid, _pid, q, ttl, t) in chunk
        ]
        scores = scorer.score_batch(prompts)
        records = []
        for (qid, pid, _q, _ttl, _t), s in zip(chunk, scores):
            new_scores[(qid, pid)] = s
            records.append({"qid": qid, "pid": pid, "score": s, "model": model_name})
        append_jsonl(cache_path, records)
        if (start // batch_size) % 10 == 0:
            elapsed = time.time() - started
            done = start + len(chunk)
            rate = done / max(1.0, elapsed)
            print(
                f"[monot5] {done}/{len(todo)} pairs ({rate:.1f} pairs/s, elapsed={elapsed:.1f}s)",
                flush=True,
            )
    return new_scores


# --------------------------------------------------------------------------- #
# Method assembly
# --------------------------------------------------------------------------- #
def select_graph_top_k(example: dict[str, Any], top_k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pid in example["graph_ids"][:top_k]:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def select_monot5_slot(
    examples: list[dict[str, Any]],
    scores: dict[tuple[str, str], float],
    base_k: int,
    slot_max_k: int,
    per_query_cap: int,
    target_avg_k: float,
) -> dict[str, list[str]]:
    """Same global-budget logic as score_slot / text_rerank slot baselines,
    only the ranking signal is replaced with the MonoT5 score.

    Tail records: pid in graph_ids[base_k:slot_max_k] for each example.
    Total slots: round(N * (target_avg_k - base_k)), capped per query.
    Tie-break order matches eval_adaptive_context_budget.build_slot_plans:
    (-score, qid, rank, pid).
    """
    n = len(examples)
    total_slots = max(0, int(round(n * (target_avg_k - base_k))))
    total_slots = min(total_slots, per_query_cap * n)

    records: list[dict[str, Any]] = []
    for ex in examples:
        qid = ex["id"]
        graph_rank = {pid: idx + 1 for idx, pid in enumerate(ex["graph_ids"])}
        seen = set(ex["graph_ids"][:base_k])
        for pid in ex["graph_ids"][base_k:slot_max_k]:
            if pid in seen:
                continue
            seen.add(pid)
            s = scores.get((qid, pid))
            if s is None:
                continue
            records.append(
                {"qid": qid, "pid": pid, "rank": graph_rank.get(pid, 999), "score": float(s)}
            )

    records.sort(key=lambda item: (-item["score"], item["qid"], item["rank"], item["pid"]))
    selected_by_qid: dict[str, list[str]] = defaultdict(list)
    selected_total = 0
    for rec in records:
        if selected_total >= total_slots:
            break
        if len(selected_by_qid[rec["qid"]]) >= per_query_cap:
            continue
        selected_by_qid[rec["qid"]].append(rec["pid"])
        selected_total += 1

    # Merge G_base + selected tail (preserve base order, dedupe).
    out: dict[str, list[str]] = {}
    for ex in examples:
        base = list(dict.fromkeys(ex["graph_ids"][:base_k]))
        base_set = set(base)
        tail = [pid for pid in selected_by_qid.get(ex["id"], []) if pid not in base_set]
        out[ex["id"]] = base + tail
    return out


# --------------------------------------------------------------------------- #
# Example construction
# --------------------------------------------------------------------------- #
def build_examples(
    deep_rows: list[dict[str, Any]],
    routing_rows: dict[str, dict[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in deep_rows:
        qid = str(row["id"])
        retrieval = row.get("retrieval", {}) or {}
        graph_passages = list(retrieval.get("graph") or [])
        graph_ids = [str(p.get("id")) for p in graph_passages if p.get("id") is not None]
        gold = list(row.get("gold_passage_ids") or row.get("gold_titles") or [])
        passage_lookup: dict[str, dict[str, Any]] = {}
        for arr in retrieval.values():
            if not isinstance(arr, list):
                continue
            for p in arr:
                pid = p.get("id")
                if pid is not None:
                    passage_lookup.setdefault(str(pid), p)
        rrow = routing_rows.get(qid, {})
        examples.append(
            {
                "id": qid,
                "dataset": dataset,
                "question": row.get("question") or rrow.get("question"),
                "answer": row.get("answer") or rrow.get("answer"),
                "gold_passage_ids": list(gold),
                "graph_ids": graph_ids,
                "passage_lookup": passage_lookup,
                "row": row,
                "routing_row": rrow,
            }
        )
    return examples


# --------------------------------------------------------------------------- #
# Generation row + retrieval metrics
# --------------------------------------------------------------------------- #
def build_generation_row(example: dict[str, Any], rankings: dict[str, list[str]]) -> dict[str, Any]:
    lookup = example["passage_lookup"]
    retrieval: dict[str, list[dict[str, Any]]] = {
        method: [lookup[pid] for pid in ids if pid in lookup]
        for method, ids in rankings.items()
    }
    methods = list(rankings.keys())
    row = example["row"]
    rrow = example.get("routing_row") or {}
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": row.get("answer") or rrow.get("answer"),
        "gold_answer": row.get("gold_answer", row.get("answer") or rrow.get("answer")),
        "gold_answers": row.get("gold_answers", [row.get("answer") or rrow.get("answer")]),
        "dataset_name": row.get("dataset_name") or rrow.get("dataset_name") or example["dataset"],
        "workload": row.get("workload") or rrow.get("workload"),
        "question_type": row.get("question_type") or rrow.get("question_type"),
        "retrieval": retrieval,
        "main_table_methods": methods,
        "main_table_decisions": {
            method: {"selected_path": method, "top_k": len(retrieval[method])} for method in methods
        },
    }


def merge_into_existing(
    existing_path: Path,
    new_rows: list[dict[str, Any]],
    new_methods: list[str],
) -> list[dict[str, Any]]:
    """Add the new methods to each matching row in an existing generation input.
    Rows are matched by id. Methods already present in the existing row are
    left untouched -- we never overwrite. Rows in the existing file with no
    matching id are passed through unchanged.
    """
    by_id_new = {str(r["id"]): r for r in new_rows}
    merged: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    existing_rows = load_jsonl(existing_path)
    for ex_row in existing_rows:
        qid = str(ex_row.get("id"))
        new_row = by_id_new.get(qid)
        if new_row is None:
            merged.append(ex_row)
            continue
        matched_ids.add(qid)
        retrieval = dict(ex_row.get("retrieval") or {})
        decisions = dict(ex_row.get("main_table_decisions") or {})
        methods = list(ex_row.get("main_table_methods") or list(retrieval.keys()))
        added = []
        for method in new_methods:
            if method in retrieval:
                continue
            retrieval[method] = list(new_row["retrieval"].get(method, []))
            decisions[method] = {"selected_path": method, "top_k": len(retrieval[method])}
            added.append(method)
        for method in added:
            if method not in methods:
                methods.append(method)
        ex_row["retrieval"] = retrieval
        ex_row["main_table_decisions"] = decisions
        ex_row["main_table_methods"] = methods
        merged.append(ex_row)
    unmatched = [qid for qid in by_id_new if qid not in matched_ids]
    if unmatched:
        print(
            f"[merge] warning: {len(unmatched)} ids in deep retrieval are not "
            f"present in {existing_path.name} (first few: {unmatched[:5]}). "
            "These rows will NOT appear in the merged output -- the merged "
            "file inherits the row set of --merge-into.",
            flush=True,
        )
    return merged


def retrieval_recall(ids: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(set(ids) & gold) / len(gold)


def summarize(
    examples: list[dict[str, Any]],
    rankings_by_qid: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"recall_sum": 0.0, "len_sum": 0, "n": 0}
    )
    for ex in examples:
        gold = set(ex["gold_passage_ids"])
        for method, ids in rankings_by_qid[ex["id"]].items():
            stats = by_method[method]
            stats["recall_sum"] += retrieval_recall(ids, gold)
            stats["len_sum"] += len(ids)
            stats["n"] += 1
    summary: dict[str, Any] = {}
    for method, s in by_method.items():
        n = max(1, s["n"])
        summary[method] = {
            "n": s["n"],
            "avg_k": round(s["len_sum"] / n, 4),
            "recall@k": round(s["recall_sum"] / n, 4),
        }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if len(args.deep_retrieval_files) != len(args.datasets):
        raise SystemExit("--deep-retrieval-files must match --datasets in length and order.")
    if args.merge_into and not args.merged_output:
        raise SystemExit("--merged-output is required when --merge-into is set.")

    routing_files: list[Path]
    if args.routing_row_files:
        if len(args.routing_row_files) != len(args.datasets):
            raise SystemExit("--routing-row-files must match --datasets in length and order.")
        routing_files = [resolve(p) for p in args.routing_row_files]
    else:
        routing_files = []
        for d in args.datasets:
            if d not in DATASET_MAP:
                raise SystemExit(f"Unknown dataset {d!r}. Pass --routing-row-files explicitly.")
            routing_files.append(resolve(DATASET_MAP[d]) / "routing_rows.jsonl")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_cache_path = resolve(args.score_cache)

    all_examples: list[dict[str, Any]] = []
    for dataset, deep_path, route_path in zip(args.datasets, args.deep_retrieval_files, routing_files):
        deep_rows = load_jsonl(resolve(deep_path))
        route_rows = load_jsonl(route_path) if route_path.exists() else []
        routing_index = {str(r["id"]): r for r in route_rows}
        examples = build_examples(deep_rows, routing_index, dataset)
        if args.max_queries:
            examples = examples[: args.max_queries]
        short = [ex["id"] for ex in examples if len(ex["graph_ids"]) < args.slot_max_k]
        if short:
            print(
                f"[warn] {dataset}: {len(short)} queries have fewer than "
                f"slot_max_k={args.slot_max_k} graph candidates "
                f"(first few: {short[:5]}). MonoT5 will rank what is available.",
                flush=True,
            )
        all_examples.extend(examples)

    # MonoT5 scoring of tail candidates only (ranks base_k..slot_max_k).
    cache: dict[tuple[str, str], float] = {}
    if not args.skip_monot5:
        cache = load_score_cache(score_cache_path, args.monot5_model)
        print(
            f"[monot5] cache hit: {len(cache)} (qid, pid) pairs from {score_cache_path}",
            flush=True,
        )
        todo = collect_pairs_to_score(all_examples, args.base_k, args.slot_max_k, cache)
        if todo and args.dry_run_no_model:
            raise SystemExit(
                f"--dry-run-no-model set but {len(todo)} pairs missing from cache."
            )
        if todo:
            scorer = MonoT5Scorer(args.monot5_model, args.device, args.monot5_max_length)
            new_scores = run_monot5_scoring(
                todo, scorer, score_cache_path, args.monot5_model, args.monot5_batch_size,
            )
            cache.update(new_scores)

    # Method names.
    graph_top_method = f"graph_top{args.graph_top_k}"
    monot5_slot_method = monot5_slot_method_name(
        args.base_k, args.slot_max_k, args.slot_per_query_cap, args.slot_target_avg_k,
    )

    # Slot allocator over MonoT5 scores (skipped if --skip-monot5).
    slot_plan: dict[str, list[str]] = {}
    if not args.skip_monot5:
        slot_plan = select_monot5_slot(
            all_examples, cache,
            args.base_k, args.slot_max_k, args.slot_per_query_cap, args.slot_target_avg_k,
        )

    # Build per-query rankings.
    rankings_by_qid: dict[str, dict[str, list[str]]] = {}
    for ex in all_examples:
        rankings: dict[str, list[str]] = {}
        if args.include_graph_top5:
            rankings["graph_top5"] = select_graph_top_k(ex, 5)
        rankings[graph_top_method] = select_graph_top_k(ex, args.graph_top_k)
        if not args.skip_monot5:
            rankings[monot5_slot_method] = slot_plan[ex["id"]]
        rankings_by_qid[ex["id"]] = rankings

    # Standalone generation input.
    standalone_path = output_dir / "monot5_generation_input.jsonl"
    standalone_rows: list[dict[str, Any]] = []
    with standalone_path.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            row = build_generation_row(ex, rankings_by_qid[ex["id"]])
            standalone_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Optional merge into an existing slot generation input.
    merged_path: Path | None = None
    if args.merge_into:
        merge_into_path = resolve(args.merge_into)
        if not merge_into_path.exists():
            raise SystemExit(f"--merge-into file does not exist: {merge_into_path}")
        new_methods = [graph_top_method]
        if not args.skip_monot5:
            new_methods.append(monot5_slot_method)
        if args.include_graph_top5:
            # graph_top5 typically already exists in the slot input, but include
            # it in case the file lacks it.
            new_methods.insert(0, "graph_top5")
        merged_rows = merge_into_existing(merge_into_path, standalone_rows, new_methods)
        merged_path = resolve(args.merged_output)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with merged_path.open("w", encoding="utf-8") as fh:
            for r in merged_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Retrieval-side summary (overall + per-dataset).
    metrics = {
        "n_examples": len(all_examples),
        "datasets": args.datasets,
        "monot5_model": None if args.skip_monot5 else args.monot5_model,
        "base_k": args.base_k,
        "slot_max_k": args.slot_max_k,
        "slot_per_query_cap": args.slot_per_query_cap,
        "slot_target_avg_k": args.slot_target_avg_k,
        "graph_top_k": args.graph_top_k,
        "methods": {
            "graph_top_method": graph_top_method,
            "monot5_slot_method": None if args.skip_monot5 else monot5_slot_method,
            "include_graph_top5": args.include_graph_top5,
        },
        "per_method": summarize(all_examples, rankings_by_qid),
    }
    metrics_path = output_dir / "monot5_retrieval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in all_examples:
        by_dataset[ex["dataset"]].append(ex)
    breakdown: dict[str, Any] = {}
    for d, exs in by_dataset.items():
        sub = {ex["id"]: rankings_by_qid[ex["id"]] for ex in exs}
        breakdown[d] = summarize(exs, sub)
    (output_dir / "monot5_retrieval_metrics_by_dataset.json").write_text(
        json.dumps(breakdown, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"[monot5] standalone generation input: {standalone_path}")
    if merged_path is not None:
        print(f"[monot5] merged generation input:     {merged_path}")
    print(f"[monot5] retrieval metrics:           {metrics_path}")
    for method, stats in metrics["per_method"].items():
        print(
            f"  {method:<46s} avgK={stats['avg_k']:.2f}  R@k={stats['recall@k']:.4f}  n={stats['n']}"
        )


if __name__ == "__main__":
    main()
