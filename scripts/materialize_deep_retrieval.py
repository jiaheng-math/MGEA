"""Materialize deep dense/graph retrieval from cached indexes.

The main study configs usually save only top-5 because top_k_values=[3,5].
For evidence packing we need a real candidate pool, e.g. dense top-20 and
HippoRAG graph top-20. This script reuses the already-built ColBERT/HippoRAG
caches and writes a retrieval_results-style JSONL file without rebuilding
indexes.

Example:
  python scripts/materialize_deep_retrieval.py \
    --config configs/study_hotpot_hipporag_colbert_shared.yaml \
    --top-k 20 \
    --output results/study_hotpot_hipporag_colbert_500/retrieval_results_deep20.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import build_corpus, load_dataset, load_shared_corpus
from src.retrieval_dense import build_dense_retriever
from src.retrieval_graph import build_graph_retriever
from src.utils import RetrievedPassage, load_spacy_model, load_yaml, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write dense/graph top-N retrieval rows from cached indexes.")
    parser.add_argument("--config", required=True, help="Study YAML config.")
    parser.add_argument("--top-k", type=int, default=20, help="Depth to retrieve from each backend.")
    parser.add_argument("--output", required=True, help="Output retrieval_results-style JSONL.")
    parser.add_argument(
        "--prefix-retrieval",
        default=None,
        help=(
            "Optional existing retrieval_results.jsonl. When set, preserve its dense/graph "
            "top --preserve-prefix-k as the prefix, then append newly materialized deep candidates."
        ),
    )
    parser.add_argument("--preserve-prefix-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--allow-index-rebuild",
        action="store_true",
        help=(
            "Allow retriever adapters to rebuild indexes if cache fingerprints do not match. "
            "Default is false because this script is intended to reuse cloud caches."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(str(config_path))
    project_root = config_path.resolve().parent.parent
    set_random_seed(int(config.get("random_seed", 42)))

    # Critical: reuse cloud-built caches. Do not rebuild ColBERT or HippoRAG here.
    config = dict(config)
    config["colbert_rebuild_index"] = False
    config["colbert_trust_existing_index"] = True
    config["hipporag_rebuild_index"] = False
    config["lightrag_rebuild_index"] = False

    dataset_path = resolve_path(project_root, str(config["dataset_path"]))
    samples = load_dataset(
        dataset_path=str(dataset_path),
        subset_size=int(config["subset_size"]) if config.get("subset_size") is not None else None,
        random_seed=int(config.get("random_seed", 42)),
    )
    if args.max_queries:
        samples = samples[: args.max_queries]

    shared_corpus_path = config.get("shared_corpus_path")
    if shared_corpus_path:
        corpus = load_shared_corpus(str(resolve_path(project_root, str(shared_corpus_path))))
    else:
        corpus = build_corpus(samples)

    if not args.allow_index_rebuild:
        validate_cache_fingerprints(config=config, corpus=corpus, project_root=project_root)
        apply_cached_colbert_manifest_overrides(config=config, project_root=project_root)

    nlp = load_spacy_model()
    dense_retriever = build_dense_retriever(corpus=corpus, config=config, project_root=project_root)
    graph_retriever = build_graph_retriever(corpus=corpus, nlp=nlp, config=config, project_root=project_root)

    output = resolve_path(project_root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_by_id = load_prefix_retrieval(project_root, args.prefix_retrieval)

    start = time.time()
    with output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            dense = dense_retriever.retrieve(sample.question, top_k=args.top_k)
            graph = graph_retriever.retrieve(sample.question, top_k=args.top_k)
            dense_items = serialize_passages(dense)
            graph_items = serialize_passages(graph)
            prefix_row = prefix_by_id.get(str(sample.id), {})
            if prefix_row:
                prefix_retrieval = prefix_row.get("retrieval", {})
                dense_items = preserve_prefix(
                    prefix_items=prefix_retrieval.get("dense", []),
                    deep_items=dense_items,
                    prefix_k=args.preserve_prefix_k,
                    total_k=args.top_k,
                )
                graph_items = preserve_prefix(
                    prefix_items=prefix_retrieval.get("graph", []),
                    deep_items=graph_items,
                    prefix_k=args.preserve_prefix_k,
                    total_k=args.top_k,
                )
            row = {
                "id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "gold_answers": sample_gold_answers(sample),
                "dataset_name": sample.dataset_name,
                "workload": sample.workload,
                "question_type": sample.question_type,
                "gold_titles": list(sample.gold_titles),
                "gold_passage_ids": list(sample.gold_passage_ids),
                "retrieval": {
                    "dense": dense_items,
                    "graph": graph_items,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.progress_every and index % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"progress {index}/{len(samples)} [{elapsed:.1f}s]", flush=True)

    print(f"wrote {output} rows={len(samples)} top_k={args.top_k}", flush=True)


def serialize_passages(passages: list[RetrievedPassage]) -> list[dict[str, Any]]:
    return [
        {
            "id": passage.id,
            "title": passage.title,
            "source_doc_id": passage.source_doc_id,
            "score": float(passage.score),
            "text": passage.text,
        }
        for passage in passages
    ]


def load_prefix_retrieval(project_root: Path, path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    resolved = resolve_path(project_root, path)
    rows = load_jsonl(resolved)
    return {str(row.get("id")): row for row in rows if row.get("id") is not None}


def preserve_prefix(
    prefix_items: Any,
    deep_items: list[dict[str, Any]],
    prefix_k: int,
    total_k: int,
) -> list[dict[str, Any]]:
    prefix = [item for item in list(prefix_items or [])[:prefix_k] if isinstance(item, dict)]
    seen = {passage_id(item) for item in prefix if passage_id(item)}
    out = list(prefix)
    for item in deep_items:
        pid = passage_id(item)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(item)
        if len(out) >= total_k:
            break
    return out[:total_k]


def passage_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("source_doc_id") or "")


def sample_gold_answers(sample: Any) -> list[str]:
    for attr in ("answer_aliases", "gold_answers", "answers", "answer_aliases_list"):
        value = getattr(sample, attr, None)
        if isinstance(value, list):
            answers = [str(item).strip() for item in value if str(item).strip()]
            if answers:
                return dedupe(answers)
    answer = str(getattr(sample, "answer", "") or "").strip()
    return [answer] if answer else []


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def validate_cache_fingerprints(config: dict[str, Any], corpus: list[Any], project_root: Path) -> None:
    expected = corpus_fingerprint(corpus)

    colbert_root = resolve_path(project_root, str(config.get("colbert_root", "colbert_cache/default")))
    colbert_manifest = colbert_root / str(config.get("colbert_experiment_name", "pilot0_colbert")) / "pilot0_colbert_manifest.json"
    check_manifest(colbert_manifest, expected, "ColBERT")

    graph_backend = str(config.get("graph_backend", "bm25")).lower()
    if graph_backend == "hipporag":
        hipporag_save_dir = resolve_path(project_root, str(config.get("hipporag_save_dir", "hipporag_cache/default")))
        check_manifest(hipporag_save_dir / "pilot0_manifest.json", expected, "HippoRAG")


def apply_cached_colbert_manifest_overrides(config: dict[str, Any], project_root: Path) -> None:
    """Align config with an existing ColBERT cache before constructing Searcher.

    Some historical cloud caches have a valid corpus fingerprint but stale
    index metadata names. For materialization we want to reuse the index, not
    rebuild it. The retriever still checks the corpus fingerprint before this
    function is called.
    """
    if str(config.get("dense_backend", "")).lower() not in {"colbert", "colbertv2"}:
        return
    colbert_root = resolve_path(project_root, str(config.get("colbert_root", "colbert_cache/default")))
    manifest_path = colbert_root / str(config.get("colbert_experiment_name", "pilot0_colbert")) / "pilot0_colbert_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for manifest_key, config_key in [
        ("index_name", "colbert_index_name"),
        ("nbits", "colbert_nbits"),
        ("partitions", "colbert_partitions"),
        ("doc_maxlen", "colbert_doc_maxlen"),
        ("query_maxlen", "colbert_query_maxlen"),
        ("kmeans_niters", "colbert_kmeans_niters"),
        ("checkpoint", "colbert_checkpoint"),
    ]:
        if manifest_key in manifest:
            config[config_key] = manifest[manifest_key]


def check_manifest(path: Path, expected_fingerprint: str, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{label} manifest not found: {path}\n"
            "Use the exact config that produced the cached index, or rerun with --allow-index-rebuild."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload.get("corpus_fingerprint")
    if actual != expected_fingerprint:
        raise SystemExit(
            f"{label} cache fingerprint mismatch: {path}\n"
            f"manifest={actual}\n"
            f"config_corpus={expected_fingerprint}\n"
            "This usually means the config points at a different corpus than the cache. "
            "Use the exact original config; do not rebuild unless this is intentional."
        )


def corpus_fingerprint(corpus: list[Any]) -> str:
    payload = [
        {
            "id": passage.id,
            "title": passage.title,
            "text": passage.text,
        }
        for passage in corpus
    ]
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
