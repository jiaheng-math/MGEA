from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import probe_feature_names, query_feature_names
from src.utils import ensure_dir, load_yaml


REQUIRED_ROW_FIELDS = {
    "id",
    "label",
    "dense_recall@3",
    "dense_recall@5",
    "graph_recall@3",
    "graph_recall@5",
    "dense_latency_ms",
    "graph_latency_ms",
    "dense_query_llm_calls",
    "graph_query_llm_calls",
    "dense_query_token_cost",
    "graph_query_token_cost",
}

REQUIRED_METRICS_TOP_KEYS = {"retrieval_methods", "routing_methods"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate backend-robustness routing artifacts.")
    parser.add_argument("--config", required=True, help="Path to backend robustness YAML config.")
    parser.add_argument(
        "--require-exact-shared",
        action="store_true",
        help="Require all pairs within a dataset to have exactly the same labeled query ids.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to <config output_dir>/validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    project_root = Path(args.config).resolve().parent.parent
    base_output_dir = resolve_path(project_root, str(config["output_dir"]))
    output_dir = ensure_dir(resolve_path(project_root, args.output_dir) if args.output_dir else base_output_dir / "validation")

    datasets = config.get("datasets", [])
    rows_summary: list[dict[str, Any]] = []
    dataset_summary: list[dict[str, Any]] = []
    errors: list[str] = []

    required_features = set(query_feature_names()) | set(probe_feature_names())
    required_row_fields = REQUIRED_ROW_FIELDS | required_features

    for dataset_cfg in datasets:
        dataset_name = str(dataset_cfg["name"])
        pairs = dataset_cfg.get("pairs", [])
        dataset_pairs: list[dict[str, Any]] = []
        raw_id_sets: dict[str, set[str]] = {}
        labeled_id_sets: dict[str, set[str]] = {}

        for pair_cfg in pairs:
            pair_name = str(pair_cfg["name"])
            dense_backend = str(pair_cfg["dense_backend"])
            graph_backend = str(pair_cfg["graph_backend"])
            routing_rows_path = resolve_path(project_root, str(pair_cfg["routing_rows_path"]))
            metrics_path = resolve_path(project_root, str(pair_cfg["metrics_json_path"])) if pair_cfg.get("metrics_json_path") else None

            pair_key = f"{dataset_name}::{pair_name}"
            pair_errors: list[str] = []

            routing_exists = routing_rows_path.exists()
            metrics_exists = bool(metrics_path and metrics_path.exists())
            labeled_ids: set[str] = set()
            raw_ids: set[str] = set()
            unlabeled_count = 0
            missing_row_fields: set[str] = set()

            if routing_exists:
                routing_rows = load_jsonl(routing_rows_path)
                for row in routing_rows:
                    raw_ids.add(str(row["id"]))
                    if row.get("label") is None:
                        unlabeled_count += 1
                        continue
                    labeled_ids.add(str(row["id"]))
                    missing_row_fields.update(required_row_fields - set(row.keys()))
                if not routing_rows:
                    pair_errors.append(f"{pair_key}: routing_rows.jsonl is empty")
            else:
                pair_errors.append(f"{pair_key}: missing routing rows at {routing_rows_path}")

            if missing_row_fields:
                pair_errors.append(
                    f"{pair_key}: missing required routing row fields: {', '.join(sorted(missing_row_fields))}"
                )

            if metrics_path:
                if metrics_exists:
                    payload = load_json(metrics_path)
                    missing_top_keys = REQUIRED_METRICS_TOP_KEYS - set(payload.keys())
                    if missing_top_keys:
                        pair_errors.append(
                            f"{pair_key}: metrics.json missing top-level keys: {', '.join(sorted(missing_top_keys))}"
                        )
                    else:
                        routing_methods = payload.get("routing_methods", {})
                        if "methods" not in routing_methods:
                            pair_errors.append(f"{pair_key}: metrics.json routing_methods.methods missing")
                        retrieval_methods = payload.get("retrieval_methods", {})
                        if not retrieval_methods:
                            pair_errors.append(f"{pair_key}: metrics.json retrieval_methods is empty")
                else:
                    pair_errors.append(f"{pair_key}: missing metrics.json at {metrics_path}")

            raw_id_sets[pair_name] = raw_ids
            labeled_id_sets[pair_name] = labeled_ids
            row = {
                "dataset": dataset_name,
                "pair": pair_name,
                "dense_backend": dense_backend,
                "graph_backend": graph_backend,
                "routing_rows_path": str(routing_rows_path),
                "metrics_json_path": str(metrics_path) if metrics_path else "",
                "routing_rows_exists": routing_exists,
                "metrics_json_exists": metrics_exists,
                "raw_queries": len(raw_ids),
                "labeled_queries": len(labeled_ids),
                "unlabeled_rows": unlabeled_count,
                "num_errors": len(pair_errors),
                "errors": " | ".join(pair_errors),
            }
            rows_summary.append(row)
            dataset_pairs.append(row)
            errors.extend(pair_errors)

        exact_shared = True
        exact_labeled_shared = True
        shared_count = 0
        shared_labeled_count = 0
        if dataset_pairs:
            all_sets = list(raw_id_sets.values())
            shared_ids = set.intersection(*all_sets) if all_sets else set()
            shared_count = len(shared_ids)
            first_set = next(iter(raw_id_sets.values()), set())
            exact_shared = all(id_set == first_set for id_set in raw_id_sets.values())

            labeled_sets = list(labeled_id_sets.values())
            shared_labeled_ids = set.intersection(*labeled_sets) if labeled_sets else set()
            shared_labeled_count = len(shared_labeled_ids)
            first_labeled_set = next(iter(labeled_id_sets.values()), set())
            exact_labeled_shared = all(id_set == first_labeled_set for id_set in labeled_id_sets.values())

            if args.require_exact_shared and not exact_shared:
                errors.append(
                    f"{dataset_name}: pairs do not share exactly the same raw query ids "
                    f"(shared={shared_count}, counts={[len(v) for v in raw_id_sets.values()]})"
                )

        dataset_summary.append(
            {
                "dataset": dataset_name,
                "num_pairs": len(dataset_pairs),
                "require_exact_shared": bool(args.require_exact_shared),
                "exact_shared_raw_ids": exact_shared,
                "shared_raw_query_count": shared_count,
                "exact_shared_labeled_ids": exact_labeled_shared,
                "shared_labeled_query_count": shared_labeled_count,
                "pair_raw_counts": {pair: len(ids) for pair, ids in raw_id_sets.items()},
                "pair_labeled_counts": {pair: len(ids) for pair, ids in labeled_id_sets.items()},
            }
        )

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir),
        "require_exact_shared": bool(args.require_exact_shared),
        "num_errors": len(errors),
        "datasets": dataset_summary,
        "pairs": rows_summary,
        "errors": errors,
    }

    (output_dir / "artifact_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "artifact_validation_pairs.csv", rows_summary)
    write_csv(output_dir / "artifact_validation_datasets.csv", dataset_summary)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
