from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib

from .schema import Example, load_jsonl, write_jsonl
from .slot_allocator import SlotAllocator, SlotConfig
from .triage import DenseProbeTriage, TriageConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mgea",
        description="Train and run Marginal Graph Evidence Allocation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    oof = subparsers.add_parser("oof", help="Run paper-style query-grouped OOF evaluation.")
    _add_input(oof)
    _add_method_options(oof)
    oof.add_argument("--output", required=True, help="Per-query JSONL output.")
    oof.add_argument("--metrics", required=True, help="Summary JSON output.")

    fit = subparsers.add_parser("fit", help="Fit both MGEA layers on all labeled queries.")
    _add_input(fit)
    _add_method_options(fit)
    fit.add_argument("--model-output", required=True, help="joblib model bundle.")

    predict = subparsers.add_parser("predict", help="Apply a fitted MGEA model bundle.")
    _add_input(predict)
    predict.add_argument("--model", required=True, help="Model bundle written by `mgea fit`.")
    predict.add_argument("--output", required=True, help="Per-query JSONL output.")
    predict.add_argument("--triage-rate", type=float, default=0.527)
    return parser


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Retrieval JSONL; see README for schema.")


def _add_method_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--target-avg-k", type=float, default=7.0)
    parser.add_argument("--per-query-cap", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--triage-rate", type=float, default=0.527)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "oof":
        run_oof(args)
    elif args.command == "fit":
        run_fit(args)
    elif args.command == "predict":
        run_predict(args)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command: {args.command}")


def run_oof(args: argparse.Namespace) -> None:
    examples = load_jsonl(args.input)
    slot = SlotAllocator(_slot_config(args))
    triage = DenseProbeTriage(_triage_config(args))
    slot_scores, slot_metrics = slot.out_of_fold(examples)
    triage_scores, triage_metrics = triage.out_of_fold(examples)
    selected = slot.allocate(examples, slot_scores)
    graph_contexts = slot.contexts(examples, selected)
    invocations = triage.select_invocations(
        examples, triage_scores, target_rate=args.triage_rate
    )
    rows = _prediction_rows(
        examples,
        slot_scores,
        selected,
        graph_contexts,
        triage_scores,
        invocations,
        slot.config.base_k,
    )
    write_jsonl(args.output, rows)
    metrics = {
        "slot_allocator": {
            **slot_metrics,
            **_selection_metrics(
                examples, selected, graph_contexts, base_k=slot.config.base_k
            ),
        },
        "dense_probe_triage": {
            **triage_metrics,
            "target_invocation_rate": args.triage_rate,
            "actual_invocation_rate": sum(invocations.values()) / max(1, len(invocations)),
        },
    }
    _write_json(args.metrics, metrics)


def run_fit(args: argparse.Namespace) -> None:
    examples = load_jsonl(args.input)
    slot = SlotAllocator(_slot_config(args)).fit(examples)
    triage = DenseProbeTriage(_triage_config(args)).fit(examples)
    destination = Path(args.model_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"slot_allocator": slot, "dense_probe_triage": triage}, destination)


def run_predict(args: argparse.Namespace) -> None:
    examples = load_jsonl(args.input)
    bundle = joblib.load(args.model)
    slot: SlotAllocator = bundle["slot_allocator"]
    triage: DenseProbeTriage = bundle["dense_probe_triage"]
    slot_scores = slot.predict_scores(examples)
    triage_scores = triage.predict_scores(examples)
    selected = slot.allocate(examples, slot_scores)
    graph_contexts = slot.contexts(examples, selected)
    invocations = triage.select_invocations(
        examples, triage_scores, target_rate=args.triage_rate
    )
    write_jsonl(
        args.output,
        _prediction_rows(
            examples,
            slot_scores,
            selected,
            graph_contexts,
            triage_scores,
            invocations,
            slot.config.base_k,
        ),
    )


def _prediction_rows(
    examples: Sequence[Example],
    slot_scores: dict[str, dict[str, float]],
    selected: dict[str, list[str]],
    graph_contexts: dict[str, list[Any]],
    triage_scores: dict[str, float],
    invocations: dict[str, bool],
    base_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        invoke_graph = bool(invocations.get(example.id, False))
        final_context = (
            graph_contexts.get(example.id, [])
            if invoke_graph
            else list(example.dense[:base_k])
        )
        rows.append(
            {
                "id": example.id,
                "question": example.question,
                "triage_probability": triage_scores.get(example.id),
                "invoke_graph": invoke_graph,
                "slot_scores": slot_scores.get(example.id, {}),
                "selected_tail_ids": selected.get(example.id, []),
                "mgea_graph_context": [
                    passage.to_dict() for passage in graph_contexts.get(example.id, [])
                ],
                "final_context_source": "graph_mgea" if invoke_graph else "dense_top5",
                "final_context": [passage.to_dict() for passage in final_context],
            }
        )
    return rows


def _selection_metrics(
    examples: Sequence[Example],
    selected: dict[str, list[str]],
    contexts: dict[str, list[Any]],
    *,
    base_k: int,
) -> dict[str, Any]:
    selected_positive = 0
    for example in examples:
        missing = example.gold_keys - example.covered_gold(example.graph[:base_k])
        selected_passages = [
            passage for passage in example.graph if passage.id in set(selected.get(example.id, []))
        ]
        selected_positive += sum(bool(passage.evidence_keys & missing) for passage in selected_passages)
    selected_count = sum(len(values) for values in selected.values())
    return {
        "selected_slots": selected_count,
        "selected_positive_slots": selected_positive,
        "average_context_k": sum(len(context) for context in contexts.values()) / max(1, len(examples)),
    }


def _slot_config(args: argparse.Namespace) -> SlotConfig:
    return SlotConfig(
        base_k=args.base_k,
        max_k=args.max_k,
        target_avg_k=args.target_avg_k,
        per_query_cap=args.per_query_cap,
        folds=args.folds,
        random_seed=args.seed,
    )


def _triage_config(args: argparse.Namespace) -> TriageConfig:
    return TriageConfig(top_k=args.base_k, folds=args.folds, random_seed=args.seed)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
