"""Run slot QA validation across one or more reader models.

This is a thin orchestrator around:

  1. scripts/batch_generate_from_retrieval.py
  2. scripts/evaluate_generations.py
  3. scripts/sig_test_qa.py

It does not build retrieval artifacts. Feed it the generation input written by
eval_adaptive_context_budget.py --generation-input-output.

Example:
  python scripts/run_slot_reader_sweep.py \
    --generation-input results/marginal_slot_allocation_generation_input_ablation.jsonl \
    --models gpt-4.1 gpt-4o-mini \
    --output-dir results/slot_reader_sweep \
    --methods graph_top5,graph_top8,graph_top10,random_slot_graph_5_to_20_cap5_avg7,score_slot_graph_5_to_20_cap5_avg7,slot_graph_5_to_20_cap5_avg7,slot_no_probe_graph_5_to_20_cap5_avg7 \
    --baseline graph_top5 \
    --compare slot_graph_5_to_20_cap5_avg7,slot_no_probe_graph_5_to_20_cap5_avg7,graph_top8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run slot QA generation/evaluation for multiple reader models.")
    parser.add_argument("--generation-input", required=True, help="JSONL produced by eval_adaptive_context_budget.py.")
    parser.add_argument("--models", nargs="+", required=True, help="Reader model names, e.g. gpt-4.1 gpt-4o-mini.")
    parser.add_argument("--output-dir", default="results/slot_reader_sweep")
    parser.add_argument(
        "--methods",
        default="auto",
        help="Comma-separated methods to generate/evaluate, or auto for methods present in the input rows.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Use 20 so variable-length slot contexts are not truncated.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Optional seed passed through to batch_generate_from_retrieval.py.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--baseline", default="graph_top5", help="Baseline method for paired significance tests.")
    parser.add_argument(
        "--compare",
        default=None,
        help="Comma-separated methods to compare with --baseline. Defaults to all evaluated methods except baseline.",
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_input = resolve_path(args.generation_input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = infer_methods(generation_input) if args.methods == "auto" else parse_csv(args.methods)
    compare_methods = parse_csv(args.compare) if args.compare else [m for m in methods if m != args.baseline]
    summary: dict[str, Any] = {
        "generation_input": str(generation_input),
        "methods": methods,
        "baseline": args.baseline,
        "compare": compare_methods,
        "models": {},
    }

    for model in args.models:
        slug = slugify(model)
        generation_output = output_dir / f"generations_{slug}.jsonl"
        metrics_output = output_dir / f"qa_metrics_{slug}.json"
        per_sample_output = output_dir / f"qa_per_sample_{slug}.jsonl"
        sig_output = output_dir / f"significance_{slug}.txt"
        cache_path = output_dir / f"generations_{slug}.sqlite"

        generate_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "batch_generate_from_retrieval.py"),
            "--input",
            str(generation_input),
            "--output",
            str(generation_output),
            "--cache-path",
            str(cache_path),
            "--model",
            model,
            "--methods",
            ",".join(methods),
            "--top-k",
            str(args.top_k),
            "--api-key-env",
            args.api_key_env,
            "--temperature",
            str(args.temperature),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
            "--max-completion-tokens",
            str(args.max_completion_tokens),
        ]
        if args.generation_seed is not None:
            generate_cmd.extend(["--seed", str(args.generation_seed)])
        if args.base_url:
            generate_cmd.extend(["--base-url", args.base_url])

        eval_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_generations.py"),
            "--input",
            str(generation_output),
            "--output",
            str(metrics_output),
            "--methods",
            ",".join(methods),
            "--per-sample-output",
            str(per_sample_output),
        ]
        sig_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "sig_test_qa.py"),
            "--input",
            str(per_sample_output),
            "--baseline",
            args.baseline,
            "--compare",
            ",".join(compare_methods),
            "--n-boot",
            str(args.n_boot),
            "--seed",
            str(args.seed),
        ]

        run_command(generate_cmd, args.dry_run)
        run_command(eval_cmd, args.dry_run)
        sig_text = run_command(sig_cmd, args.dry_run, capture=True)
        if not args.dry_run:
            sig_output.write_text(sig_text, encoding="utf-8")

        summary["models"][model] = {
            "generation_output": str(generation_output),
            "metrics_output": str(metrics_output),
            "per_sample_output": str(per_sample_output),
            "significance_output": str(sig_output),
            "metrics": load_json(metrics_output) if metrics_output.exists() else None,
        }

    summary_path = output_dir / "reader_sweep_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"summary: {summary_path}")


def run_command(cmd: list[str], dry_run: bool, *, capture: bool = False) -> str:
    printable = " ".join(shell_quote(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return ""
    if capture:
        completed = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, end="")
        return completed.stdout
    subprocess.run(cmd, check=True)
    return ""


def infer_methods(path: Path) -> list[str]:
    methods: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            retrieval = row.get("retrieval", {})
            if isinstance(retrieval, dict):
                methods.update(str(method) for method in retrieval)
            main_methods = row.get("main_table_methods")
            if isinstance(main_methods, list):
                methods.update(str(method) for method in main_methods)
            break
    preferred = [
        "graph_top5",
        "graph_top8",
        "graph_top10",
        "random_slot_graph_5_to_20_cap5_avg7",
        "score_slot_graph_5_to_20_cap5_avg7",
        "slot_graph_5_to_20_cap5_avg7",
        "slot_no_probe_graph_5_to_20_cap5_avg7",
        "slot_probe_only_graph_5_to_20_cap5_avg7",
        "slot_graph_only_graph_5_to_20_cap5_avg7",
        "slot_no_dense_support_graph_5_to_20_cap5_avg7",
        "slot_no_novelty_graph_5_to_20_cap5_avg7",
        "slot_slot_only_graph_5_to_20_cap5_avg7",
    ]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(sorted(methods - set(ordered)))
    return ordered


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "model"


def shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
