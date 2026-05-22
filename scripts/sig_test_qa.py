"""Paired significance test on per-sample QA metrics.

McNemar on EM (binary), paired bootstrap on F1 (continuous).
Uses the per-sample JSONL written by evaluate_generations.py --per-sample-output.

Usage:
  python3 scripts/sig_test_qa.py <per_sample.jsonl> \\
      --baseline graph --compare graph_plus_rewrite_B1,graph_plus_union_B1,graph_plus_dense_B1 \\
      --n-boot 5000

  # Equivalent:
  python3 scripts/sig_test_qa.py --input <per_sample.jsonl> \\
      --baseline graph --compare graph_plus_rewrite_B1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value using binomial tail. b,c are discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided p = 2 * sum(binom(n, i) * 0.5^n, i=0..k)
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_pvalue(deltas: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Returns (mean_delta, lower_95, upper_95). p-value via bootstrap CI sign."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        samp = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return sum(deltas) / n, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("per_sample", type=Path, nargs="?")
    ap.add_argument("--input", dest="input_path", type=Path, default=None)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--compare", required=True, help="Comma-separated method names to test vs baseline.")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_sample_path = args.input_path or args.per_sample
    if per_sample_path is None:
        ap.error("per-sample JSONL is required, either positionally or via --input")

    rows = load(per_sample_path)
    # per-sample schema written by evaluate_generations.py:
    #   {"id": ..., "methods": {method_name: {"exact_match": .., "f1": .., ...}, ...}}
    per_id: dict[str, dict[str, dict]] = {}
    for r in rows:
        qid = r.get("id") or r.get("sample_id")
        methods = r.get("methods") or {}
        if qid is None or not isinstance(methods, dict):
            continue
        for m, rec in methods.items():
            if not isinstance(rec, dict):
                continue
            per_id.setdefault(qid, {})[m] = {
                "em": float(rec.get("exact_match", 0)),
                "f1": float(rec.get("f1", 0)),
            }

    compare_methods = [x.strip() for x in args.compare.split(",") if x.strip()]

    print(f"N samples with baseline+any-compare paired = {len(per_id)}")
    for m in compare_methods:
        paired = [(v[args.baseline], v[m]) for v in per_id.values()
                  if args.baseline in v and m in v]
        if not paired:
            print(f"[{m}] no paired samples"); continue

        em_base = [p[0]["em"] for p in paired]
        em_cmp  = [p[1]["em"] for p in paired]
        b = sum(1 for a, c in zip(em_base, em_cmp) if a == 1 and c == 0)  # baseline only
        c = sum(1 for a, c in zip(em_base, em_cmp) if a == 0 and c == 1)  # compare only
        p_em = mcnemar_p(b, c)
        em_delta = (sum(em_cmp) - sum(em_base)) / len(paired)

        f1_deltas = [p[1]["f1"] - p[0]["f1"] for p in paired]
        mean_d, lo, hi = paired_bootstrap_pvalue(f1_deltas, args.n_boot, args.seed)
        p_f1_sign = "sig" if (lo > 0 or hi < 0) else "ns"

        print(f"\n=== {m} vs {args.baseline}  (n={len(paired)}) ===")
        print(f"  EM Δ = {em_delta:+.4f}   McNemar b={b}, c={c}, p = {p_em:.4g}")
        print(f"  F1 Δ = {mean_d:+.4f}   bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]  ({p_f1_sign})")


if __name__ == "__main__":
    main()
