#!/usr/bin/env python3
"""Assemble per-dataset summary.json (spec section 26)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.utils import read_json, read_jsonl, write_json  # noqa: E402


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gen-jsonl", default=None)
    args = ap.parse_args()

    dataset = args.dataset
    gen_path = args.gen_jsonl or f"outputs/generations/{dataset}/generations.jsonl"
    gens = list(read_jsonl(gen_path))

    prompt_tok = np.asarray([g["input_token_count"] for g in gens])
    gen_tok = np.asarray([g["generated_token_count"] for g in gens])
    prefixes = {g["prefix_id"] for g in gens}

    exp1 = {}
    e1_path = Path("outputs/metrics/exp1_aggregate.json")
    if e1_path.exists():
        agg = read_json(e1_path)
        for win in ("first_64", "first_128"):
            key = f"{dataset}|{win}"
            if key in agg:
                exp1[f"within_prefix_cosine_{win}_mean"] = agg[key]["cosine_mean"]

    exp2 = {}
    e2_path = Path("outputs/metrics/exp2_metrics.json")
    if e2_path.exists():
        m = read_json(e2_path)
        for win in ("first_64", "first_128"):
            key = f"{dataset}|{win}"
            if key in m:
                exp2[f"cross_better_rate_{win}"] = m[key]["cross_better_rate"]
        k128 = f"{dataset}|first_128"
        if k128 in m:
            exp2["mean_cross_minus_same_margin_first128"] = m[k128]["margin_mean"]

    summary = {
        "dataset": dataset,
        "num_prefix_groups": len(prefixes),
        "num_requests": len(gens),
        "prompt_tokens": {
            "mean": float(prompt_tok.mean()) if len(prompt_tok) else 0.0,
            "p50": pct(prompt_tok, 50),
            "p90": pct(prompt_tok, 90),
        },
        "generated_tokens": {
            "mean": float(gen_tok.mean()) if len(gen_tok) else 0.0,
            "p50": pct(gen_tok, 50),
            "p90": pct(gen_tok, 90),
            "fraction_ge_64": float((gen_tok >= 64).mean()) if len(gen_tok) else 0.0,
            "fraction_ge_128": float((gen_tok >= 128).mean()) if len(gen_tok) else 0.0,
            "fraction_hit_cap": float(
                np.mean([g["hit_max_new_tokens"] for g in gens])
            ) if gens else 0.0,
        },
        "exp1": exp1,
        "exp2": exp2,
    }
    out = f"outputs/metrics/{dataset}_summary.json"
    write_json(out, summary)
    print(f"[summary:{dataset}] wrote {out}")
    import json

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
