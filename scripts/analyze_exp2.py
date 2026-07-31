#!/usr/bin/env python3
"""Experiment 2: cross-prefix expert affinity.

For each request, find its best same-prefix neighbor and best cross-prefix
neighbor (by mean per-layer cosine). Compute the Cross-Better Rate, similarity
margin distribution, and threshold sweeps. Also emit within- vs cross-prefix
pair samples (each request + K random cross negatives + best cross) for
distribution comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import mean_cosine  # noqa: E402
from src.sig_store import SignatureStore  # noqa: E402
from src.utils import write_json  # noqa: E402

PRIMARY_WINDOWS = ["first_64", "first_128", "full_decode"]
THRESHOLDS = [0.7, 0.8, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/metrics/signatures.npz")
    ap.add_argument("--meta", default="outputs/metrics/signatures_meta.parquet")
    ap.add_argument("--windows", default=",".join(PRIMARY_WINDOWS))
    ap.add_argument("--neg-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/metrics/exp2_cross_prefix_nn.parquet")
    ap.add_argument("--pairs-out", default="outputs/metrics/exp2_pair_samples.parquet")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    store = SignatureStore(args.npz, args.meta)
    windows = args.windows.split(",")

    nn_rows = []
    pair_rows = []
    metrics_summary: dict[str, dict] = {}

    for dataset in store.datasets():
        for window in windows:
            sub = store.subset(dataset, window)
            sub = sub[sub["valid_window"]].reset_index(drop=True)
            n_prefixes = sub["prefix_id"].nunique()
            if len(sub) < 2 or n_prefixes < 2:
                # Exp2 needs >= 2 prefix groups
                continue

            recs = sub.to_dict("records")
            norms = {r["request_id"]: store.normalized(r["key"]) for r in recs}
            ids = [r["request_id"] for r in recs]
            pref = {r["request_id"]: r["prefix_id"] for r in recs}

            # precompute full pairwise mean-cosine (n is small, a few hundred)
            n = len(ids)
            sim = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    s = mean_cosine(norms[ids[i]], norms[ids[j]])
                    sim[i, j] = sim[j, i] = s

            cross_better = 0
            margins = []
            for i in range(n):
                same_idx = [
                    j for j in range(n) if j != i and pref[ids[j]] == pref[ids[i]]
                ]
                cross_idx = [j for j in range(n) if pref[ids[j]] != pref[ids[i]]]
                if not same_idx or not cross_idx:
                    continue
                bs_j = max(same_idx, key=lambda j: sim[i, j])
                bc_j = max(cross_idx, key=lambda j: sim[i, j])
                best_same = sim[i, bs_j]
                best_cross = sim[i, bc_j]
                beats = bool(best_cross > best_same)
                cross_better += int(beats)
                margins.append(best_cross - best_same)
                nn_rows.append(
                    {
                        "dataset": dataset,
                        "request_id": ids[i],
                        "prefix_id": pref[ids[i]],
                        "analysis_window": window,
                        "best_same_request": ids[bs_j],
                        "best_same_similarity": float(best_same),
                        "best_cross_request": ids[bc_j],
                        "best_cross_prefix_id": pref[ids[bc_j]],
                        "best_cross_similarity": float(best_cross),
                        "cross_beats_same": beats,
                        "margin_cross_minus_same": float(best_cross - best_same),
                    }
                )
                # pair samples: best cross (label cross) + K random cross negs
                pair_rows.append(
                    {
                        "dataset": dataset, "analysis_window": window,
                        "request_id": ids[i], "other": ids[bc_j],
                        "pair_type": "best_cross", "similarity": float(best_cross),
                    }
                )
                for j in same_idx:
                    pair_rows.append(
                        {
                            "dataset": dataset, "analysis_window": window,
                            "request_id": ids[i], "other": ids[j],
                            "pair_type": "same", "similarity": float(sim[i, j]),
                        }
                    )
                negs = rng.choice(
                    cross_idx, size=min(args.neg_k, len(cross_idx)), replace=False
                )
                for j in negs:
                    pair_rows.append(
                        {
                            "dataset": dataset, "analysis_window": window,
                            "request_id": ids[i], "other": ids[int(j)],
                            "pair_type": "cross_random", "similarity": float(sim[i, j]),
                        }
                    )

            n_eval = len(margins)
            if n_eval == 0:
                continue
            margins_arr = np.asarray(margins)
            # threshold sweep over ALL cross-prefix pairs
            cross_pairs = [
                sim[i, j]
                for i in range(n)
                for j in range(i + 1, n)
                if pref[ids[i]] != pref[ids[j]]
            ]
            cross_pairs = np.asarray(cross_pairs) if cross_pairs else np.zeros(0)
            metrics_summary[f"{dataset}|{window}"] = {
                "dataset": dataset,
                "analysis_window": window,
                "num_requests_evaluated": int(n_eval),
                "cross_better_rate": float(cross_better / n_eval),
                "margin_mean": float(margins_arr.mean()),
                "margin_median": float(np.median(margins_arr)),
                "margin_p90": float(np.percentile(margins_arr, 90)),
                "num_cross_pairs": int(len(cross_pairs)),
                **{
                    f"cross_pairs_ge_{t}": int((cross_pairs >= t).sum())
                    for t in THRESHOLDS
                },
                **{
                    f"cross_pairs_ge_{t}_frac": float((cross_pairs >= t).mean())
                    if len(cross_pairs)
                    else 0.0
                    for t in THRESHOLDS
                },
            }

    nn_df = pd.DataFrame(nn_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nn_df.to_parquet(args.out, engine="pyarrow", compression="zstd", index=False)
    pd.DataFrame(pair_rows).to_parquet(
        args.pairs_out, engine="pyarrow", compression="zstd", index=False
    )
    write_json("outputs/metrics/exp2_metrics.json", metrics_summary)

    summ = pd.DataFrame(list(metrics_summary.values()))
    if len(summ):
        cols = [
            "dataset", "analysis_window", "num_requests_evaluated",
            "cross_better_rate", "margin_mean", "margin_median",
        ]
        print(summ[cols].to_string(index=False))
    print("EXP2_DONE_OK")


if __name__ == "__main__":
    main()
