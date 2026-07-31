#!/usr/bin/env python3
"""Experiment 1: within-prefix expert-pattern similarity.

For every prefix group and every analysis window, compute all within-group
request pairs' per-layer cosine similarity summary + top-N Jaccard.
Writes outputs/metrics/exp1_within_prefix_pairs.parquet and an aggregate.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import cosine_summary, topn_jaccard  # noqa: E402
from src.sig_store import SignatureStore  # noqa: E402
from src.utils import write_json  # noqa: E402

PRIMARY_WINDOWS = ["first_64", "first_128", "full_decode"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/metrics/signatures.npz")
    ap.add_argument("--meta", default="outputs/metrics/signatures_meta.parquet")
    ap.add_argument("--windows", default=",".join(PRIMARY_WINDOWS))
    ap.add_argument("--out", default="outputs/metrics/exp1_within_prefix_pairs.parquet")
    ap.add_argument("--agg-out", default="outputs/metrics/exp1_aggregate.parquet")
    args = ap.parse_args()

    store = SignatureStore(args.npz, args.meta)
    windows = args.windows.split(",")

    rows = []
    for dataset in store.datasets():
        for window in windows:
            sub = store.subset(dataset, window)
            # only requests whose window is valid (fully covered)
            sub = sub[sub["valid_window"]]
            by_prefix = sub.groupby("prefix_id")
            for prefix_id, grp in by_prefix:
                members = grp.to_dict("records")
                for a, b in itertools.combinations(members, 2):
                    na = store.normalized(a["key"])
                    nb = store.normalized(b["key"])
                    cs = cosine_summary(na, nb)
                    ca = store.counts(a["key"])
                    cb = store.counts(b["key"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "prefix_id": prefix_id,
                            "request_a": a["request_id"],
                            "request_b": b["request_id"],
                            "analysis_window": window,
                            "valid_tokens_a": a["valid_tokens"],
                            "valid_tokens_b": b["valid_tokens"],
                            **cs,
                            "jaccard_top8_mean": topn_jaccard(ca, cb, 8),
                            "jaccard_top16_mean": topn_jaccard(ca, cb, 16),
                            "jaccard_top32_mean": topn_jaccard(ca, cb, 32),
                        }
                    )

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", compression="zstd", index=False)

    agg_rows = []
    for (dataset, window), grp in df.groupby(["dataset", "analysis_window"]):
        c = grp["cosine_mean_layers"].to_numpy()
        agg_rows.append(
            {
                "dataset": dataset,
                "analysis_window": window,
                "pair_count": int(len(grp)),
                "cosine_mean": float(np.mean(c)),
                "cosine_median": float(np.median(c)),
                "cosine_std": float(np.std(c)),
                "cosine_p10": float(np.percentile(c, 10)),
                "cosine_p25": float(np.percentile(c, 25)),
                "cosine_p75": float(np.percentile(c, 75)),
                "cosine_p90": float(np.percentile(c, 90)),
                "jaccard_top8_mean": float(grp["jaccard_top8_mean"].mean()),
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_parquet(args.agg_out, engine="pyarrow", compression="zstd", index=False)

    write_json(
        "outputs/metrics/exp1_aggregate.json",
        {r["dataset"] + "|" + r["analysis_window"]: r for r in agg_rows},
    )
    print(agg.to_string(index=False))
    print("EXP1_DONE_OK")


if __name__ == "__main__":
    main()
