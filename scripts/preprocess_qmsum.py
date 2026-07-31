#!/usr/bin/env python3
"""Preprocess QMSum test split -> common request jsonl + manifest."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.qmsum import iter_requests  # noqa: E402
from src.utils import write_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/qmsum")
    ap.add_argument("--split", default="test")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--output", default="data/processed/qmsum_requests.jsonl")
    ap.add_argument("--manifest", default="data/manifests/qmsum_manifest.json")
    args = ap.parse_args()

    requests = []
    excluded = []
    for item in iter_requests(args.raw_dir, args.split, args.group_size):
        if item.get("_status") == "ok":
            item.pop("_status", None)
            requests.append(item)
        else:
            excluded.append({"prefix_id": item["prefix_id"], "reason": item["_reason"]})

    n = write_jsonl(args.output, requests)

    groups = defaultdict(list)
    for r in requests:
        groups[r["prefix_id"]].append(r)
    group_sizes = Counter(len(v) for v in groups.values())

    manifest = {
        "dataset": "qmsum",
        "split": args.split,
        "raw_dir": args.raw_dir,
        "group_size": args.group_size,
        "num_requests": n,
        "num_prefix_groups": len(groups),
        "group_size_distribution": dict(group_sizes),
        "num_excluded_groups": len(excluded),
        "excluded": excluded,
    }
    write_json(args.manifest, manifest)
    print(
        f"[qmsum] wrote {n} requests, {len(groups)} groups "
        f"(sizes={dict(group_sizes)}), excluded={len(excluded)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
