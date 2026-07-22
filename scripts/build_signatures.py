#!/usr/bin/env python3
"""Build windowed expert signatures from per-request decode traces.

Reads outputs/traces/<dataset>/<prefix>/<request>.parquet, accumulates
per-window activation histograms, and writes:
    outputs/metrics/signatures.npz         (counts + weight_mass per req|window)
    outputs/metrics/signatures_meta.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model_loader import MODEL_DIR, extract_runtime_config  # noqa: E402
from src.signatures import NAMED_WINDOWS, SLIDING, build_signatures  # noqa: E402
from src.trace_io import read_request_trace  # noqa: E402
from src.utils import read_json  # noqa: E402


def get_runtime():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    return extract_runtime_config(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="outputs/traces")
    ap.add_argument("--datasets", default="squality,qmsum")
    ap.add_argument("--out-npz", default="outputs/metrics/signatures.npz")
    ap.add_argument("--out-meta", default="outputs/metrics/signatures_meta.parquet")
    ap.add_argument("--run-manifest", default="outputs/run_manifest.json")
    args = ap.parse_args()

    rc = get_runtime()
    # prefer moe_layer_ids recorded in the run manifest if present
    moe_layer_ids = rc.moe_layer_ids
    if Path(args.run_manifest).exists():
        rm = read_json(args.run_manifest)
        if rm.get("moe_layer_ids"):
            moe_layer_ids = rm["moe_layer_ids"]
    num_experts = rc.num_experts

    counts_store: dict[str, np.ndarray] = {}
    wmass_store: dict[str, np.ndarray] = {}
    meta_rows = []

    for dataset in args.datasets.split(","):
        droot = Path(args.trace_dir) / dataset
        if not droot.is_dir():
            print(f"[sig] skip missing {droot}")
            continue
        parquets = sorted(droot.glob("*/*.parquet"))
        print(f"[sig] {dataset}: {len(parquets)} request traces")
        for pq in parquets:
            df = read_request_trace(pq)
            decode = df[df["phase"] == "decode"]
            if len(decode) == 0:
                continue
            sigs = build_signatures(
                decode, moe_layer_ids, num_experts, NAMED_WINDOWS, SLIDING
            )
            for s in sigs:
                key = f"{s.request_id}||{s.window}"
                counts_store[key] = s.counts.astype(np.int32)
                wmass_store[key] = s.weight_mass.astype(np.float32)
                meta_rows.append(
                    {
                        "request_id": s.request_id,
                        "prefix_id": s.prefix_id,
                        "dataset": s.dataset,
                        "window": s.window,
                        "valid_tokens": s.valid_tokens,
                        "valid_window": s.valid_window,
                        "key": key,
                    }
                )

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    # store counts and weight-mass under prefixed keys in a single npz
    save_dict = {}
    for k, v in counts_store.items():
        save_dict[f"c::{k}"] = v
    for k, v in wmass_store.items():
        save_dict[f"w::{k}"] = v
    save_dict["__moe_layer_ids__"] = np.asarray(moe_layer_ids, dtype=np.int32)
    save_dict["__num_experts__"] = np.asarray([num_experts], dtype=np.int32)
    np.savez_compressed(args.out_npz, **save_dict)

    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(args.out_meta, engine="pyarrow", compression="zstd", index=False)
    print(
        f"[sig] wrote {len(counts_store)} signatures over "
        f"{meta['request_id'].nunique()} requests -> {args.out_npz}"
    )
    print("SIG_DONE_OK")


if __name__ == "__main__":
    main()
