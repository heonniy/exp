"""Parquet / npz IO for router traces and generations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACE_COLUMNS = [
    "dataset",
    "split",
    "prefix_id",
    "request_id",
    "phase",           # "prefill_suffix" | "decode"
    "token_position",  # absolute position in the full sequence
    "decode_step",     # int for decode, -1 for prefill_suffix
    "token_id",
    "layer_id",
    "topk_expert_ids",     # list<int>
    "topk_router_weights",  # list<float>
    "generated_length",
    "hit_max_new_tokens",
]


def write_request_trace(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def write_shared_prefill_agg(
    path: str | Path,
    counts: np.ndarray,       # [num_moe_layers, num_experts] int
    weight_mass: np.ndarray,  # [num_moe_layers, num_experts] float
    moe_layer_ids: list[int],
    num_tokens: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=counts.astype(np.int32),
        weight_mass=weight_mass.astype(np.float32),
        moe_layer_ids=np.asarray(moe_layer_ids, dtype=np.int32),
        num_tokens=np.asarray([num_tokens], dtype=np.int64),
    )


def read_request_trace(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")
