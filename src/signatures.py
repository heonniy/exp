"""Expert-signature construction from decode router traces.

A signature for (request, analysis window) is a per-MoE-layer expert
activation histogram: signature[layer, expert] = number of times that expert
was selected (across all top-k picks of all decode tokens in the window).
We store the L2-normalized-per-layer frequency as the primary signature, plus
raw counts and router-weight mass for secondary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Named analysis windows over decode steps. (start, length) with length None
# meaning "to end". Sliding windows are generated separately.
NAMED_WINDOWS: dict[str, tuple[int, int | None]] = {
    "full_decode": (0, None),
    "first_16": (0, 16),
    "first_32": (0, 32),
    "first_64": (0, 64),
    "first_128": (0, 128),
}

SLIDING = {
    "window_16": 16,
    "window_32": 32,
}


@dataclass
class Signature:
    request_id: str
    prefix_id: str
    dataset: str
    window: str
    valid_tokens: int          # number of decode tokens actually in the window
    valid_window: bool         # window fully covered by generated length
    counts: np.ndarray         # [L, E] int
    weight_mass: np.ndarray    # [L, E] float


def _accumulate(
    decode: pd.DataFrame,
    layer_index: dict[int, int],
    num_layers: int,
    num_experts: int,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    wmass = np.zeros((num_layers, num_experts), dtype=np.float64)
    step_set = set(int(s) for s in steps)
    sub = decode[decode["decode_step"].isin(step_set)]
    for layer_id, grp in sub.groupby("layer_id"):
        li = layer_index[int(layer_id)]
        # flatten all top-k picks in the window for this layer
        ids_flat = np.concatenate(
            [np.asarray(x, dtype=np.int64) for x in grp["topk_expert_ids"]]
        )
        w_flat = np.concatenate(
            [np.asarray(x, dtype=np.float64) for x in grp["topk_router_weights"]]
        )
        counts[li] = np.bincount(ids_flat, minlength=num_experts)
        wmass[li] = np.bincount(ids_flat, weights=w_flat, minlength=num_experts)
    return counts, wmass


def build_signatures(
    decode: pd.DataFrame,
    moe_layer_ids: list[int],
    num_experts: int,
    windows: dict[str, tuple[int, int | None]] | None = None,
    sliding: dict[str, int] | None = None,
) -> list[Signature]:
    """Build all windowed signatures for a single request's decode trace."""
    windows = windows or NAMED_WINDOWS
    sliding = sliding or SLIDING
    layer_index = {lid: i for i, lid in enumerate(moe_layer_ids)}
    num_layers = len(moe_layer_ids)

    request_id = decode["request_id"].iloc[0]
    prefix_id = decode["prefix_id"].iloc[0]
    dataset = decode["dataset"].iloc[0]
    gen_len = int(decode["decode_step"].max()) + 1 if len(decode) else 0

    sigs: list[Signature] = []

    def emit(name: str, start: int, length: int | None):
        end = gen_len if length is None else start + length
        steps = np.arange(start, min(end, gen_len))
        valid_tokens = int(len(steps))
        valid_window = (length is None) or (gen_len >= start + length)
        if valid_tokens == 0:
            counts = np.zeros((num_layers, num_experts), dtype=np.int64)
            wmass = np.zeros((num_layers, num_experts), dtype=np.float64)
        else:
            counts, wmass = _accumulate(
                decode, layer_index, num_layers, num_experts, steps
            )
        sigs.append(
            Signature(request_id, prefix_id, dataset, name, valid_tokens,
                      valid_window, counts, wmass)
        )

    for name, (start, length) in windows.items():
        emit(name, start, length)

    for prefix, size in sliding.items():
        for start in range(0, gen_len, size):
            emit(f"{prefix}_step_{start}", start, size)

    return sigs


def normalized_freq(counts: np.ndarray) -> np.ndarray:
    """Row(layer)-normalized frequency; zero rows stay zero."""
    counts = counts.astype(np.float64)
    row = counts.sum(axis=-1, keepdims=True)
    row[row == 0] = 1.0
    return counts / row
