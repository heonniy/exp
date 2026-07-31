from __future__ import annotations

import numpy as np

from experiments.trace.trace_schema import RoutingTrace


def score_experts(trace: RoutingTrace, method: str = "presence") -> np.ndarray:
    """Return a [layer, Expert] calibration score matrix.

    A trace row represents one request-token layer-step and top-k contains no
    duplicate Expert. Therefore presence and token-assignment frequency coincide
    for single-sequence traces. The explicit methods remain separate so batched
    trace formats can extend this without changing the selection API.
    """

    if method not in {"presence", "token_frequency", "streaming_reload"}:
        raise ValueError(f"unknown permanent selection method: {method}")
    scores = np.zeros(
        (trace.num_layers, int(trace.metadata.get("num_experts", 128))),
        dtype=np.int64,
    )
    for layer_id in range(trace.num_layers):
        scores[layer_id] = np.bincount(
            trace.routing_expert_ids[:, :, layer_id, :].reshape(-1),
            minlength=scores.shape[1],
        )
    if method == "streaming_reload":
        scores = np.maximum(scores - 1, 0)
    return scores


def select_topk(trace: RoutingTrace, k: int, method: str = "presence") -> np.ndarray:
    scores = score_experts(trace, method)
    if not 0 <= k <= scores.shape[1]:
        raise ValueError(f"k={k} is outside [0, {scores.shape[1]}]")
    selected = np.empty((trace.num_layers, k), dtype=np.uint8)
    expert_ids = np.arange(scores.shape[1])
    for layer_id in range(trace.num_layers):
        # Primary key is descending score, deterministic tie-break is Expert ID.
        order = np.lexsort((expert_ids, -scores[layer_id]))
        selected[layer_id] = order[:k].astype(np.uint8)
    return selected

