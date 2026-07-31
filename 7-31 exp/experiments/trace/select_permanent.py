from __future__ import annotations

import numpy as np

from experiments.trace.trace_schema import RoutingTrace


PERMANENT_METHODS = {
    "presence",
    "token_frequency",
    "batch_step_union_presence",
    "streaming_reload",
}


def score_experts(
    trace: RoutingTrace,
    method: str = "presence",
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return a [layer, Expert] calibration score matrix.

    ``presence`` is retained as the legacy token-assignment-frequency baseline.
    ``batch_step_union_presence`` counts an Expert at most once for every
    (batch wave, decode step, layer), matching the unit at which one Expert H2D
    fetch serves all routed tokens in that layer-step.
    """

    if method not in PERMANENT_METHODS:
        raise ValueError(f"unknown permanent selection method: {method}")
    if method == "batch_step_union_presence" and (
        batch_size is None or batch_size <= 0
    ):
        raise ValueError(
            "batch_step_union_presence requires a positive batch_size"
        )
    scores = np.zeros(
        (trace.num_layers, int(trace.metadata.get("num_experts", 128))),
        dtype=np.int64,
    )
    if method == "batch_step_union_presence":
        assert batch_size is not None
        wave_ids = np.arange(trace.num_requests, dtype=np.int64) // batch_size
        step_ids = np.arange(trace.output_tokens, dtype=np.int64)
        num_waves = int(wave_ids[-1]) + 1 if len(wave_ids) else 0
        for layer_id in range(trace.num_layers):
            routes = trace.routing_expert_ids[:, :, layer_id, :]
            present = np.zeros(
                (num_waves, trace.output_tokens, scores.shape[1]),
                dtype=np.bool_,
            )
            present[
                np.broadcast_to(wave_ids[:, None, None], routes.shape),
                np.broadcast_to(step_ids[None, :, None], routes.shape),
                routes,
            ] = True
            scores[layer_id] = present.sum(axis=(0, 1), dtype=np.int64)
    else:
        for layer_id in range(trace.num_layers):
            scores[layer_id] = np.bincount(
                trace.routing_expert_ids[:, :, layer_id, :].reshape(-1),
                minlength=scores.shape[1],
            )
    if method == "streaming_reload":
        scores = np.maximum(scores - 1, 0)
    return scores


def select_topk(
    trace: RoutingTrace,
    k: int,
    method: str = "presence",
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    scores = score_experts(trace, method, batch_size=batch_size)
    return select_topk_from_scores(scores, k)


def select_topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be [layer, Expert]")
    if not 0 <= k <= scores.shape[1]:
        raise ValueError(f"k={k} is outside [0, {scores.shape[1]}]")
    selected = np.empty((scores.shape[0], k), dtype=np.uint8)
    expert_ids = np.arange(scores.shape[1])
    for layer_id in range(scores.shape[0]):
        # Primary key is descending score, deterministic tie-break is Expert ID.
        order = np.lexsort((expert_ids, -scores[layer_id]))
        selected[layer_id] = order[:k].astype(np.uint8)
    return selected
