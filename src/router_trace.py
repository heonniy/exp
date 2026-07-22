"""MoE router-trace extraction for Qwen3-MoE.

We reconstruct the actual routing decision the model makes inside
``Qwen3MoeSparseMoeBlock``::

    routing_weights = softmax(router_logits, dim=-1)          # over all experts
    topk_weights, topk_ids = topk(routing_weights, top_k)
    if norm_topk_prob:
        topk_weights /= topk_weights.sum(-1, keepdim=True)    # renormalize

So the stored ``topk_router_weights`` are the *renormalized* weights actually
used to combine the selected experts.
"""

from __future__ import annotations

from typing import Sequence

import torch


def moe_layer_ids_from_config(config) -> list[int]:
    """Return the decoder layer indices that use a sparse MoE block.

    Mirrors Qwen3-MoE's own rule rather than assuming 0..num_layers-1.
    """
    num_layers = config.num_hidden_layers
    sparse_step = getattr(config, "decoder_sparse_step", 1)
    mlp_only = set(getattr(config, "mlp_only_layers", []) or [])
    ids = []
    for layer_idx in range(num_layers):
        is_moe = (layer_idx not in mlp_only) and (
            sparse_step == 0 or (layer_idx + 1) % sparse_step == 0
        )
        if is_moe:
            ids.append(layer_idx)
    return ids


@torch.no_grad()
def topk_from_logits(
    logits: torch.Tensor, top_k: int, norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: [N, E] -> (topk_ids [N, k] int, topk_weights [N, k] float)."""
    probs = torch.softmax(logits.float(), dim=-1)
    weights, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights


@torch.no_grad()
def extract_step(
    router_logits: Sequence[torch.Tensor],
    positions: slice | None,
    top_k: int,
    norm_topk_prob: bool,
    num_experts: int,
) -> list[tuple[list[int], list[float]]]:
    """Extract top-k for a *single* token position from every MoE layer.

    ``router_logits`` is the per-forward tuple (one [num_tokens, E] tensor per
    MoE layer). ``positions`` selects which rows to keep; for decode we pass a
    slice of the last row. Returns a list (len = num_moe_layers) of
    (expert_ids, weights) for that one position.
    """
    out: list[tuple[list[int], list[float]]] = []
    for layer_logits in router_logits:
        sel = layer_logits if positions is None else layer_logits[positions]
        # sel: [1, E]
        ids, weights = topk_from_logits(sel, top_k, norm_topk_prob)
        assert int(ids.max()) < num_experts and int(ids.min()) >= 0, (
            "expert id out of range"
        )
        out.append((ids[0].tolist(), weights[0].tolist()))
    return out


@torch.no_grad()
def extract_range(
    router_logits: Sequence[torch.Tensor],
    top_k: int,
    norm_topk_prob: bool,
    num_experts: int,
) -> list[list[tuple[list[int], list[float]]]]:
    """Extract top-k for *all* token positions in a (prefill) forward.

    Returns per-layer list of per-token (expert_ids, weights):
        result[layer_index][token_position] = (ids, weights)
    """
    out: list[list[tuple[list[int], list[float]]]] = []
    for layer_logits in router_logits:
        ids, weights = topk_from_logits(layer_logits, top_k, norm_topk_prob)
        assert int(ids.max()) < num_experts and int(ids.min()) >= 0, (
            "expert id out of range"
        )
        layer_rows = [
            (ids[t].tolist(), weights[t].tolist()) for t in range(ids.shape[0])
        ]
        out.append(layer_rows)
    return out
