"""Forward helper that captures router logits WITHOUT the aux-loss OOM.

Qwen3MoeForCausalLM.forward calls load_balancing_loss_func whenever
output_router_logits=True -- it materializes an expert_mask over all tokens
(num_tokens x num_experts x top_k) and OOMs on long prefills (e.g. 33k-token
QMSum meetings), even though the result is discarded when no labels are given.

We instead call the *base* model (model.model), which returns router_logits
with no aux-loss computation, and apply lm_head only to the last position to
get the next-token logits cheaply.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def base_forward(model, input_ids, past=None):
    """Return (last_token_logits [1, vocab], router_logits tuple, past)."""
    base = model.model(
        input_ids=input_ids,
        past_key_values=past,
        use_cache=True,
        output_router_logits=True,
    )
    last_hidden = base.last_hidden_state[:, -1:, :]
    logits = model.lm_head(last_hidden)[:, -1, :]
    return logits, base.router_logits, base.past_key_values
