from __future__ import annotations

import torch
from transformers import StaticCache


def make_static_kv_cache(
    model,
    *,
    batch_size: int,
    max_cache_length: int,
    initial_sequence_length: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda:0",
) -> StaticCache:
    if not 0 <= initial_sequence_length < max_cache_length:
        raise ValueError("initial sequence length must leave room for decode")
    cache = StaticCache(model.config, max_cache_len=max_cache_length)
    text_config = model.config.get_text_config(decoder=True)
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        dtype=dtype,
        device=torch.device(device),
    )
    # A zero-valued cache is a decode-only memory/performance fixture: KV
    # creation happens outside the timer, while every layer uses the real
    # attention implementation and the full allocated cache shape.
    for layer in cache.layers:
        if layer.is_initialized:
            layer.cumulative_length.fill_(initial_sequence_length)
    return cache
