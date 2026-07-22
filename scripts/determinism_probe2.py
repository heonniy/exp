#!/usr/bin/env python3
"""Test DECODE-path determinism and full in-process greedy reproduction."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model_loader import (enable_determinism, extract_runtime_config,  # noqa: E402
                              first_param_device, load_model, load_tokenizer)
from src.prompts import encode_prompt  # noqa: E402
from src.router_trace import extract_step  # noqa: E402
from src.utils import read_jsonl  # noqa: E402


from contextlib import nullcontext  # noqa: E402

from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402


@torch.no_grad()
def greedy(model, rc, dev, ids, n=32, decode_math=True):
    # prefill: default (memory-efficient) SDPA
    out = model(input_ids=torch.tensor([ids], device=dev), logits_to_keep=1,
                use_cache=True, output_router_logits=True)
    past = out.past_key_values
    cur = int(out.logits[:, -1, :].argmax())
    toks, experts, decode_logits = [], [], []
    for step in range(n):
        dctx = sdpa_kernel(SDPBackend.MATH) if decode_math else nullcontext()
        with dctx:
            so = model(input_ids=torch.tensor([[cur]], device=dev), past_key_values=past,
                       logits_to_keep=1, use_cache=True, output_router_logits=True)
        per_layer = extract_step(so.router_logits, slice(-1, None),
                                 rc.num_experts_per_tok, rc.norm_topk_prob, rc.num_experts)
        toks.append(cur)
        experts.append([tuple(p[0]) for p in per_layer])
        decode_logits.append(so.logits[:, -1, :].float().cpu())
        past = so.past_key_values
        cur = int(so.logits[:, -1, :].argmax())
    return toks, experts, decode_logits


def main():
    import os
    req = next(iter(read_jsonl("data/processed/squality_requests.jsonl")))
    tok = load_tokenizer()
    attn = os.environ.get("ATTN_IMPL", "sdpa")
    decode_math = os.environ.get("DECODE_MATH", "1") == "1"
    enable_determinism()
    print("attn_implementation:", attn, "decode_math:", decode_math)
    model = load_model(attn_implementation=attn)
    rc = extract_runtime_config(model.config)
    dev = first_param_device(model)
    ids = encode_prompt(tok, req["shared_content"], req["query"], enable_thinking=False)
    print("prompt tokens:", len(ids))

    t1, e1, dl1 = greedy(model, rc, dev, ids, n=32, decode_math=decode_math)
    t2, e2, dl2 = greedy(model, rc, dev, ids, n=32, decode_math=decode_math)

    tok_match = t1 == t2
    first_div = next((i for i in range(len(t1)) if t1[i] != t2[i]), None)
    # per-step decode logits bit equality
    logit_bit_equal = [bool(torch.equal(a, b)) for a, b in zip(dl1, dl2)]
    first_logit_div = next((i for i, x in enumerate(logit_bit_equal) if not x), None)
    max_diffs = [float((a - b).abs().max()) for a, b in zip(dl1, dl2)]
    expert_mismatch_steps = sum(1 for a, b in zip(e1, e2) if a != b)
    print("decode token_match:", tok_match, "first_token_divergence:", first_div)
    print("first decode-logit bit divergence at step:", first_logit_div)
    print("max abs logit diff over steps: max=%.3e" % max(max_diffs))
    print("expert-set mismatched steps:", expert_mismatch_steps, "/", len(e1))
    print("DETPROBE2_DONE_OK")


if __name__ == "__main__":
    main()
