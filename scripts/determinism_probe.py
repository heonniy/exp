#!/usr/bin/env python3
"""Isolate forward-pass determinism: run the SAME input twice and compare
next-token logits and router top-k exactly, under a few determinism settings.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model_loader import (extract_runtime_config, first_param_device,  # noqa: E402
                              load_model, load_tokenizer)
from src.prompts import encode_prompt  # noqa: E402
from src.router_trace import extract_step  # noqa: E402
from src.utils import read_jsonl  # noqa: E402


def one_forward(model, dev, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev),
                    logits_to_keep=1, use_cache=True, output_router_logits=True)
    return out.logits[:, -1, :].float().cpu(), out.router_logits


def compare(tag, model, rc, dev, ids):
    l1, rl1 = one_forward(model, dev, ids)
    l2, rl2 = one_forward(model, dev, ids)
    logits_equal = bool(torch.equal(l1, l2))
    max_abs = float((l1 - l2).abs().max())
    argmax_equal = int(l1.argmax()) == int(l2.argmax())
    e1 = extract_step(rl1, slice(-1, None), rc.num_experts_per_tok,
                      rc.norm_topk_prob, rc.num_experts)
    e2 = extract_step(rl2, slice(-1, None), rc.num_experts_per_tok,
                      rc.norm_topk_prob, rc.num_experts)
    layer_mismatch = sum(1 for a, b in zip(e1, e2) if list(a[0]) != list(b[0]))
    print(f"[{tag}] logits_bit_equal={logits_equal} max_abs_diff={max_abs:.3e} "
          f"argmax_equal={argmax_equal} router_layer_mismatch={layer_mismatch}/{len(e1)}")
    return logits_equal, layer_mismatch


def main():
    req = next(iter(read_jsonl("data/processed/squality_requests.jsonl")))
    tok = load_tokenizer()

    # setting A: default
    model = load_model()
    rc = extract_runtime_config(model.config)
    dev = first_param_device(model)
    ids = encode_prompt(tok, req["shared_content"], req["query"], enable_thinking=False)
    print("prompt tokens:", len(ids))
    compare("default", model, rc, dev, ids)

    # setting B: disable TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    compare("no_tf32", model, rc, dev, ids)

    print("DETPROBE_DONE_OK")


if __name__ == "__main__":
    main()
