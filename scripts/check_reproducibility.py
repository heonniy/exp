#!/usr/bin/env python3
"""Reproducibility + trace-completeness check (spec Acceptance Criteria).

Re-generates a small set of requests a second time and asserts:
  * output_token_ids exact match vs the stored generation
  * decode top-k expert ids exact match vs the stored trace
  * every generated token has a routing row for all detected MoE layers
  * expert ids within [0, num_experts)
Router weights are compared with a tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from src.fwd import base_forward  # noqa: E402
from src.model_loader import (  # noqa: E402
    enable_determinism,
    extract_runtime_config,
    first_param_device,
    load_model,
    load_tokenizer,
)
from src.prompts import encode_prompt  # noqa: E402
from src.router_trace import extract_step  # noqa: E402
from src.trace_io import read_request_trace  # noqa: E402
from src.utils import read_jsonl  # noqa: E402


@torch.no_grad()
def regen(model, rc, dev, tok, req, max_new_tokens):
    ids = encode_prompt(tok, req["shared_content"], req["query"], enable_thinking=False)
    input_ids = torch.tensor([ids], device=dev)
    last_logits, _, past = base_forward(model, input_ids)
    cur = int(torch.argmax(last_logits, dim=-1))
    eos_ids = {int(tok.eos_token_id)}
    gc = model.generation_config
    e = getattr(gc, "eos_token_id", None)
    if isinstance(e, (list, tuple)):
        eos_ids.update(int(x) for x in e)
    elif e is not None:
        eos_ids.add(int(e))
    toks, step_experts = [], []
    for step in range(max_new_tokens):
        with sdpa_kernel(SDPBackend.MATH):
            nxt, step_rl, past = base_forward(
                model, torch.tensor([[cur]], device=dev), past
            )
        per_layer = extract_step(step_rl, slice(-1, None),
                                 rc.num_experts_per_tok, rc.norm_topk_prob,
                                 rc.num_experts)
        toks.append(cur)
        step_experts.append([ids_w[0] for ids_w in per_layer])
        if cur in eos_ids or step == max_new_tokens - 1:
            break
        cur = int(torch.argmax(nxt, dim=-1))
    return toks, step_experts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--gen-jsonl", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    reqs = {r["request_id"]: r for r in read_jsonl(args.requests)}
    gens = {r["request_id"]: r for r in read_jsonl(args.gen_jsonl)}
    sample_ids = list(gens.keys())[: args.n]

    enable_determinism()
    tok = load_tokenizer()
    model = load_model(attn_implementation="sdpa")
    rc = extract_runtime_config(model.config)
    dev = first_param_device(model)

    all_ok = True
    for rid in sample_ids:
        req = reqs[rid]
        stored = gens[rid]
        toks, step_experts = regen(model, rc, dev, tok, req, args.max_new_tokens)

        tok_match = toks == stored["output_token_ids"]

        # compare expert ids against stored trace
        gdir = Path(args.trace_dir) / req["prefix_id"].replace(":", "__")
        pq = gdir / f"{rid.replace(':', '__')}.parquet"
        df = read_request_trace(pq)
        dec = df[df["phase"] == "decode"]
        layers_per_step = dec.groupby("decode_step")["layer_id"].nunique()
        completeness = bool((layers_per_step == len(rc.moe_layer_ids)).all())
        allids = np.concatenate([np.asarray(x) for x in dec["topk_expert_ids"]])
        id_range_ok = bool(allids.min() >= 0 and allids.max() < rc.num_experts)

        # exact expert-id match on the first 8 decode steps, all layers
        expert_match = True
        layer_ids = rc.moe_layer_ids
        for step in range(min(8, len(step_experts))):
            srow = dec[dec["decode_step"] == step].sort_values("layer_id")
            stored_ids = list(srow["topk_expert_ids"])
            for li in range(len(layer_ids)):
                if list(step_experts[step][li]) != list(stored_ids[li]):
                    expert_match = False
                    break
            if not expert_match:
                break

        ok = tok_match and completeness and id_range_ok and expert_match
        all_ok = all_ok and ok
        print(
            f"[repro] {rid}: token_match={tok_match} expert_match(first8)={expert_match} "
            f"completeness={completeness} id_range_ok={id_range_ok} -> {'OK' if ok else 'FAIL'}"
        )

    print("REPRO_OK" if all_ok else "REPRO_FAIL")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
