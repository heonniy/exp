#!/usr/bin/env python3
"""Capability probe: confirm the installed transformers loads Qwen3-MoE and
returns router logits in the expected structure. Run on GPU 6,7.

Prints a JSON summary and asserts the routing contract we rely on.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model_loader import (  # noqa: E402
    extract_runtime_config,
    first_param_device,
    load_model,
    load_tokenizer,
)
from src.prompts import encode_prompt  # noqa: E402
from src.router_trace import extract_step, extract_range  # noqa: E402


def main() -> None:
    print("[probe] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    import transformers

    print("[probe] transformers", transformers.__version__)

    tok = load_tokenizer()
    model = load_model()
    rc = extract_runtime_config(model.config)
    print("[probe] runtime config:", rc)

    dev = first_param_device(model)
    ids = encode_prompt(
        tok, "The cat sat on the mat. It was warm.", "Summarize.", enable_thinking=False
    )
    input_ids = torch.tensor([ids], device=dev)
    print("[probe] prompt tokens:", input_ids.shape[1])

    # --- Prefill forward with router logits ---
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_router_logits=True,
        )
    assert hasattr(out, "router_logits") and out.router_logits is not None, (
        "model output has no router_logits"
    )
    rl = out.router_logits
    print("[probe] num router-logit layers:", len(rl))
    print("[probe] one layer shape:", tuple(rl[0].shape))
    assert len(rl) == len(rc.moe_layer_ids), (
        f"router-logit layer count {len(rl)} != moe_layer_ids {len(rc.moe_layer_ids)}"
    )
    assert rl[0].shape[-1] == rc.num_experts, "expert dim mismatch"
    assert rl[0].shape[0] == input_ids.shape[1], "prefill token dim mismatch"

    # extract prefill range for first layer
    prange = extract_range(rl, rc.num_experts_per_tok, rc.norm_topk_prob, rc.num_experts)
    assert len(prange) == len(rc.moe_layer_ids)
    assert len(prange[0]) == input_ids.shape[1]
    ids0, w0 = prange[0][0]
    assert len(ids0) == rc.num_experts_per_tok, "top_k count mismatch"
    print("[probe] prefill layer0 tok0 top-k ids:", ids0)
    print("[probe] prefill layer0 tok0 weights sum:", round(sum(w0), 4))

    # --- One decode step, greedy ---
    logits = out.logits[:, -1, :]
    next_id = int(torch.argmax(logits, dim=-1))
    past = out.past_key_values
    with torch.no_grad():
        out2 = model(
            input_ids=torch.tensor([[next_id]], device=dev),
            past_key_values=past,
            use_cache=True,
            output_router_logits=True,
        )
    rl2 = out2.router_logits
    print("[probe] decode router layer shape:", tuple(rl2[0].shape))
    assert rl2[0].shape[0] == 1, "decode step should have 1 token row"
    step = extract_step(
        rl2, slice(-1, None), rc.num_experts_per_tok, rc.norm_topk_prob, rc.num_experts
    )
    assert len(step) == len(rc.moe_layer_ids)
    d_ids, d_w = step[0]
    assert len(d_ids) == rc.num_experts_per_tok
    print("[probe] decode layer0 top-k ids:", d_ids)

    summary = {
        "transformers_version": transformers.__version__,
        "num_hidden_layers": rc.num_hidden_layers,
        "num_experts": rc.num_experts,
        "num_experts_per_tok": rc.num_experts_per_tok,
        "norm_topk_prob": rc.norm_topk_prob,
        "num_moe_layers": len(rc.moe_layer_ids),
        "moe_layer_ids_head": rc.moe_layer_ids[:5],
        "moe_layer_ids_tail": rc.moe_layer_ids[-5:],
        "router_logits_ok": True,
        "next_token_id": next_id,
    }
    Path("logs").mkdir(exist_ok=True)
    with open("logs/probe_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[probe] PROBE_OK", json.dumps(summary))


if __name__ == "__main__":
    main()
