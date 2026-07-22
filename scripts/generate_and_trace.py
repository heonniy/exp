#!/usr/bin/env python3
"""Deterministic (greedy) generation with single-pass MoE router-trace capture.

For each request we run one autoregressive pass and, at every decode step,
capture the top-k expert routing of the token being processed (same forward
that produces the next logits). Prefill routing over the shared prefix is
aggregated once per group; the query-suffix prefill routing is stored per
request token-level alongside the decode trace.

Storage layout (per dataset root, e.g. outputs/traces/squality/):
    <prefix_id>/shared_prefill_agg.npz     (once per group)
    <prefix_id>/<request_id>.parquet       (query-suffix prefill + decode)
Generations are appended to outputs/generations/<dataset>/generations.jsonl.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
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
from src.router_trace import extract_range, extract_step  # noqa: E402
from src.trace_io import write_request_trace, write_shared_prefill_agg  # noqa: E402
from src.utils import append_jsonl, read_jsonl, sha256_ids  # noqa: E402


def load_enriched(dataset: str) -> dict[str, dict]:
    path = Path(f"data/manifests/{dataset}_enriched.jsonl")
    out = {}
    for row in read_jsonl(path):
        out[row["request_id"]] = row
    return out


@torch.no_grad()
def aggregate_prefill(
    router_logits, start: int, end: int, top_k, norm_topk_prob, num_experts
):
    """Aggregate top-k activation counts + weight mass over positions [start,end)
    for every MoE layer. Returns (counts[L,E] int, weight_mass[L,E] float)."""
    L = len(router_logits)
    counts = np.zeros((L, num_experts), dtype=np.int64)
    wmass = np.zeros((L, num_experts), dtype=np.float64)
    for li, layer_logits in enumerate(router_logits):
        sel = layer_logits[start:end]  # [n, E]
        probs = torch.softmax(sel.float(), dim=-1)
        w, ids = torch.topk(probs, top_k, dim=-1)
        if norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        # counts + weight-mass via GPU scatter_add, then a single copy to host
        flat_ids = ids.reshape(-1)
        cnt = torch.zeros(num_experts, device=ids.device, dtype=torch.float32)
        cnt.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float32))
        wm = torch.zeros(num_experts, device=ids.device, dtype=torch.float32)
        wm.scatter_add_(0, flat_ids, w.reshape(-1).float())
        counts[li] = cnt.to("cpu").numpy().astype(np.int64)
        wmass[li] = wm.to("cpu").numpy()
    return counts, wmass


@torch.no_grad()
def run_request(model, rc, dev, req, enriched, dataset_root, gen_jsonl, args, tok):
    request_id = req["request_id"]
    prefix_id = req["prefix_id"]
    dataset = req["dataset"]
    split = req["split"]
    shared_len = enriched[request_id]["shared_prefix_token_count"]

    ids = encode_prompt(tok, req["shared_content"], req["query"], enable_thinking=False)
    n_prompt = len(ids)
    input_ids = torch.tensor([ids], device=dev)

    # --- Prefill --- (base model, no aux-loss OOM; last-token logits only)
    last_logits, prefill_rl, past = base_forward(model, input_ids)

    rows: list[dict] = []
    # shared-prefix aggregate (store once per group -> first request writes it)
    group_dir = Path(dataset_root) / prefix_id.replace(":", "__")
    shared_npz = group_dir / "shared_prefill_agg.npz"
    if not shared_npz.exists():
        counts, wmass = aggregate_prefill(
            prefill_rl, 0, shared_len, rc.num_experts_per_tok,
            rc.norm_topk_prob, rc.num_experts,
        )
        write_shared_prefill_agg(
            shared_npz, counts, wmass, rc.moe_layer_ids, shared_len
        )

    # query-suffix prefill routing, token-level (only the suffix positions)
    suffix_logits = [rl[shared_len:n_prompt] for rl in prefill_rl]
    suffix_range = extract_range(
        suffix_logits, rc.num_experts_per_tok, rc.norm_topk_prob, rc.num_experts,
    )
    for li, layer_id in enumerate(rc.moe_layer_ids):
        layer_rows = suffix_range[li]
        for j, pos in enumerate(range(shared_len, n_prompt)):
            ids_w = layer_rows[j]
            rows.append(
                {
                    "dataset": dataset, "split": split, "prefix_id": prefix_id,
                    "request_id": request_id, "phase": "prefill_suffix",
                    "token_position": pos, "decode_step": -1,
                    "token_id": ids[pos], "layer_id": layer_id,
                    "topk_expert_ids": ids_w[0], "topk_router_weights": ids_w[1],
                    "generated_length": 0, "hit_max_new_tokens": False,
                }
            )

    # free prefill router tensors early (last_logits + past came from base_forward)
    del prefill_rl, suffix_range, suffix_logits

    # --- Decode (greedy, single pass, capture routing of each generated token) ---
    eos_ids = set()
    gc = model.generation_config
    gc_eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(gc_eos, (list, tuple)):
        eos_ids.update(int(x) for x in gc_eos)
    elif gc_eos is not None:
        eos_ids.add(int(gc_eos))
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))

    cur = int(torch.argmax(last_logits, dim=-1))
    generated: list[int] = []
    decode_rows_tmp: list[tuple[int, int, list]] = []  # (step, token_id, per-layer)
    eos_reached = False
    for step in range(args.max_new_tokens):
        # deterministic decode attention (MATH backend); prefill above used the
        # memory-efficient default SDPA path.
        with sdpa_kernel(SDPBackend.MATH):
            nxt_logits, step_rl, past = base_forward(
                model, torch.tensor([[cur]], device=dev), past
            )
        per_layer = extract_step(
            step_rl, slice(-1, None), rc.num_experts_per_tok,
            rc.norm_topk_prob, rc.num_experts,
        )
        generated.append(cur)
        decode_rows_tmp.append((step, cur, per_layer))
        del step_rl
        if cur in eos_ids:
            eos_reached = True
            break
        if step == args.max_new_tokens - 1:
            break
        cur = int(torch.argmax(nxt_logits, dim=-1))

    generated_length = len(generated)
    hit_max = (generated_length == args.max_new_tokens) and not eos_reached

    for step, token_id, per_layer in decode_rows_tmp:
        abs_pos = n_prompt + step
        for li, layer_id in enumerate(rc.moe_layer_ids):
            ids_w = per_layer[li]
            rows.append(
                {
                    "dataset": dataset, "split": split, "prefix_id": prefix_id,
                    "request_id": request_id, "phase": "decode",
                    "token_position": abs_pos, "decode_step": step,
                    "token_id": token_id, "layer_id": layer_id,
                    "topk_expert_ids": ids_w[0], "topk_router_weights": ids_w[1],
                    "generated_length": generated_length,
                    "hit_max_new_tokens": hit_max,
                }
            )

    trace_path = group_dir / f"{request_id.replace(':', '__')}.parquet"
    write_request_trace(trace_path, rows)

    gen_text = tok.decode(generated, skip_special_tokens=True)
    append_jsonl(
        gen_jsonl,
        {
            "request_id": request_id, "prefix_id": prefix_id,
            "dataset": dataset, "split": split,
            "output_token_ids": generated,
            "generated_text": gen_text,
            "input_token_count": n_prompt,
            "generated_token_count": generated_length,
            "eos_reached": eos_reached,
            "hit_max_new_tokens": hit_max,
            "prompt_sha256": sha256_ids(ids),
            "shared_prefix_sha256": sha256_ids(ids[:shared_len]),
            "generation_config": {
                "do_sample": False, "num_beams": 1,
                "max_new_tokens": args.max_new_tokens, "enable_thinking": False,
            },
        },
    )
    del past, last_logits
    return {
        "request_id": request_id, "generated_length": generated_length,
        "eos_reached": eos_reached, "hit_max": hit_max, "n_prompt": n_prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--output-dir", required=True, help="e.g. outputs/traces/squality")
    ap.add_argument("--gen-jsonl", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--disable-thinking", action="store_true", default=True)
    ap.add_argument("--limit-groups", type=int, default=0, help="0 = all")
    ap.add_argument("--only-prefix", default=None, help="comma list of prefix_ids")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    requests = list(read_jsonl(args.requests))
    dataset = requests[0]["dataset"]
    enriched = load_enriched(dataset)

    # drop context-overflow requests
    requests = [
        r for r in requests
        if not enriched.get(r["request_id"], {}).get("excluded_context_overflow", False)
    ]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in requests:
        groups[r["prefix_id"]].append(r)
    for g in groups.values():
        g.sort(key=lambda m: m["request_index_in_group"])

    prefix_ids = sorted(groups.keys())
    if args.only_prefix:
        wanted = set(args.only_prefix.split(","))
        prefix_ids = [p for p in prefix_ids if p in wanted]
    if args.limit_groups > 0:
        prefix_ids = prefix_ids[: args.limit_groups]

    gen_jsonl = args.gen_jsonl or f"outputs/generations/{dataset}/generations.jsonl"
    if not args.skip_existing and Path(gen_jsonl).exists():
        Path(gen_jsonl).unlink()

    enable_determinism()
    tok = load_tokenizer()
    model = load_model(attn_implementation="sdpa")
    rc = extract_runtime_config(model.config)
    dev = first_param_device(model)
    print(f"[gen:{dataset}] runtime={rc}", flush=True)
    print(f"[gen:{dataset}] {len(prefix_ids)} groups to process", flush=True)

    torch.manual_seed(0)
    t0 = time.time()
    done = 0
    for pi, prefix_id in enumerate(prefix_ids):
        for req in groups[prefix_id]:
            trace_path = (
                Path(args.output_dir)
                / prefix_id.replace(":", "__")
                / f"{req['request_id'].replace(':', '__')}.parquet"
            )
            if args.skip_existing and trace_path.exists():
                done += 1
                continue
            r = run_request(
                model, rc, dev, req, enriched, args.output_dir, gen_jsonl, args, tok
            )
            done += 1
            elapsed = time.time() - t0
            print(
                f"[gen:{dataset}] {done} {r['request_id']} "
                f"prompt={r['n_prompt']} gen={r['generated_length']} "
                f"eos={r['eos_reached']} hitmax={r['hit_max']} "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )
    print(f"[gen:{dataset}] GEN_DONE_OK processed={done} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
