from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import OffloadedExpertEngine, load_offloaded_qwen
from experiments.runtime.residency_manager import StreamingRuntimeManager
from experiments.trace.collect_forced_routing_trace import _router_topk
from experiments.trace.trace_schema import RoutingTrace


def _read_row(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise ValueError(f"{path} does not contain row {index}")


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-resident and offloaded logits using a real 4K prefill "
            "cache and exact schema-v2 Expert ID plus routing-weight replay."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    trace = RoutingTrace.load(args.trace)
    trace.validate(config.model.num_experts_per_layer, require_weights=True)
    if not 0 <= args.request_index < trace.num_requests:
        raise ValueError("request index is outside the trace")
    if not 0 < args.decode_steps <= trace.output_tokens:
        raise ValueError("decode steps are outside the trace")
    row = _read_row(args.workload, args.request_index)
    if str(row["conversation_id"]) != str(trace.conversation_ids[args.request_index]):
        raise ValueError("trace and workload conversation IDs differ")
    forced = np.asarray(row["forced_output_ids"][: args.decode_steps], dtype=np.int32)
    if not np.array_equal(
        forced, trace.forced_output_ids[args.request_index, : args.decode_steps]
    ):
        raise ValueError("trace and workload forced tokens differ")

    dtype = getattr(torch, config.model.dtype)
    full_model = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    prompt = torch.as_tensor(
        row["input_ids"], dtype=torch.long, device="cuda:0"
    ).unsqueeze(0)
    prefill = full_model(
        input_ids=prompt,
        use_cache=True,
        output_router_logits=False,
        logits_to_keep=1,
        return_dict=True,
    )
    # The untouched copy is consumed by the offloaded decoder. It is a real
    # full-model 4K prefill cache, so static-zero KV cannot affect correctness.
    offloaded_past = copy.deepcopy(prefill.past_key_values)
    reference_past = prefill.past_key_values
    reference_logits: list[torch.Tensor] = []
    route_id_exact = []
    route_weight_exact = []
    for step in range(args.decode_steps):
        token = torch.as_tensor([[int(forced[step])]], device="cuda:0")
        output = full_model(
            input_ids=token,
            past_key_values=reference_past,
            use_cache=True,
            output_router_logits=True,
            logits_to_keep=1,
            return_dict=True,
        )
        reference_past = output.past_key_values
        ids, weights = _router_topk(
            output,
            config.model.num_moe_layers,
            config.model.router_top_k,
            1,
        )
        recorded_ids = trace.routing_expert_ids[
            args.request_index, step
        ][None, ...]
        recorded_weights = trace.require_routing_weights()[
            args.request_index, step
        ][None, ...]
        route_id_exact.append(bool(np.array_equal(ids, recorded_ids)))
        route_weight_exact.append(bool(np.array_equal(weights, recorded_weights)))
        reference_logits.append(output.logits.detach().to("cpu"))

    del output, prefill, reference_past, prompt, full_model
    torch.cuda.empty_cache()

    host_store = PinnedExpertStore(
        config.model.path, max_pinned_experts=16, pin_weights=False
    )
    engine = OffloadedExpertEngine(
        host_store,
        StreamingRuntimeManager(),
        prefetch_depth=1,
        track_timeline=False,
        prefetch_submit_order="compute_first",
    )
    offloaded_model = load_offloaded_qwen(config.model.path, engine).eval()
    replay_ids = trace.routing_expert_ids[
        args.request_index : args.request_index + 1, : args.decode_steps
    ]
    replay_weights = trace.require_routing_weights()[
        args.request_index : args.request_index + 1, : args.decode_steps
    ]
    engine.set_forced_routing(replay_ids, replay_weights)
    comparisons = []
    for step in range(args.decode_steps):
        engine.decode_step = step
        token = torch.as_tensor([[int(forced[step])]], device="cuda:0")
        output = offloaded_model(
            input_ids=token,
            past_key_values=offloaded_past,
            use_cache=True,
            output_router_logits=False,
            logits_to_keep=1,
            return_dict=True,
        )
        offloaded_past = output.past_key_values
        actual = output.logits.detach().to("cpu")
        expected = reference_logits[step]
        difference = (actual.float() - expected.float()).abs()
        comparisons.append(
            {
                "decode_step": step,
                "reference_logits_sha256": _digest(expected),
                "offloaded_logits_sha256": _digest(actual),
                "bitwise_equal": bool(torch.equal(actual, expected)),
                "allclose_atol_0p1_rtol_0p01": bool(
                    torch.allclose(actual, expected, atol=0.1, rtol=0.01)
                ),
                "max_absolute_error": float(difference.max().item()),
                "mean_absolute_error": float(difference.mean().item()),
                "argmax_equal": bool(
                    torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
                ),
            }
        )

    result = {
        "validation": "full_vs_offloaded_weighted_routing_replay",
        "gpu_physical_index": 0,
        "config": config.name,
        "request_index": args.request_index,
        "conversation_id": str(row["conversation_id"]),
        "input_tokens": len(row["input_ids"]),
        "decode_steps": args.decode_steps,
        "kv_setup": "real_full_model_prefill_cache",
        "trace_schema_version": trace.metadata.get("schema_version", 2),
        "forced_routing_weight_source": "recorded_trace_weights",
        "forced_routing_trace_sha256": trace.digest(),
        "reference_route_ids_exact_by_step": route_id_exact,
        "reference_route_weights_exact_by_step": route_weight_exact,
        "reference_routes_exact": all(route_id_exact),
        "reference_weights_exact": all(route_weight_exact),
        "logits_allclose_all_steps": all(
            row["allclose_atol_0p1_rtol_0p01"] for row in comparisons
        ),
        "argmax_equal_all_steps": all(row["argmax_equal"] for row in comparisons),
        "comparisons": comparisons,
        "engine_metrics": engine.metrics(),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
