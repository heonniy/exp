from __future__ import annotations

import argparse
import hashlib
import json
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch

from experiments.benchmark.run_offloaded_decode import _manager, _read_examples
from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.kv_cache import make_static_kv_cache
from experiments.runtime.offloaded_model import (
    OffloadedExpertEngine,
    attach_engine,
    load_offloaded_qwen,
)
from experiments.runtime.residency_manager import StreamingRuntimeManager
from experiments.trace.trace_schema import RoutingTrace


SUM_METRICS = (
    "expert_h2d_fetches",
    "expert_h2d_bytes",
    "expert_executions",
    "permanent_hits",
    "quota_hits",
    "quota_misses",
    "quota_evictions",
    "logical_ownership_swaps",
    "d2d_admission_copies",
    "natural_route_assignments",
    "natural_route_mismatches",
)


def _detach_engine(model) -> None:
    for layer in model.model.layers:
        object.__setattr__(layer.mlp.experts, "_engine", None)


def _metric_delta(before: dict, after: dict, name: str) -> int:
    return int(after.get(name, 0)) - int(before.get(name, 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run every evaluation request in exact full/partial waves while "
            "retaining one model and, by default, warm quota state."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path)
    parser.add_argument("--forced-routing-trace", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=["stream2", "permanent_k", "quota_lru_k", "full_resident"],
        required=True,
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--cold-each-wave", action="store_true")
    parser.add_argument("--prefetch-depth", choices=[0, 1], type=int, default=1)
    parser.add_argument("--max-pinned-experts", type=int, default=6144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    trace = RoutingTrace.load(args.forced_routing_trace)
    requests = args.requests or trace.num_requests
    if not 0 < requests <= trace.num_requests:
        raise ValueError("requests is outside the forced trace")
    steps = args.decode_steps or config.dataset.output_tokens
    if not 0 < steps <= trace.output_tokens:
        raise ValueError("decode steps exceed the forced trace")
    examples = _read_examples(args.workload, requests)
    workload_ids = [str(row["conversation_id"]) for row in examples]
    trace_ids = [str(value) for value in trace.conversation_ids[:requests]]
    if workload_ids != trace_ids:
        raise ValueError("forced routing trace does not match workload row order")
    calibration = (
        RoutingTrace.load(args.calibration_trace)
        if args.calibration_trace is not None
        else None
    )

    host_store = PinnedExpertStore(
        config.model.path,
        max_pinned_experts=args.max_pinned_experts,
        pin_weights=True,
    )
    host_preload_seconds = host_store.preload_all(
        config.model.num_moe_layers,
        config.model.num_experts_per_layer,
    )
    bootstrap = OffloadedExpertEngine(host_store, StreamingRuntimeManager())
    model_load_started = time.perf_counter()
    model = load_offloaded_qwen(config.model.path, bootstrap)
    model_load_seconds = time.perf_counter() - model_load_started
    _detach_engine(model)
    del bootstrap
    torch.cuda.empty_cache()

    policy_initialization_seconds = 0.0
    manager = engine = None

    def initialize_policy() -> None:
        nonlocal manager, engine, policy_initialization_seconds
        _detach_engine(model)
        previous_manager, previous_engine = manager, engine
        manager = engine = None
        del previous_manager, previous_engine
        torch.cuda.empty_cache()
        manager, elapsed = _manager(
            args.policy,
            args.k,
            config.model.num_moe_layers,
            config.model.num_experts_per_layer,
            host_store,
            calibration,
        )
        policy_initialization_seconds += elapsed
        engine = OffloadedExpertEngine(
            host_store,
            manager,
            prefetch_depth=args.prefetch_depth,
            track_timeline=False,
        )
        attach_engine(model, engine)

    initialize_policy()
    totals = {name: 0 for name in SUM_METRICS}
    waves = []
    kv_setup_seconds = 0.0
    full_wave_seconds = []
    logits_digests = []
    for wave_index, start in enumerate(range(0, requests, args.batch_size)):
        if wave_index and args.cold_each_wave and args.policy == "quota_lru_k":
            initialize_policy()
        stop = min(start + args.batch_size, requests)
        wave_batch = stop - start
        kv_started = time.perf_counter()
        past = make_static_kv_cache(
            model,
            batch_size=wave_batch,
            max_cache_length=config.peak_sequence_length,
            initial_sequence_length=config.dataset.input_tokens,
        )
        torch.cuda.synchronize()
        wave_kv_setup = time.perf_counter() - kv_started
        kv_setup_seconds += wave_kv_setup
        forced_tokens = np.asarray(
            [row["forced_output_ids"][:steps] for row in examples[start:stop]],
            dtype=np.int64,
        )
        forced_routing = trace.routing_expert_ids[start:stop, :steps]
        engine.set_forced_routing(forced_routing)
        before = engine.metrics()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(steps):
                engine.decode_step = step
                token = torch.as_tensor(
                    forced_tokens[:, step, None],
                    dtype=torch.long,
                    device="cuda:0",
                )
                output = model(
                    input_ids=token,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    output_router_logits=False,
                    return_dict=True,
                )
                past = output.past_key_values
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        after = engine.metrics()
        for name in SUM_METRICS:
            totals[name] += _metric_delta(before, after, name)
        digest = hashlib.sha256(
            output.logits.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        ).hexdigest()
        logits_digests.append(digest)
        if wave_batch == args.batch_size:
            full_wave_seconds.append(elapsed)
        waves.append(
            {
                "wave_index": wave_index,
                "start": start,
                "stop": stop,
                "batch_size": wave_batch,
                "generated_tokens": wave_batch * steps,
                "kv_setup_seconds": wave_kv_setup,
                "decode_wall_seconds": elapsed,
                "decode_tokens_per_second": wave_batch * steps / elapsed,
                "expert_h2d_fetches": _metric_delta(
                    before, after, "expert_h2d_fetches"
                ),
                "final_logits_sha256": digest,
            }
        )
        del output, past
        torch.cuda.empty_cache()
        print(
            f"wave {wave_index + 1}/{ceil(requests / args.batch_size)} "
            f"batch={wave_batch} decode={elapsed:.3f}s",
            flush=True,
        )

    decode_makespan = sum(wave["decode_wall_seconds"] for wave in waves)
    generated = requests * steps
    natural_assignments = totals["natural_route_assignments"]
    result = {
        "config": config.name,
        "gpu_physical_index": 0,
        "policy": args.policy,
        "k": args.k,
        "batch_size": args.batch_size,
        "requests": requests,
        "decode_steps": steps,
        "waves": len(waves),
        "partial_wave_batch_size": requests % args.batch_size,
        "quota_state": "cold_each_wave" if args.cold_each_wave else "warm_across_waves",
        "kv_setup": "static_zero",
        "generated_tokens": generated,
        "fixed_workload_decode_makespan_seconds": decode_makespan,
        "fixed_workload_tokens_per_second": generated / decode_makespan,
        "steady_full_batch_tokens_per_second": (
            len(full_wave_seconds) * args.batch_size * steps / sum(full_wave_seconds)
            if full_wave_seconds
            else None
        ),
        "cold_start_seconds": (
            host_preload_seconds + model_load_seconds + policy_initialization_seconds
        ),
        "kv_setup_seconds": kv_setup_seconds,
        "cold_start_and_kv_included_makespan_seconds": (
            host_preload_seconds
            + model_load_seconds
            + policy_initialization_seconds
            + kv_setup_seconds
            + decode_makespan
        ),
        "host_store_preload_seconds": host_preload_seconds,
        "model_load_seconds": model_load_seconds,
        "policy_initialization_seconds": policy_initialization_seconds,
        "forced_routing_trace_sha256": trace.digest(),
        "forced_output_ids_sha256": hashlib.sha256(
            np.asarray(
                [row["forced_output_ids"][:steps] for row in examples],
                dtype=np.int64,
            ).tobytes()
        ).hexdigest(),
        "final_logits_sha256_by_wave": logits_digests,
        "natural_route_mismatch_rate": (
            totals["natural_route_mismatches"] / natural_assignments
            if natural_assignments
            else 0.0
        ),
        **totals,
        "wave_results": waves,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
