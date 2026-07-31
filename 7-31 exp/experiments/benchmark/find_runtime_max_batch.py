from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from experiments.benchmark.memory_accounting import GIB, account_memory
from experiments.benchmark.measure_kv_bytes import logical_kv_bytes_per_token
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


def _detach_engine(model) -> None:
    for layer in model.model.layers:
        object.__setattr__(layer.mlp.experts, "_engine", None)


def _make_peak_cache(model, batch_size: int, peak_sequence_length: int):
    return make_static_kv_cache(
        model,
        batch_size=batch_size,
        max_cache_length=peak_sequence_length,
        initial_sequence_length=peak_sequence_length - 1,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )


def probe_candidate(
    *,
    model,
    host_store: PinnedExpertStore,
    policy: str,
    k: int,
    num_layers: int,
    num_experts: int,
    calibration: RoutingTrace | None,
    batch_size: int,
    peak_sequence_length: int,
    safety_margin_bytes: int,
    token_ids: np.ndarray,
    routing: np.ndarray,
) -> dict:
    manager = engine = cache = reserve = output = None
    _detach_engine(model)
    torch.cuda.empty_cache()
    started = time.perf_counter()
    try:
        manager, policy_initialization_seconds = _manager(
            policy,
            k,
            num_layers,
            num_experts,
            host_store,
            calibration,
        )
        engine = OffloadedExpertEngine(host_store, manager, prefetch_depth=1)
        attach_engine(model, engine)
        reserve = torch.empty(
            safety_margin_bytes, dtype=torch.uint8, device="cuda:0"
        )
        reserve.zero_()
        cache = _make_peak_cache(model, batch_size, peak_sequence_length)
        engine.set_forced_routing(routing[:batch_size, None, :, :])
        engine.decode_step = 0
        input_ids = torch.as_tensor(
            token_ids[:batch_size, None], dtype=torch.long, device="cuda:0"
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                output_router_logits=False,
                return_dict=True,
            )
        torch.cuda.synchronize()
        metrics = engine.metrics()
        if policy == "quota_lru_k":
            counts = metrics["resident_count_by_layer"]
            if len(counts) != num_layers or any(value > k for value in counts):
                raise AssertionError("quota residency invariant failed")
        if metrics.get("d2d_admission_copies", 0):
            raise AssertionError("runtime performed a D2D admission copy")
        return {
            "batch_size": batch_size,
            "feasible": True,
            "elapsed_seconds": time.perf_counter() - started,
            "policy_initialization_seconds": policy_initialization_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "expert_h2d_fetches": metrics["expert_h2d_fetches"],
            "forced_routing": metrics["forced_routing"],
        }
    except torch.OutOfMemoryError as error:
        return {
            "batch_size": batch_size,
            "feasible": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error).splitlines()[0],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        _detach_engine(model)
        del output, cache, reserve, engine, manager
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find measured Bmax with the real dense model, real static KV cache, "
            "real routing replay, and the selected Expert residency policy."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--forced-routing-trace", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path)
    parser.add_argument(
        "--policy",
        choices=["stream2", "permanent_k", "quota_lru_k", "full_resident"],
        required=True,
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--dense-bytes", type=int, required=True)
    parser.add_argument("--fixed-workspace-bytes", type=int, default=0)
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--max-pinned-experts", type=int, default=6144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    gpu = require_gpu0(torch)
    if args.policy == "stream2" and args.k != 0:
        raise ValueError("stream2 requires k=0")
    if args.policy == "full_resident" and (
        args.k != config.model.num_experts_per_layer
    ):
        raise ValueError("full_resident requires k=128")

    model_config = AutoConfig.from_pretrained(
        config.model.path, local_files_only=True
    ).get_text_config(decoder=True)
    kv_bytes_per_token = logical_kv_bytes_per_token(
        config.model.num_moe_layers,
        model_config.num_key_value_heads,
        model_config.head_dim,
        torch.bfloat16,
    )
    accounting = account_memory(
        total_hbm_bytes=gpu.total_memory,
        dense_resident_bytes=args.dense_bytes,
        fixed_workspace_bytes=args.fixed_workspace_bytes,
        safety_margin_bytes=int(config.runtime.hbm_safety_margin_gib * GIB),
        expert_bytes=args.expert_bytes,
        num_layers=config.model.num_moe_layers,
        k=args.k,
        transient_slots=config.runtime.transient_expert_slots,
        kv_bytes_per_token=kv_bytes_per_token,
        peak_sequence_length=config.peak_sequence_length,
    )
    upper = accounting.theoretical_bmax
    if args.max_batch is not None:
        upper = min(upper, args.max_batch)
    if upper <= 0:
        raise ValueError("theoretical Bmax is zero")

    examples = _read_examples(args.workload, upper)
    trace = RoutingTrace.load(args.forced_routing_trace)
    workload_ids = [str(row["conversation_id"]) for row in examples]
    trace_ids = [str(value) for value in trace.conversation_ids[:upper]]
    if workload_ids != trace_ids:
        raise ValueError("forced routing trace does not match workload row order")
    workload_forced = np.asarray(
        [row["forced_output_ids"] for row in examples], dtype=np.int32
    )
    if not np.array_equal(
        trace.forced_output_ids[:upper, : workload_forced.shape[1]],
        workload_forced,
    ):
        raise ValueError("forced token IDs differ between trace and workload")
    token_ids = np.asarray(
        [row["forced_output_ids"][0] for row in examples], dtype=np.int64
    )
    routing = trace.routing_expert_ids[:upper, 0]
    calibration = (
        RoutingTrace.load(args.calibration_trace)
        if args.calibration_trace is not None
        else None
    )

    # The host cache is intentionally large enough to avoid mmap-to-pinned
    # staging cost changing between binary-search candidates.
    host_store = PinnedExpertStore(
        config.model.path,
        max_pinned_experts=args.max_pinned_experts,
        pin_weights=True,
    )
    host_store_preload_seconds = host_store.preload_all(
        config.model.num_moe_layers,
        config.model.num_experts_per_layer,
    )
    bootstrap = OffloadedExpertEngine(host_store, StreamingRuntimeManager())
    model = load_offloaded_qwen(config.model.path, bootstrap)
    _detach_engine(model)
    del bootstrap
    torch.cuda.empty_cache()

    low, high = 0, upper
    probes: dict[int, dict] = {}
    while low < high:
        candidate = (low + high + 1) // 2
        result = probe_candidate(
            model=model,
            host_store=host_store,
            policy=args.policy,
            k=args.k,
            num_layers=config.model.num_moe_layers,
            num_experts=config.model.num_experts_per_layer,
            calibration=calibration,
            batch_size=candidate,
            peak_sequence_length=config.peak_sequence_length,
            safety_margin_bytes=accounting.safety_margin_bytes,
            token_ids=token_ids,
            routing=routing,
        )
        probes[candidate] = result
        print(
            f"candidate={candidate} feasible={result['feasible']} "
            f"elapsed={result['elapsed_seconds']:.2f}s",
            flush=True,
        )
        if result["feasible"]:
            low = candidate
        else:
            high = candidate - 1

    result = {
        "probe_mode": "real_runtime_static_peak_kv_one_decode_step",
        "gpu_physical_index": 0,
        "config": config.name,
        "policy": args.policy,
        "k": args.k,
        **accounting.as_dict(),
        "measured_bmax": low,
        "search_upper_batch": upper,
        "user_max_batch_cap": args.max_batch,
        "search_truncated_below_theoretical": (
            upper < accounting.theoretical_bmax
        ),
        "forced_routing_trace": str(args.forced_routing_trace),
        "forced_routing_trace_sha256": trace.digest(),
        "host_memory_mode": "pinned_weights",
        "host_store_preload_seconds": host_store_preload_seconds,
        "probes": [probes[key] for key in sorted(probes)],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
