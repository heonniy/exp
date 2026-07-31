from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import (
    EXPERT_SHAPES,
    OffloadedExpertEngine,
    attach_engine,
    load_offloaded_qwen,
)
from experiments.runtime.residency_manager import (
    PermanentRuntimeManager,
    QuotaLRURuntimeManager,
    StreamingRuntimeManager,
)
from experiments.trace.select_permanent import select_topk
from experiments.trace.trace_schema import RoutingTrace


def _read_examples(path: Path, count: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"{path} contains only {len(rows)} rows; need {count}")
    return rows


def _manager(
    policy: str,
    k: int,
    num_layers: int,
    host_store: PinnedExpertStore,
    calibration_trace: RoutingTrace | None,
) -> tuple[object, float]:
    started = time.perf_counter()
    if policy == "stream2":
        if k != 0:
            raise ValueError("stream2 requires k=0")
        manager = StreamingRuntimeManager()
    elif policy == "permanent_k":
        if calibration_trace is None:
            raise ValueError("permanent_k requires --calibration-trace")
        selections = select_topk(calibration_trace, k, "presence")
        manager = PermanentRuntimeManager(
            selections.tolist(),
            EXPERT_SHAPES,
            torch.bfloat16,
            host_store,
        )
    elif policy == "quota_lru_k":
        manager = QuotaLRURuntimeManager(
            num_layers,
            k,
            EXPERT_SHAPES,
            torch.bfloat16,
        )
    else:
        raise ValueError(f"unknown policy: {policy}")
    torch.cuda.synchronize()
    return manager, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real serial per-Expert Qwen decode with CPU Expert offload on GPU 0."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=["stream2", "permanent_k", "quota_lru_k"], required=True
    )
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument("--calibration-trace", type=Path)
    parser.add_argument(
        "--forced-routing-trace",
        type=Path,
        help="Force identical token/layer top-k Expert IDs across policies.",
    )
    parser.add_argument("--max-pinned-experts", type=int, default=16)
    parser.add_argument(
        "--host-memory-mode",
        choices=["pinned_weights", "pinned_staging"],
        default="pinned_weights",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    steps = args.decode_steps or config.dataset.output_tokens
    if not 0 < steps <= config.dataset.output_tokens:
        raise ValueError("decode steps exceed the fixed workload output length")
    examples = _read_examples(args.workload, args.batch_size)
    forced_tokens = np.asarray(
        [row["forced_output_ids"][:steps] for row in examples], dtype=np.int64
    )
    if args.host_memory_mode == "pinned_weights" and args.max_pinned_experts < (
        config.model.num_moe_layers * config.model.num_experts_per_layer
    ):
        raise ValueError(
            "pinned_weights mode requires --max-pinned-experts 6144 so cyclic "
            "host-cache eviction cannot contaminate H2D timing"
        )
    host_store = PinnedExpertStore(
        config.model.path,
        args.max_pinned_experts,
        pin_weights=args.host_memory_mode == "pinned_weights",
    )

    # Prefill uses streaming only to construct real KV. Its cache state and
    # counters are discarded before the decode-only timer.
    prefill_manager = StreamingRuntimeManager()
    prefill_engine = OffloadedExpertEngine(host_store, prefill_manager)
    model_load_started = time.perf_counter()
    model = load_offloaded_qwen(config.model.path, prefill_engine)
    model_load_seconds = time.perf_counter() - model_load_started
    input_ids = torch.tensor(
        [row["input_ids"] for row in examples],
        dtype=torch.long,
        device="cuda:0",
    )
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
            output_router_logits=False,
            return_dict=True,
        )
    past = prefill.past_key_values
    del prefill, input_ids
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started

    calibration = (
        RoutingTrace.load(args.calibration_trace)
        if args.calibration_trace is not None
        else None
    )
    manager, policy_initialization_seconds = _manager(
        args.policy,
        args.k,
        config.model.num_moe_layers,
        host_store,
        calibration,
    )
    engine = OffloadedExpertEngine(host_store, manager)
    attach_engine(model, engine)
    del prefill_engine
    torch.cuda.empty_cache()

    expected = None
    if args.forced_routing_trace is not None:
        trace = RoutingTrace.load(args.forced_routing_trace)
        expected_ids = [str(value) for value in trace.conversation_ids[: args.batch_size]]
        workload_ids = [str(row["conversation_id"]) for row in examples]
        if expected_ids != workload_ids:
            raise ValueError("forced routing trace does not match workload row order")
        expected = trace.routing_expert_ids[: args.batch_size, :steps]
        engine.set_forced_routing(expected)

    torch.cuda.reset_peak_memory_stats()
    host_before = host_store.metrics()
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    cuda_started = torch.cuda.Event(enable_timing=True)
    cuda_stopped = torch.cuda.Event(enable_timing=True)
    cuda_started.record()
    with torch.inference_mode():
        for step in range(steps):
            engine.decode_step = step
            token = torch.as_tensor(
                forced_tokens[:, step, None], dtype=torch.long, device="cuda:0"
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
    cuda_stopped.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = cuda_started.elapsed_time(cuda_stopped) / 1000
    generated = args.batch_size * steps
    engine_metrics = engine.metrics()
    host_after = host_store.metrics()
    metrics = {
        "config": config.name,
        "gpu_physical_index": 0,
        "policy": args.policy,
        "k": args.k,
        "batch_size": args.batch_size,
        "decode_steps": steps,
        "generated_tokens": generated,
        "decode_wall_seconds": wall_seconds,
        "decode_cuda_seconds": cuda_seconds,
        "decode_tokens_per_second": generated / wall_seconds,
        "decode_ms_per_generated_token": wall_seconds * 1000 / generated,
        "policy_initialization_seconds": policy_initialization_seconds,
        "model_load_seconds": model_load_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_host_stage_calls": (
            host_after["host_stage_calls"] - host_before["host_stage_calls"]
        ),
        "decode_host_stage_cache_hits": (
            host_after["host_stage_cache_hits"]
            - host_before["host_stage_cache_hits"]
        ),
        "decode_host_stage_seconds": (
            host_after["host_stage_seconds"] - host_before["host_stage_seconds"]
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "forced_output_ids_sha256": __import__("hashlib")
        .sha256(forced_tokens.tobytes())
        .hexdigest(),
        **engine_metrics,
    }
    atomic_write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
