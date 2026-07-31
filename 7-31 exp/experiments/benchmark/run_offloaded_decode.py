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
from experiments.runtime.kv_cache import make_static_kv_cache
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


class CudaModuleTimer:
    def __init__(self, modules: list[torch.nn.Module]):
        self._started: dict[torch.nn.Module, torch.cuda.Event] = {}
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._handles = []
        for module in modules:
            self._handles.append(module.register_forward_pre_hook(self._pre))
            self._handles.append(module.register_forward_hook(self._post))

    def _pre(self, module, _inputs) -> None:
        started = torch.cuda.Event(enable_timing=True)
        started.record(torch.cuda.current_stream())
        self._started[module] = started

    def _post(self, module, _inputs, _output) -> None:
        stopped = torch.cuda.Event(enable_timing=True)
        stopped.record(torch.cuda.current_stream())
        self._pairs.append((self._started.pop(module), stopped))

    def elapsed_ms(self) -> float:
        return sum(start.elapsed_time(stop) for start, stop in self._pairs)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


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
    num_experts: int,
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
    elif policy == "full_resident":
        if k != num_experts:
            raise ValueError("full_resident requires k=num_experts")
        manager = PermanentRuntimeManager(
            [range(num_experts) for _ in range(num_layers)],
            EXPERT_SHAPES,
            torch.bfloat16,
            host_store,
            name="full_resident",
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
        "--policy",
        choices=["stream2", "permanent_k", "quota_lru_k", "full_resident"],
        required=True,
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
    parser.add_argument("--prefetch-depth", choices=[0, 1], type=int, default=1)
    parser.add_argument("--timeline-events", action="store_true")
    parser.add_argument(
        "--kv-setup",
        choices=["real_prefill", "static_zero"],
        default="real_prefill",
        help=(
            "real_prefill constructs KV from the 4K prompt; static_zero "
            "preallocates the full peak KV fixture outside the decode timer"
        ),
    )
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
        [row["forced_output_ids"][:steps] for row in examples], dtype=np.int32
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
    host_store_preload_seconds = 0.0
    if args.host_memory_mode == "pinned_weights":
        host_store_preload_seconds = host_store.preload_all(
            config.model.num_moe_layers,
            config.model.num_experts_per_layer,
        )

    # Prefill uses streaming only to construct real KV. Its cache state and
    # counters are discarded before the decode-only timer.
    prefill_manager = StreamingRuntimeManager()
    prefill_engine = OffloadedExpertEngine(
        host_store,
        prefill_manager,
        prefetch_depth=args.prefetch_depth,
        track_timeline=False,
    )
    model_load_started = time.perf_counter()
    model = load_offloaded_qwen(config.model.path, prefill_engine)
    model_load_seconds = time.perf_counter() - model_load_started
    kv_setup_started = time.perf_counter()
    if args.kv_setup == "real_prefill":
        input_ids = torch.tensor(
            [row["input_ids"] for row in examples],
            dtype=torch.long,
            device="cuda:0",
        )
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
    else:
        past = make_static_kv_cache(
            model,
            batch_size=args.batch_size,
            max_cache_length=config.peak_sequence_length,
            initial_sequence_length=config.dataset.input_tokens,
        )
    torch.cuda.synchronize()
    kv_setup_seconds = time.perf_counter() - kv_setup_started

    calibration = (
        RoutingTrace.load(args.calibration_trace)
        if args.calibration_trace is not None
        else None
    )
    # Drop bootstrap transient slots before allocating policy residency.
    for layer in model.model.layers:
        object.__setattr__(layer.mlp.experts, "_engine", None)
    del prefill_engine
    torch.cuda.empty_cache()
    manager, policy_initialization_seconds = _manager(
        args.policy,
        args.k,
        config.model.num_moe_layers,
        config.model.num_experts_per_layer,
        host_store,
        calibration,
    )
    engine = OffloadedExpertEngine(
        host_store,
        manager,
        prefetch_depth=args.prefetch_depth,
        track_timeline=args.timeline_events,
    )
    attach_engine(model, engine)
    torch.cuda.empty_cache()

    expected = None
    forced_routing_trace_sha256 = None
    forced_routing_ids_sha256 = None
    if args.forced_routing_trace is not None:
        trace = RoutingTrace.load(args.forced_routing_trace)
        expected_ids = [str(value) for value in trace.conversation_ids[: args.batch_size]]
        workload_ids = [str(row["conversation_id"]) for row in examples]
        if expected_ids != workload_ids:
            raise ValueError("forced routing trace does not match workload row order")
        if not np.array_equal(
            trace.forced_output_ids[: args.batch_size, :steps],
            forced_tokens,
        ):
            raise ValueError("forced token IDs differ between trace and workload")
        expected = trace.routing_expert_ids[: args.batch_size, :steps]
        forced_routing_trace_sha256 = trace.digest()
        forced_routing_ids_sha256 = __import__("hashlib").sha256(
            expected.tobytes()
        ).hexdigest()
        engine.set_forced_routing(expected)

    torch.cuda.reset_peak_memory_stats()
    host_before = host_store.metrics()
    attention_timer = router_timer = None
    if args.timeline_events:
        attention_timer = CudaModuleTimer(
            [layer.self_attn for layer in model.model.layers]
        )
        router_timer = CudaModuleTimer(
            [layer.mlp.gate for layer in model.model.layers]
        )
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
    attention_ms = attention_timer.elapsed_ms() if attention_timer else None
    router_ms = router_timer.elapsed_ms() if router_timer else None
    if attention_timer:
        attention_timer.remove()
    if router_timer:
        router_timer.remove()
    generated = args.batch_size * steps
    engine_metrics = engine.metrics()
    host_after = host_store.metrics()
    final_logits_sha256 = __import__("hashlib").sha256(
        output.logits.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()
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
        "kv_setup": args.kv_setup,
        "kv_setup_seconds": kv_setup_seconds,
        "prefill_seconds": (
            kv_setup_seconds if args.kv_setup == "real_prefill" else 0.0
        ),
        "host_store_preload_seconds": host_store_preload_seconds,
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
        "copy_engine_utilization": (
            engine_metrics["total_h2d_duration_ms"] / (cuda_seconds * 1000)
            if engine_metrics["timeline_events_enabled"] and cuda_seconds
            else None
        ),
        "attention_ms": attention_ms,
        "router_ms": router_ms,
        "other_dense_host_idle_ms": (
            max(
                0.0,
                wall_seconds * 1000
                - attention_ms
                - router_ms
                - engine_metrics["expert_compute_ms"]
                - engine_metrics["exposed_h2d_stall_ms"],
            )
            if args.timeline_events
            else None
        ),
        "forced_output_ids_sha256": __import__("hashlib")
        .sha256(forced_tokens.tobytes())
        .hexdigest(),
        "forced_routing_trace_sha256": forced_routing_trace_sha256,
        "forced_routing_ids_sha256": forced_routing_ids_sha256,
        "final_logits_sha256": final_logits_sha256,
        **engine_metrics,
    }
    atomic_write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
