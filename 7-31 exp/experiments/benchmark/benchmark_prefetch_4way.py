from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json, git_sha
from experiments.runtime.expert_slot import contiguous_projection_views
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import EXPERT_SHAPES
from experiments.runtime.serial_expert_executor import SerialExpertExecutor


MODES = ("compute_only", "copy_only", "sequential_copy_compute", "double_buffer_overlap")


def rotated_modes(repeat: int) -> tuple[str, ...]:
    """Deterministically rotate modes to remove fixed-order clock/thermal bias."""
    offset = repeat % len(MODES)
    return MODES[offset:] + MODES[:offset]


def _digest(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _trial(
    *,
    mode: str,
    source: torch.Tensor,
    buffers: list[torch.Tensor],
    views: list[dict[str, torch.Tensor]],
    inputs: torch.Tensor,
    iterations: int,
    instrument_operations: bool,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    compute_stream = torch.cuda.Stream(device=0)
    copy_stream = torch.cuda.Stream(device=0)
    origin = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_stop = torch.cuda.Event(enable_timing=True)
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_stop = torch.cuda.Event(enable_timing=True)
    operation_copy_events = []
    operation_compute_events = []
    ready = [torch.cuda.Event() for _ in range(iterations)]
    done = [torch.cuda.Event() for _ in range(iterations)]
    initial_done = [torch.cuda.Event(), torch.cuda.Event()]
    initial_done[0].record(torch.cuda.current_stream())
    initial_done[1].record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    wall_started = time.perf_counter()
    origin.record(torch.cuda.current_stream())
    compute_stream.wait_event(origin)
    copy_stream.wait_event(origin)
    output = None

    if mode == "compute_only":
        with torch.cuda.stream(compute_stream):
            compute_start.record(compute_stream)
            for _ in range(iterations):
                started = stopped = None
                if instrument_operations:
                    started = torch.cuda.Event(enable_timing=True)
                    stopped = torch.cuda.Event(enable_timing=True)
                    started.record(compute_stream)
                output = SerialExpertExecutor._mlp(inputs, views[0])
                if stopped is not None:
                    stopped.record(compute_stream)
                    operation_compute_events.append((started, stopped))
            compute_stop.record(compute_stream)
        torch.cuda.current_stream().wait_event(compute_stop)
    elif mode == "copy_only":
        with torch.cuda.stream(copy_stream):
            copy_start.record(copy_stream)
            for iteration in range(iterations):
                started = stopped = None
                if instrument_operations:
                    started = torch.cuda.Event(enable_timing=True)
                    stopped = torch.cuda.Event(enable_timing=True)
                    started.record(copy_stream)
                buffers[iteration % 2].copy_(source, non_blocking=True)
                if stopped is not None:
                    stopped.record(copy_stream)
                    operation_copy_events.append((started, stopped))
            copy_stop.record(copy_stream)
        torch.cuda.current_stream().wait_event(copy_stop)
    elif mode == "sequential_copy_compute":
        with torch.cuda.stream(compute_stream):
            compute_start.record(compute_stream)
            copy_start.record(compute_stream)
            for iteration in range(iterations):
                slot = iteration % 2
                copy_started = copy_stopped = None
                compute_started = compute_stopped = None
                if instrument_operations:
                    copy_started = torch.cuda.Event(enable_timing=True)
                    copy_stopped = torch.cuda.Event(enable_timing=True)
                    copy_started.record(compute_stream)
                buffers[slot].copy_(source, non_blocking=True)
                if copy_stopped is not None:
                    copy_stopped.record(compute_stream)
                    operation_copy_events.append((copy_started, copy_stopped))
                    compute_started = torch.cuda.Event(enable_timing=True)
                    compute_stopped = torch.cuda.Event(enable_timing=True)
                    compute_started.record(compute_stream)
                output = SerialExpertExecutor._mlp(inputs, views[slot])
                if compute_stopped is not None:
                    compute_stopped.record(compute_stream)
                    operation_compute_events.append((compute_started, compute_stopped))
            copy_stop.record(compute_stream)
            compute_stop.record(compute_stream)
        torch.cuda.current_stream().wait_event(compute_stop)
    else:
        last_done = list(initial_done)
        copy_start.record(copy_stream)
        compute_start.record(compute_stream)
        for iteration in range(iterations):
            slot = iteration % 2
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(last_done[slot])
                copy_started = copy_stopped = None
                if instrument_operations:
                    copy_started = torch.cuda.Event(enable_timing=True)
                    copy_stopped = torch.cuda.Event(enable_timing=True)
                    copy_started.record(copy_stream)
                buffers[slot].copy_(source, non_blocking=True)
                if copy_stopped is not None:
                    copy_stopped.record(copy_stream)
                    operation_copy_events.append((copy_started, copy_stopped))
                ready[iteration].record(copy_stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(ready[iteration])
                compute_started = compute_stopped = None
                if instrument_operations:
                    compute_started = torch.cuda.Event(enable_timing=True)
                    compute_stopped = torch.cuda.Event(enable_timing=True)
                    compute_started.record(compute_stream)
                output = SerialExpertExecutor._mlp(inputs, views[slot])
                if compute_stopped is not None:
                    compute_stopped.record(compute_stream)
                    operation_compute_events.append((compute_started, compute_stopped))
                done[iteration].record(compute_stream)
            last_done[slot] = done[iteration]
        copy_stop.record(copy_stream)
        compute_stop.record(compute_stream)
        torch.cuda.current_stream().wait_event(copy_stop)
        torch.cuda.current_stream().wait_event(compute_stop)

    finished.record(torch.cuda.current_stream())
    finished.synchronize()
    result = {
        "mode": mode,
        "wall_ms": (time.perf_counter() - wall_started) * 1000,
        "cuda_makespan_ms": origin.elapsed_time(finished),
        "compute_stream_span_ms": (
            compute_start.elapsed_time(compute_stop)
            if mode != "copy_only"
            else None
        ),
        "copy_stream_span_ms": (
            copy_start.elapsed_time(copy_stop)
            if mode != "compute_only"
            else None
        ),
        "instrumented_operation_active_compute_ms": (
            sum(start.elapsed_time(stop) for start, stop in operation_compute_events)
            if instrument_operations
            else None
        ),
        "instrumented_operation_active_copy_ms": (
            sum(start.elapsed_time(stop) for start, stop in operation_copy_events)
            if instrument_operations
            else None
        ),
        "final_output_sha256": _digest(output),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-way packed-Expert copy/compute overlap microbenchmark."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--tokens-per-expert", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 5 or args.repeats < 5 or args.profile_repeats < 5:
        raise ValueError("audit protocol requires at least five warmups and repeats")
    if args.iterations <= 1 or args.tokens_per_expert <= 0:
        raise ValueError("iterations and tokens per Expert must be positive")

    config = load_config(args.config)
    require_gpu0(torch)
    if args.output is None:
        args.output = (
            Path("experiments/results/by_commit")
            / git_sha()[:12]
            / "prefetch_4way.json"
        )
    store = PinnedExpertStore(config.model.path, max_pinned_experts=2, pin_weights=True)
    source = store.get(args.layer, args.expert)
    if not isinstance(source, torch.Tensor) or not source.is_pinned():
        raise AssertionError("microbenchmark requires one packed pinned Expert")
    buffers = [
        torch.empty(source.numel(), dtype=source.dtype, device="cuda:0")
        for _ in range(2)
    ]
    views = [contiguous_projection_views(buffer, EXPERT_SHAPES) for buffer in buffers]
    for buffer in buffers:
        buffer.copy_(source)
    inputs = torch.randn(
        args.tokens_per_expert,
        EXPERT_SHAPES["gate_proj"][1],
        dtype=source.dtype,
        device="cuda:0",
    )
    torch.cuda.synchronize()

    for warmup in range(args.warmups):
        for mode in rotated_modes(warmup):
            _trial(
                mode=mode,
                source=source,
                buffers=buffers,
                views=views,
                inputs=inputs,
                iterations=args.iterations,
                instrument_operations=False,
            )
    trials = {mode: [] for mode in MODES}
    for repeat in range(args.repeats):
        for sequence_position, mode in enumerate(rotated_modes(repeat)):
            value = _trial(
                mode=mode,
                source=source,
                buffers=buffers,
                views=views,
                inputs=inputs,
                iterations=args.iterations,
                instrument_operations=False,
            )
            value["repeat"] = repeat
            value["execution_sequence_position"] = sequence_position
            trials[mode].append(value)
    profiles = {mode: [] for mode in MODES}
    for repeat in range(args.profile_repeats):
        for sequence_position, mode in enumerate(rotated_modes(repeat)):
            value = _trial(
                mode=mode,
                source=source,
                buffers=buffers,
                views=views,
                inputs=inputs,
                iterations=args.iterations,
                instrument_operations=True,
            )
            value["repeat"] = repeat
            value["execution_sequence_position"] = sequence_position
            profiles[mode].append(value)

    summaries = {
        mode: {
            "wall_ms": _summary([row["wall_ms"] for row in trials[mode]]),
            "cuda_makespan_ms": _summary(
                [row["cuda_makespan_ms"] for row in trials[mode]]
            ),
        }
        for mode in MODES
    }
    sequential = summaries["sequential_copy_compute"]["cuda_makespan_ms"]["median"]
    overlap = summaries["double_buffer_overlap"]["cuda_makespan_ms"]["median"]
    isolated_profile_compute = statistics.median(
        row["instrumented_operation_active_compute_ms"]
        for row in profiles["compute_only"]
    )
    isolated_profile_copy = statistics.median(
        row["instrumented_operation_active_copy_ms"]
        for row in profiles["copy_only"]
    )
    overlap_profile_compute = statistics.median(
        row["instrumented_operation_active_compute_ms"]
        for row in profiles["double_buffer_overlap"]
    )
    overlap_profile_copy = statistics.median(
        row["instrumented_operation_active_copy_ms"]
        for row in profiles["double_buffer_overlap"]
    )
    result = {
        "benchmark": "packed_expert_prefetch_4way",
        "gpu_physical_index": 0,
        "config": config.name,
        "layer": args.layer,
        "expert": args.expert,
        "tokens_per_expert": args.tokens_per_expert,
        "iterations_per_repeat": args.iterations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
        "mode_order_control": "deterministic_rotation_by_repeat",
        "expert_bytes": source.numel() * source.element_size(),
        "expert_layout": "single_contiguous_pinned_tensor",
        "h2d_copy_operations_per_iteration": 1,
        "summaries": summaries,
        "useful_hidden_time_ms": sequential - overlap,
        "useful_hidden_fraction_of_sequential": (
            (sequential - overlap) / sequential if sequential else None
        ),
        "overlap_compute_active_slowdown_ratio": (
            overlap_profile_compute / isolated_profile_compute
        ),
        "overlap_copy_active_slowdown_ratio": (
            overlap_profile_copy / isolated_profile_copy
        ),
        "active_slowdown_measurement": (
            "separate instrumented trials; excluded from uninstrumented makespan medians"
        ),
        "logits_or_output_digests_match": (
            len(
                {
                    row["final_output_sha256"]
                    for mode in ("compute_only", "sequential_copy_compute", "double_buffer_overlap")
                    for row in trials[mode]
                }
            )
            == 1
        ),
        "trials": trials,
        "instrumented_profiles": profiles,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "benchmark", "useful_hidden_time_ms", "useful_hidden_fraction_of_sequential",
        "overlap_compute_active_slowdown_ratio", "overlap_copy_active_slowdown_ratio",
        "logits_or_output_digests_match",
    )}, indent=2))


if __name__ == "__main__":
    main()
