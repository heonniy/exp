from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from experiments.benchmark.run_offloaded_decode import _read_examples
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


CONFIGURATIONS = (
    ("prefetch_off", 0, "compute_first"),
    ("prefetch_on_copy_first", 1, "copy_first"),
    ("prefetch_on_compute_first", 1, "compute_first"),
)


def _detach_engine(model) -> None:
    for layer in model.model.layers:
        object.__setattr__(layer.mlp.experts, "_engine", None)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


def _trial(
    *,
    model,
    host_store: PinnedExpertStore,
    trace: RoutingTrace,
    forced_tokens: np.ndarray,
    peak_sequence_length: int,
    input_tokens: int,
    steps: int,
    name: str,
    prefetch_depth: int,
    submit_order: str,
    timeline: bool,
) -> dict:
    batch_size = int(forced_tokens.shape[0])
    _detach_engine(model)
    torch.cuda.empty_cache()
    engine = OffloadedExpertEngine(
        host_store,
        StreamingRuntimeManager(),
        prefetch_depth=prefetch_depth,
        track_timeline=timeline,
        prefetch_submit_order=submit_order,
    )
    attach_engine(model, engine)
    past = make_static_kv_cache(
        model,
        batch_size=batch_size,
        max_cache_length=peak_sequence_length,
        initial_sequence_length=input_tokens,
    )
    engine.set_forced_routing(trace.routing_expert_ids[:batch_size, :steps])
    torch.cuda.synchronize()
    cuda_started = torch.cuda.Event(enable_timing=True)
    cuda_stopped = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
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
    result = {
        "name": name,
        "prefetch_depth": prefetch_depth,
        "prefetch_submit_order": submit_order if prefetch_depth else None,
        "timeline_events_enabled": timeline,
        "decode_wall_seconds": time.perf_counter() - wall_started,
        "decode_cuda_seconds": cuda_started.elapsed_time(cuda_stopped) / 1000,
        "final_logits_sha256": _digest(output.logits),
        **engine.metrics(),
    }
    # Scheduling is the subject of this benchmark. Per-step/layer routing
    # details are already covered by the dedicated routing diagnostic and
    # would dominate both this result file and stdout at larger batches.
    result.pop("natural_route_mismatch_by_step_layer", None)
    _detach_engine(model)
    del output, past, engine
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate prefetch OFF, copy-first ON, and compute-first ON with "
            "one shared packed pinned-weight store on physical GPU 0."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/h100_lmsys_4k256.yaml"),
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("artifacts/data/lmsys_4k256_evaluation.jsonl"),
    )
    parser.add_argument(
        "--forced-routing-trace",
        type=Path,
        default=Path("artifacts/traces/evaluation_4k256.npz"),
    )
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeline-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    if not 0 < args.decode_steps <= config.dataset.output_tokens:
        raise ValueError("decode steps are outside the workload")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.warmups < 0 or args.repeats <= 0 or args.timeline_repeats < 0:
        raise ValueError("warmups/repeats must be non-negative/positive")
    examples = _read_examples(args.workload, args.batch_size)
    forced_tokens = np.asarray(
        [row["forced_output_ids"][: args.decode_steps] for row in examples],
        dtype=np.int64,
    )
    trace = RoutingTrace.load(args.forced_routing_trace)
    if [str(value) for value in trace.conversation_ids[: args.batch_size]] != [
        str(row["conversation_id"]) for row in examples
    ]:
        raise ValueError("trace/workload conversation mismatch")

    host_store = PinnedExpertStore(
        config.model.path,
        max_pinned_experts=(
            config.model.num_moe_layers * config.model.num_experts_per_layer
        ),
        pin_weights=True,
    )
    preload_seconds = host_store.preload_all(
        config.model.num_moe_layers, config.model.num_experts_per_layer
    )
    bootstrap = OffloadedExpertEngine(
        host_store, StreamingRuntimeManager(), prefetch_depth=0
    )
    model = load_offloaded_qwen(config.model.path, bootstrap)
    _detach_engine(model)
    del bootstrap
    torch.cuda.empty_cache()

    warmups = []
    for _ in range(args.warmups):
        for name, depth, order in CONFIGURATIONS:
            warmups.append(
                _trial(
                    model=model,
                    host_store=host_store,
                    trace=trace,
                    forced_tokens=forced_tokens,
                    peak_sequence_length=config.peak_sequence_length,
                    input_tokens=config.dataset.input_tokens,
                    steps=args.decode_steps,
                    name=name,
                    prefetch_depth=depth,
                    submit_order=order,
                    timeline=False,
                )
            )

    trials = []
    for repeat in range(args.repeats):
        ordered = CONFIGURATIONS[repeat % len(CONFIGURATIONS) :] + CONFIGURATIONS[
            : repeat % len(CONFIGURATIONS)
        ]
        for name, depth, order in ordered:
            value = _trial(
                model=model,
                host_store=host_store,
                trace=trace,
                forced_tokens=forced_tokens,
                peak_sequence_length=config.peak_sequence_length,
                input_tokens=config.dataset.input_tokens,
                steps=args.decode_steps,
                name=name,
                prefetch_depth=depth,
                submit_order=order,
                timeline=False,
            )
            value["repeat"] = repeat
            trials.append(value)

    timeline_trials = []
    for repeat in range(args.timeline_repeats):
        ordered = CONFIGURATIONS[repeat % len(CONFIGURATIONS) :] + CONFIGURATIONS[
            : repeat % len(CONFIGURATIONS)
        ]
        for name, depth, order in ordered:
            value = _trial(
                model=model,
                host_store=host_store,
                trace=trace,
                forced_tokens=forced_tokens,
                peak_sequence_length=config.peak_sequence_length,
                input_tokens=config.dataset.input_tokens,
                steps=args.decode_steps,
                name=name,
                prefetch_depth=depth,
                submit_order=order,
                timeline=True,
            )
            value["repeat"] = repeat
            timeline_trials.append(value)

    summary = {}
    for name, _, _ in CONFIGURATIONS:
        values = [row for row in trials if row["name"] == name]
        summary[name] = {
            "decode_wall_seconds_median": statistics.median(
                row["decode_wall_seconds"] for row in values
            ),
            "decode_cuda_seconds_median": statistics.median(
                row["decode_cuda_seconds"] for row in values
            ),
            "final_logits_sha256": sorted(
                {row["final_logits_sha256"] for row in values}
            ),
        }
        timeline_values = [row for row in timeline_trials if row["name"] == name]
        if timeline_values:
            for metric in (
                "decode_wall_seconds",
                "total_h2d_duration_ms",
                "overlapped_h2d_ms",
                "overlap_ratio",
                "exposed_h2d_stall_ms",
                "compute_stream_h2d_wait_ms",
                "expert_compute_ms",
            ):
                summary[name][f"instrumented_{metric}_median"] = statistics.median(
                    row[metric] for row in timeline_values
                )
    baseline = summary["prefetch_off"]["decode_wall_seconds_median"]
    for name in ("prefetch_on_copy_first", "prefetch_on_compute_first"):
        value = summary[name]["decode_wall_seconds_median"]
        summary[name]["wall_speedup_vs_off"] = baseline / value
        summary[name]["wall_reduction_vs_off"] = 1 - value / baseline

    digests = {
        row["final_logits_sha256"] for row in trials + timeline_trials
    }
    result = {
        "config": config.name,
        "gpu_physical_index": 0,
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
        "host_memory_mode": "pinned_weights",
        "host_expert_layout": "one_layer_contiguous_pinned_slabs",
        "expert_h2d_copy_operations_per_fetch": 1,
        "host_store_preload_seconds": preload_seconds,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeline_repeats": args.timeline_repeats,
        "all_logits_digests_match": len(digests) == 1,
        "summary": summary,
        "timeline_trials": timeline_trials,
        "trials": trials,
    }
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "batch_size": args.batch_size,
                "decode_steps": args.decode_steps,
                "all_logits_digests_match": len(digests) == 1,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
