from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as functional

from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json


def _timed(operation, stream: torch.cuda.Stream, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            operation()
            stop.record(stream)
        stop.synchronize()
        values.append(float(start.elapsed_time(stop)))
    return statistics.median(values)


def measure_m(
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    repeats: int,
    iterations: int,
) -> dict:
    dtype = torch.bfloat16
    x = torch.randn(tokens_per_expert, hidden_size, dtype=dtype, device="cuda:0")
    gate = torch.randn(intermediate_size, hidden_size, dtype=dtype, device="cuda:0")
    up = torch.randn(intermediate_size, hidden_size, dtype=dtype, device="cuda:0")
    down = torch.randn(hidden_size, intermediate_size, dtype=dtype, device="cuda:0")
    copy_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (gate, up, down)
    )
    host = torch.empty(copy_bytes, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty(copy_bytes, dtype=torch.uint8, device="cuda:0")
    compute_stream = torch.cuda.Stream(device=0)
    copy_stream = torch.cuda.Stream(device=0)

    def compute_once() -> None:
        hidden = functional.silu(functional.linear(x, gate)) * functional.linear(x, up)
        functional.linear(hidden, down)

    def copy_once() -> None:
        destination.copy_(host, non_blocking=True)

    def compute() -> None:
        for _ in range(iterations):
            compute_once()

    def copy() -> None:
        for _ in range(iterations):
            copy_once()

    for _ in range(5):
        with torch.cuda.stream(compute_stream):
            compute()
        with torch.cuda.stream(copy_stream):
            copy()
    torch.cuda.synchronize()
    compute_ms = _timed(compute, compute_stream, repeats) / iterations
    copy_ms = _timed(copy, copy_stream, repeats) / iterations

    concurrent_values = []
    for _ in range(repeats):
        wall_start = torch.cuda.Event(enable_timing=True)
        compute_done = torch.cuda.Event(enable_timing=True)
        copy_done = torch.cuda.Event(enable_timing=True)
        wall_start.record(torch.cuda.default_stream())
        compute_stream.wait_event(wall_start)
        copy_stream.wait_event(wall_start)
        with torch.cuda.stream(compute_stream):
            compute()
            compute_done.record(compute_stream)
        with torch.cuda.stream(copy_stream):
            copy()
            copy_done.record(copy_stream)
        torch.cuda.default_stream().wait_event(compute_done)
        torch.cuda.default_stream().wait_event(copy_done)
        wall_stop = torch.cuda.Event(enable_timing=True)
        wall_stop.record(torch.cuda.default_stream())
        wall_stop.synchronize()
        concurrent_values.append(float(wall_start.elapsed_time(wall_stop)) / iterations)
    concurrent_ms = statistics.median(concurrent_values)
    exposed_copy_ms = max(0.0, concurrent_ms - compute_ms)
    return {
        "tokens_per_expert": tokens_per_expert,
        "iterations_per_repeat": iterations,
        "expert_bytes": copy_bytes,
        "compute_ms": compute_ms,
        "copy_ms": copy_ms,
        "concurrent_ms": concurrent_ms,
        "exposed_h2d_ms": exposed_copy_ms,
        "overlap_ratio": max(0.0, min(1.0, 1.0 - exposed_copy_ms / copy_ms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure one-ahead Expert H2D/MLP overlap on physical GPU 0."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-expert", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    gpu = require_gpu0(torch)
    result = {
        "gpu_physical_index": 0,
        "gpu_name": gpu.name,
        "dtype": "bfloat16",
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "measurements": [
            measure_m(
                tokens,
                args.hidden_size,
                args.intermediate_size,
                args.repeats,
                args.iterations,
            )
            for tokens in args.tokens_per_expert
        ],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
