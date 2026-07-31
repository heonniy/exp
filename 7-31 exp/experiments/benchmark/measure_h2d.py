from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json


def measure_copy(size_bytes: int, copies: int, repeats: int) -> dict:
    source = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    destinations = [
        torch.empty(size_bytes, dtype=torch.uint8, device="cuda:0")
        for _ in range(2)
    ]
    stream = torch.cuda.Stream(device=0)
    for index in range(4):
        destinations[index % 2].copy_(source, non_blocking=True)
    torch.cuda.synchronize()

    durations_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for index in range(copies):
                destinations[index % 2].copy_(source, non_blocking=True)
            stop.record(stream)
        stop.synchronize()
        durations_ms.append(float(start.elapsed_time(stop)))
    median_ms = statistics.median(durations_ms)
    total_bytes = size_bytes * copies
    return {
        "size_bytes": size_bytes,
        "copies_per_repeat": copies,
        "repeats": repeats,
        "duration_ms_median": median_ms,
        "duration_ms_min": min(durations_ms),
        "duration_ms_max": max(durations_ms),
        "bandwidth_gib_per_second": total_bytes / (median_ms / 1000) / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure pinned-memory Expert H2D copies on physical GPU 0."
    )
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--continuous-copies", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = require_gpu0(torch)
    result = {
        "gpu_physical_index": gpu.physical_index,
        "gpu_name": gpu.name,
        "source_memory": "pinned",
        "single_expert": measure_copy(args.expert_bytes, 1, args.repeats),
        "continuous_experts": measure_copy(
            args.expert_bytes, args.continuous_copies, args.repeats
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

