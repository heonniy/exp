from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.benchmark.memory_accounting import GIB, account_memory
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json


def _allocate_bytes(size: int) -> torch.Tensor | None:
    if size == 0:
        return None
    return torch.empty(size, dtype=torch.uint8, device="cuda:0")


def probe_candidate(
    batch_size: int,
    *,
    static_bytes: int,
    persistent_bytes: int,
    transient_bytes: int,
    workspace_bytes: int,
    kv_bytes_per_request: int,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocations = []
    try:
        for size in (
            static_bytes,
            persistent_bytes,
            transient_bytes,
            workspace_bytes,
            kv_bytes_per_request * batch_size,
        ):
            allocation = _allocate_bytes(size)
            if allocation is not None:
                allocation.zero_()
                allocations.append(allocation)
        # Real routing engines can register a callback here. This probe is
        # intentionally labeled synthetic and only validates allocator/HBM math.
        torch.cuda.synchronize()
        return {
            "batch_size": batch_size,
            "feasible": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size,
            "feasible": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        allocations.clear()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic HBM allocation probe for a residency configuration."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--dense-bytes", type=int, required=True)
    parser.add_argument("--workspace-bytes", type=int, default=0)
    parser.add_argument("--safety-margin-gib", type=float, default=2.0)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--transient-slots", type=int, default=2)
    parser.add_argument("--kv-bytes-per-token", type=int, default=98304)
    parser.add_argument("--peak-sequence-length", type=int, default=4352)
    parser.add_argument("--max-batch", type=int)
    args = parser.parse_args()

    gpu = require_gpu0(torch)
    accounting = account_memory(
        total_hbm_bytes=gpu.total_memory,
        dense_resident_bytes=args.dense_bytes,
        fixed_workspace_bytes=args.workspace_bytes,
        safety_margin_bytes=int(args.safety_margin_gib * GIB),
        expert_bytes=args.expert_bytes,
        num_layers=args.num_layers,
        k=args.k,
        transient_slots=args.transient_slots,
        kv_bytes_per_token=args.kv_bytes_per_token,
        peak_sequence_length=args.peak_sequence_length,
    )
    upper = accounting.theoretical_bmax
    if args.max_batch is not None:
        upper = min(upper, args.max_batch)
    if upper == 0:
        result = {
            "probe_mode": "synthetic_accounting",
            **accounting.as_dict(),
            "measured_bmax": 0,
            "probes": [],
        }
        atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return

    low, high = 0, upper
    probes: dict[int, dict] = {}
    persistent = accounting.persistent_expert_bytes
    transient = accounting.transient_expert_bytes
    while low < high:
        candidate = (low + high + 1) // 2
        probe = probe_candidate(
            candidate,
            static_bytes=args.dense_bytes,
            persistent_bytes=persistent,
            transient_bytes=transient,
            workspace_bytes=args.workspace_bytes,
            kv_bytes_per_request=accounting.kv_bytes_per_request,
        )
        probes[candidate] = probe
        if probe["feasible"]:
            low = candidate
        else:
            high = candidate - 1
    result = {
        "probe_mode": "synthetic_accounting",
        "gpu_physical_index": 0,
        **accounting.as_dict(),
        "measured_bmax": low,
        "probes": [probes[key] for key in sorted(probes)],
        "warning": (
            "This allocator probe is not a substitute for the real-runtime "
            "attention/router/decode OOM probe."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

