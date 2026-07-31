from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def logical_kv_bytes_per_token(
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    return (
        2
        * num_layers
        * num_key_value_heads
        * head_dim
        * torch.empty((), dtype=dtype).element_size()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure logical and allocated synthetic KV-cache bytes on GPU 0."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--peak-sequence-length", type=int, default=4352)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    args = parser.parse_args()

    gpu = require_gpu0(torch)
    dtype = DTYPES[args.dtype]
    bytes_per_token = logical_kv_bytes_per_token(
        args.num_layers, args.num_key_value_heads, args.head_dim, dtype
    )
    requested = bytes_per_token * args.peak_sequence_length * args.batch_size
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    # A flat byte allocation measures allocator behavior while avoiding a layout
    # assumption that would accidentally become part of the runtime contract.
    allocation = torch.empty(requested, dtype=torch.uint8, device="cuda:0")
    allocation.zero_()
    torch.cuda.synchronize()
    result = {
        "gpu_physical_index": gpu.physical_index,
        "gpu_name": gpu.name,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "peak_sequence_length": args.peak_sequence_length,
        "logical_kv_bytes_per_token": bytes_per_token,
        "logical_peak_kv_bytes": requested,
        "allocated_delta_bytes": torch.cuda.memory_allocated() - before_allocated,
        "reserved_delta_bytes": torch.cuda.memory_reserved() - before_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del allocation
    torch.cuda.empty_cache()
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

