from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open


EXPERT_PATTERN = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>[^.]+)\.(?P<parameter>[^.]+)$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _numel(shape: list[int]) -> int:
    return math.prod(int(dimension) for dimension in shape)


def inspect_checkpoint(model_path: str | Path) -> dict[str, Any]:
    model_dir = Path(model_path)
    with (model_dir / "model.safetensors.index.json").open(
        "r", encoding="utf-8"
    ) as handle:
        index = json.load(handle)
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard in weight_map.items():
        by_shard[shard].append(tensor_name)

    expert_bytes: dict[tuple[int, int], int] = defaultdict(int)
    expert_components: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    dense_bytes = 0
    total_tensor_bytes = 0
    dtype_totals: dict[str, int] = defaultdict(int)

    for shard, names in sorted(by_shard.items()):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as archive:
            for tensor_name in names:
                tensor_slice = archive.get_slice(tensor_name)
                shape = [int(value) for value in tensor_slice.get_shape()]
                dtype = str(tensor_slice.get_dtype())
                try:
                    size = _numel(shape) * DTYPE_BYTES[dtype]
                except KeyError as error:
                    raise ValueError(f"unhandled safetensors dtype: {dtype}") from error
                total_tensor_bytes += size
                dtype_totals[dtype] += size
                match = EXPERT_PATTERN.match(tensor_name)
                if match is None:
                    dense_bytes += size
                    continue
                key = (int(match["layer"]), int(match["expert"]))
                expert_bytes[key] += size
                expert_components[key].append(
                    {
                        "name": tensor_name,
                        "projection": match["projection"],
                        "parameter": match["parameter"],
                        "shape": shape,
                        "dtype": dtype,
                        "bytes": size,
                        "shard": shard,
                    }
                )

    if not expert_bytes:
        raise ValueError("no Qwen MoE Expert tensors found in checkpoint")
    sizes = list(expert_bytes.values())
    layers = sorted({key[0] for key in expert_bytes})
    experts = sorted({key[1] for key in expert_bytes})
    component_names = sorted(
        {
            (item["projection"], item["parameter"])
            for values in expert_components.values()
            for item in values
        }
    )
    return {
        "model_path": str(model_dir),
        "index_total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "logical_tensor_bytes": total_tensor_bytes,
        "dense_nonexpert_tensor_bytes": dense_bytes,
        "all_expert_tensor_bytes": sum(sizes),
        "num_layers": len(layers),
        "num_experts_per_layer": len(experts),
        "num_expert_instances": len(expert_bytes),
        "expert_bytes_min": min(sizes),
        "expert_bytes_max": max(sizes),
        "expert_bytes_mean": sum(sizes) / len(sizes),
        "expert_bytes_uniform": len(set(sizes)) == 1,
        "expert_components": [
            {"projection": projection, "parameter": parameter}
            for projection, parameter in component_names
        ],
        "dtype_bytes": dict(sorted(dtype_totals.items())),
        "shard_count": len(by_shard),
    }

