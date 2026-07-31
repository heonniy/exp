from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.benchmark.measure_kv_bytes import logical_kv_bytes_per_token
from experiments.benchmark.memory_accounting import GIB, account_memory
from experiments.common.config import load_config

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the theoretical HBM/KV batch curve for every trace k."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--expert-layout", type=Path, required=True)
    parser.add_argument("--fixed-workspace-gib", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    environment = json.loads(args.environment.read_text())
    layout = json.loads(args.expert_layout.read_text())
    dtype = getattr(torch, config.model.dtype)
    model_config = json.loads((config.model.path / "config.json").read_text())
    kv_bytes = logical_kv_bytes_per_token(
        model_config["num_hidden_layers"],
        model_config["num_key_value_heads"],
        model_config["head_dim"],
        dtype,
    )
    rows = []
    for k in config.trace_k:
        memory = account_memory(
            total_hbm_bytes=int(environment["gpu_total_hbm_bytes"]),
            dense_resident_bytes=int(layout["dense_nonexpert_tensor_bytes"]),
            fixed_workspace_bytes=int(args.fixed_workspace_gib * GIB),
            safety_margin_bytes=int(config.runtime.hbm_safety_margin_gib * GIB),
            expert_bytes=int(layout["expert_bytes_mean"]),
            num_layers=config.model.num_moe_layers,
            k=k,
            transient_slots=config.runtime.transient_expert_slots,
            kv_bytes_per_token=kv_bytes,
            peak_sequence_length=config.peak_sequence_length,
        )
        rows.append(
            {
                "k": k,
                "policy_endpoint": (
                    "stream2"
                    if k == 0
                    else "full_resident"
                    if k == config.model.num_experts_per_layer
                    else "permanent_k|quota_lru_k"
                ),
                **memory.as_dict(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()

