from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from experiments.benchmark.find_runtime_max_batch import (
    _detach_engine,
    probe_candidate,
)
from experiments.benchmark.measure_kv_bytes import logical_kv_bytes_per_token
from experiments.benchmark.memory_accounting import GIB, account_memory
from experiments.benchmark.run_offloaded_decode import _read_examples
from experiments.benchmark.run_runtime_sweep import configurations
from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import OffloadedExpertEngine, load_offloaded_qwen
from experiments.runtime.residency_manager import StreamingRuntimeManager
from experiments.trace.trace_schema import RoutingTrace


def _validated_inputs(
    workload: Path,
    trace: RoutingTrace,
    upper: int,
) -> tuple[np.ndarray, np.ndarray]:
    examples = _read_examples(workload, upper)
    workload_ids = [str(row["conversation_id"]) for row in examples]
    trace_ids = [str(value) for value in trace.conversation_ids[:upper]]
    if workload_ids != trace_ids:
        raise ValueError("forced routing trace does not match workload row order")
    workload_forced = np.asarray(
        [row["forced_output_ids"] for row in examples], dtype=np.int32
    )
    if not np.array_equal(
        trace.forced_output_ids[:upper, : workload_forced.shape[1]],
        workload_forced,
    ):
        raise ValueError("forced token IDs differ between trace and workload")
    token_ids = np.asarray(
        [row["forced_output_ids"][0] for row in examples], dtype=np.int64
    )
    return token_ids, trace.routing_expert_ids[:upper, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure every policy/k Bmax while reusing one packed pinned host "
            "store and one dense GPU model on physical GPU 0."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--forced-routing-trace", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--dense-bytes", type=int, required=True)
    parser.add_argument("--fixed-workspace-bytes", type=int, default=0)
    parser.add_argument(
        "--permanent-method",
        choices=[
            "presence",
            "token_frequency",
            "batch_step_union_presence",
            "streaming_reload",
        ],
        default="batch_step_union_presence",
    )
    parser.add_argument("--max-batch", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results/bmax")
    )
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("use ./scripts/gpu0.sh for the shared Bmax sweep")
    config = load_config(args.config)
    gpu = require_gpu0(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = list(
        configurations(config.runtime_k, config.model.num_experts_per_layer)
    )
    outputs = {
        (policy, k): args.output_dir / f"{policy}_k{k}.json"
        for policy, k in runs
    }
    pending = [pair for pair in runs if not outputs[pair].exists()]
    if not pending:
        print(json.dumps({"runs": len(runs), "completed": len(runs)}, indent=2))
        return

    model_config = AutoConfig.from_pretrained(
        config.model.path, local_files_only=True
    ).get_text_config(decoder=True)
    kv_bytes_per_token = logical_kv_bytes_per_token(
        config.model.num_moe_layers,
        model_config.num_key_value_heads,
        model_config.head_dim,
        torch.bfloat16,
    )
    accounting = {}
    for policy, k in pending:
        value = account_memory(
            total_hbm_bytes=gpu.total_memory,
            dense_resident_bytes=args.dense_bytes,
            fixed_workspace_bytes=args.fixed_workspace_bytes,
            safety_margin_bytes=int(config.runtime.hbm_safety_margin_gib * GIB),
            expert_bytes=args.expert_bytes,
            num_layers=config.model.num_moe_layers,
            k=k,
            transient_slots=config.runtime.transient_expert_slots,
            kv_bytes_per_token=kv_bytes_per_token,
            peak_sequence_length=config.peak_sequence_length,
        )
        accounting[(policy, k)] = value
    search_upper = {
        pair: min(
            accounting[pair].theoretical_bmax,
            args.max_batch
            if args.max_batch is not None
            else accounting[pair].theoretical_bmax,
        )
        for pair in pending
    }
    if any(value <= 0 for value in search_upper.values()):
        raise ValueError("theoretical Bmax is zero")

    trace = RoutingTrace.load(args.forced_routing_trace)
    token_ids, routing = _validated_inputs(
        args.workload, trace, max(search_upper.values())
    )
    calibration = RoutingTrace.load(args.calibration_trace)
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
        host_store, StreamingRuntimeManager(), prefetch_depth=1
    )
    model = load_offloaded_qwen(config.model.path, bootstrap)
    _detach_engine(model)
    del bootstrap
    torch.cuda.empty_cache()

    manifest_runs = []
    for policy, k in runs:
        manifest_runs.append(
            {
                "policy": policy,
                "k": k,
                "output": str(outputs[(policy, k)]),
                "completed": outputs[(policy, k)].exists(),
            }
        )
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "config": config.name,
            "gpu_physical_index": 0,
            "probe_mode": "real_runtime_static_peak_kv_one_decode_step",
            "shared_host_store_and_model": True,
            "cublas_workspaces_cleared_between_probes": True,
            "permanent_method": args.permanent_method,
            "runs": manifest_runs,
        },
    )

    for policy, k in pending:
        current_accounting = accounting[(policy, k)]
        upper = search_upper[(policy, k)]
        low, high = 0, upper
        probes: dict[int, dict] = {}
        while low < high:
            candidate = (low + high + 1) // 2
            value = probe_candidate(
                model=model,
                host_store=host_store,
                policy=policy,
                k=k,
                num_layers=config.model.num_moe_layers,
                num_experts=config.model.num_experts_per_layer,
                calibration=calibration,
                batch_size=candidate,
                peak_sequence_length=config.peak_sequence_length,
                safety_margin_bytes=current_accounting.safety_margin_bytes,
                token_ids=token_ids,
                routing=routing,
                permanent_method=args.permanent_method,
            )
            probes[candidate] = value
            print(
                f"policy={policy} k={k} candidate={candidate} "
                f"feasible={value['feasible']} "
                f"elapsed={value['elapsed_seconds']:.2f}s",
                flush=True,
            )
            if value["feasible"]:
                low = candidate
            else:
                high = candidate - 1
        result = {
            "probe_mode": "real_runtime_static_peak_kv_one_decode_step",
            "gpu_physical_index": 0,
            "config": config.name,
            "policy": policy,
            "k": k,
            "permanent_method": (
                args.permanent_method if policy == "permanent_k" else None
            ),
            **current_accounting.as_dict(),
            "measured_bmax": low,
            "search_upper_batch": upper,
            "user_max_batch_cap": args.max_batch,
            "search_truncated_below_theoretical": (
                upper < current_accounting.theoretical_bmax
            ),
            "forced_routing_trace": str(args.forced_routing_trace),
            "forced_routing_trace_sha256": trace.digest(),
            "host_memory_mode": "pinned_weights",
            "host_store_preload_seconds": preload_seconds,
            "host_store_preload_scope": "shared_bmax_sweep",
            "prefetch_submit_order": "compute_first",
            "cublas_workspaces_cleared_between_probes": True,
            "probes": [probes[key] for key in sorted(probes)],
        }
        atomic_write_json(outputs[(policy, k)], result)
        for row in manifest_runs:
            if row["policy"] == policy and row["k"] == k:
                row["completed"] = True
                break
        atomic_write_json(
            args.output_dir / "manifest.json",
            {
                "config": config.name,
                "gpu_physical_index": 0,
                "probe_mode": "real_runtime_static_peak_kv_one_decode_step",
                "shared_host_store_and_model": True,
                "cublas_workspaces_cleared_between_probes": True,
                "permanent_method": args.permanent_method,
                "runs": manifest_runs,
            },
        )

    print(
        json.dumps(
            {"runs": len(runs), "completed": len(runs), "output_dir": str(args.output_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
