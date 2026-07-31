from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.benchmark.run_runtime_sweep import configurations
from experiments.common.config import load_config
from experiments.common.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-runtime Bmax probes sequentially on physical GPU 0."
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" and not args.dry_run:
        raise RuntimeError("use ./scripts/gpu0.sh to run the Bmax sweep")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for policy, k in configurations(
        config.runtime_k, config.model.num_experts_per_layer
    ):
        output = args.output_dir / f"{policy}_k{k}.json"
        command = [
            sys.executable,
            "-m",
            "experiments.benchmark.find_runtime_max_batch",
            "--config",
            str(args.config),
            "--workload",
            str(args.workload),
            "--calibration-trace",
            str(args.calibration_trace),
            "--forced-routing-trace",
            str(args.forced_routing_trace),
            "--policy",
            policy,
            "--k",
            str(k),
            "--expert-bytes",
            str(args.expert_bytes),
            "--dense-bytes",
            str(args.dense_bytes),
            "--fixed-workspace-bytes",
            str(args.fixed_workspace_bytes),
            "--permanent-method",
            args.permanent_method,
            "--max-pinned-experts",
            str(config.model.num_moe_layers * config.model.num_experts_per_layer),
            "--output",
            str(output),
        ]
        if args.max_batch is not None:
            command.extend(["--max-batch", str(args.max_batch)])
        run = {"policy": policy, "k": k, "output": str(output), "command": command}
        runs.append(run)
        if not args.dry_run and not output.exists():
            subprocess.run(command, check=True)
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "config": config.name,
            "gpu_physical_index": 0,
            "dry_run": args.dry_run,
            "probe_mode": "real_runtime_static_peak_kv_one_decode_step",
            "permanent_method": args.permanent_method,
            "runs": runs,
        },
    )
    print(json.dumps({"runs": len(runs), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
