from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.common.config import load_config
from experiments.common.io import atomic_write_json


def configurations(k_values: tuple[int, ...], num_experts: int):
    for k in k_values:
        if k == 0:
            yield "stream2", k
        elif k == num_experts:
            yield "full_resident", k
        else:
            yield "permanent_k", k
            yield "quota_lru_k", k


def resolve_batch_size(
    fixed_batch_size: int | None,
    bmax_dir: Path | None,
    policy: str,
    k: int,
) -> int:
    if fixed_batch_size is not None:
        return fixed_batch_size
    if bmax_dir is None:
        raise ValueError("either fixed batch size or Bmax directory is required")
    path = bmax_dir / f"{policy}_k{k}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    batch_size = int(value["measured_bmax"])
    if value.get("policy") != policy or int(value.get("k", -1)) != k:
        raise ValueError(f"Bmax result identity mismatch: {path}")
    if batch_size <= 0:
        raise ValueError(f"measured Bmax is not positive: {path}")
    return batch_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase-A H100 offloaded decode sweep sequentially."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--forced-routing-trace", type=Path, required=True)
    batch = parser.add_mutually_exclusive_group(required=True)
    batch.add_argument("--batch-size", type=int)
    batch.add_argument(
        "--bmax-dir",
        type=Path,
        help="Use each policy/k measured_bmax from this result directory.",
    )
    parser.add_argument("--decode-steps", type=int)
    parser.add_argument(
        "--kv-setup",
        choices=["real_prefill", "static_zero"],
        default="static_zero",
    )
    parser.add_argument("--prefetch-depth", choices=[0, 1], type=int, default=1)
    parser.add_argument("--timeline-events", action="store_true")
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results/runtime_sweep")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" and not args.dry_run:
        raise RuntimeError("use ./scripts/gpu0.sh to run the runtime sweep")

    commands = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for policy, k in configurations(
        config.runtime_k, config.model.num_experts_per_layer
    ):
        batch_size = resolve_batch_size(
            args.batch_size, args.bmax_dir, policy, k
        )
        output = args.output_dir / f"{policy}_k{k}_b{batch_size}.json"
        command = [
            sys.executable,
            "-m",
            "experiments.benchmark.run_offloaded_decode",
            "--config",
            str(args.config),
            "--workload",
            str(args.workload),
            "--policy",
            policy,
            "--k",
            str(k),
            "--batch-size",
            str(batch_size),
            "--calibration-trace",
            str(args.calibration_trace),
            "--forced-routing-trace",
            str(args.forced_routing_trace),
            "--host-memory-mode",
            "pinned_weights",
            "--max-pinned-experts",
            str(config.model.num_moe_layers * config.model.num_experts_per_layer),
            "--kv-setup",
            args.kv_setup,
            "--prefetch-depth",
            str(args.prefetch_depth),
            "--permanent-method",
            args.permanent_method,
            "--output",
            str(output),
        ]
        if args.decode_steps is not None:
            command.extend(["--decode-steps", str(args.decode_steps)])
        if args.timeline_events:
            command.append("--timeline-events")
        commands.append(
            {
                "policy": policy,
                "k": k,
                "batch_size": batch_size,
                "output": str(output),
                "command": command,
            }
        )
        if not args.dry_run and not output.exists():
            subprocess.run(command, check=True)
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "config": config.name,
            "gpu_physical_index": 0,
            "batch_mode": (
                "fixed_batch" if args.batch_size is not None else "measured_bmax"
            ),
            "fixed_batch_size": args.batch_size,
            "bmax_dir": str(args.bmax_dir) if args.bmax_dir is not None else None,
            "decode_steps": args.decode_steps or config.dataset.output_tokens,
            "kv_setup": args.kv_setup,
            "prefetch_depth": args.prefetch_depth,
            "timeline_events": args.timeline_events,
            "permanent_method": args.permanent_method,
            "dry_run": args.dry_run,
            "runs": commands,
        },
    )
    print(json.dumps({"runs": len(commands), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
