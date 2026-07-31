from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.benchmark.run_runtime_sweep import (
    configurations,
    resolve_batch_size,
)
from experiments.common.config import load_config
from experiments.common.io import atomic_write_json, git_sha
from experiments.trace.trace_schema import RoutingTrace


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _result_paths(
    bmax_dir: Path, k_values: tuple[int, ...], num_experts: int
) -> list[Path]:
    return [
        bmax_dir / f"{policy}_k{k}.json"
        for policy, k in configurations(k_values, num_experts)
    ]


def _fixed_command(
    *,
    config: Path,
    workload: Path,
    calibration_trace: Path,
    forced_routing_trace: Path,
    policy: str,
    k: int,
    batch_size: int,
    requests: int,
    decode_steps: int,
    output: Path,
    permanent_method: str,
    timeline_events: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.benchmark.run_fixed_workload",
        "--config",
        str(config),
        "--workload",
        str(workload),
        "--calibration-trace",
        str(calibration_trace),
        "--forced-routing-trace",
        str(forced_routing_trace),
        "--policy",
        policy,
        "--k",
        str(k),
        "--batch-size",
        str(batch_size),
        "--decode-steps",
        str(decode_steps),
        "--requests",
        str(requests),
        "--prefetch-depth",
        "1",
        "--prefetch-submit-order",
        "compute_first",
        "--permanent-method",
        permanent_method,
        "--max-pinned-experts",
        "6144",
        "--output",
        str(output),
    ]
    if timeline_events:
        command.append("--timeline-events")
    return command


def _write_manifest(path: Path, payload: dict, runs: list[dict]) -> None:
    value = {
        **payload,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {**run, "completed": Path(run["output"]).exists()} for run in runs
        ],
    }
    atomic_write_json(path, value)


def resolve_common_batch_size(requested: int, minimum_bmax: int) -> tuple[int, bool]:
    if requested <= minimum_bmax:
        return requested, False
    if requested == 40 and minimum_bmax >= 32:
        return 32, True
    raise ValueError(
        f"requested common B={requested} exceeds the minimum measured "
        f"Bmax={minimum_bmax}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete the strict 4K/256 Bmax, measured-Bmax workload, and "
            "physical common-batch workload on physical GPU 0."
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
        "--calibration-trace",
        type=Path,
        default=Path("artifacts/traces/calibration_4k256.npz"),
    )
    parser.add_argument(
        "--forced-routing-trace",
        type=Path,
        default=Path("artifacts/traces/evaluation_4k256.npz"),
    )
    parser.add_argument("--requests", type=int, default=1200)
    parser.add_argument("--common-batch-size", type=int, default=40)
    parser.add_argument("--expert-bytes", type=int, default=9437184)
    parser.add_argument("--dense-bytes", type=int, default=3082186752)
    parser.add_argument("--fixed-workspace-bytes", type=int, default=0)
    parser.add_argument(
        "--permanent-method",
        choices=["batch_step_union_presence", "token_frequency"],
        default="batch_step_union_presence",
    )
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "bmax",
            "runtime_at_bmax",
            "common_batch",
            "profile_common_batch",
        ],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("use ./scripts/gpu0.sh for the completion sweep")
    if config.dataset.input_tokens != 4096 or config.dataset.output_tokens != 256:
        raise ValueError("completion sweep requires the 4K/256 configuration")
    if args.requests != config.dataset.evaluation_requests:
        raise ValueError(
            "--requests must equal the explicit validation size in the config"
        )
    if args.common_batch_size <= 0:
        raise ValueError("common batch size must be positive")
    trace = RoutingTrace.load(args.forced_routing_trace)
    trace.validate(config.model.num_experts_per_layer, require_weights=True)
    trace.require_serial_reference()
    if args.requests > trace.num_requests:
        raise ValueError("validation request count exceeds the forced trace")

    if args.output_dir is None:
        args.output_dir = (
            Path("experiments/results/by_commit")
            / git_sha()[:12]
            / "completion_4k256_1200"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bmax_dir = args.output_dir / "bmax"
    bmax_dir.mkdir(parents=True, exist_ok=True)
    provisional_bmax_dir = args.output_dir / "bmax_one_step_provisional"
    bmax_paths = _result_paths(
        bmax_dir, config.runtime_k, config.model.num_experts_per_layer
    )

    if args.phase in {"all", "bmax"}:
        _run(
            [
                sys.executable,
                "-m",
                "experiments.benchmark.run_shared_bmax_sweep",
                "--config",
                str(args.config),
                "--workload",
                str(args.workload),
                "--calibration-trace",
                str(args.calibration_trace),
                "--forced-routing-trace",
                str(args.forced_routing_trace),
                "--expert-bytes",
                str(args.expert_bytes),
                "--dense-bytes",
                str(args.dense_bytes),
                "--fixed-workspace-bytes",
                str(args.fixed_workspace_bytes),
                "--permanent-method",
                args.permanent_method,
                "--output-dir",
                str(provisional_bmax_dir),
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "experiments.benchmark.validate_bmax_256",
                "--config",
                str(args.config),
                "--workload",
                str(args.workload),
                "--calibration-trace",
                str(args.calibration_trace),
                "--forced-routing-trace",
                str(args.forced_routing_trace),
                "--provisional-dir",
                str(provisional_bmax_dir),
                "--output-dir",
                str(bmax_dir),
                "--decode-steps",
                "256",
                "--permanent-method",
                args.permanent_method,
            ]
        )
        _run(
            [
                sys.executable,
                "-m",
                "experiments.analysis.aggregate_bmax",
                *[str(path) for path in bmax_paths],
                "--output",
                str(bmax_dir / "bmax.csv"),
            ]
        )
    missing_bmax = [str(path) for path in bmax_paths if not path.exists()]
    if args.phase != "bmax" and missing_bmax:
        raise RuntimeError(f"missing {len(missing_bmax)} Bmax results")

    requested_common_batch_size = args.common_batch_size
    common_batch_fallback_applied = False
    if not missing_bmax:
        minimum_bmax = min(
            resolve_batch_size(None, bmax_dir, policy, k)
            for policy, k in configurations(
                config.runtime_k, config.model.num_experts_per_layer
            )
        )
        args.common_batch_size, common_batch_fallback_applied = (
            resolve_common_batch_size(args.common_batch_size, minimum_bmax)
        )
    else:
        minimum_bmax = None

    manifest_base = {
        "config": config.name,
        "gpu_physical_index": 0,
        "validation_requests": args.requests,
        "input_tokens": 4096,
        "decode_steps": 256,
        "requested_common_batch_size": requested_common_batch_size,
        "common_batch_size": args.common_batch_size,
        "minimum_measured_bmax": minimum_bmax,
        "common_batch_fallback_applied": common_batch_fallback_applied,
        "common_batch_selection_rule": (
            "prefer B=40; fall back to the pre-approved B=32 when any "
            "measured Bmax is below 40"
        ),
        "permanent_method": args.permanent_method,
        "quota_runtime_control": "ascending_id_always_admit_lru",
        "performance_timeline_events": False,
        "profile_timeline_events": True,
        "profile_requests": args.common_batch_size * 2,
        "profile_interpretation": (
            "instrumented two-wave common-batch breakdown; excluded from "
            "throughput and makespan comparisons"
        ),
        "prefetch_submit_order": "compute_first",
        "cold_start_and_steady_state_separated": True,
        "forced_routing_trace": str(args.forced_routing_trace),
        "forced_routing_trace_sha256": trace.digest(),
    }
    all_runs: list[dict] = []
    if args.phase in {"all", "runtime_at_bmax"}:
        output_dir = args.output_dir / "runtime_at_bmax"
        output_dir.mkdir(parents=True, exist_ok=True)
        for policy, k in configurations(
            config.runtime_k, config.model.num_experts_per_layer
        ):
            batch_size = resolve_batch_size(None, bmax_dir, policy, k)
            output = output_dir / (
                f"{policy}_k{k}_b{batch_size}_n{args.requests}.json"
            )
            command = _fixed_command(
                config=args.config,
                workload=args.workload,
                calibration_trace=args.calibration_trace,
                forced_routing_trace=args.forced_routing_trace,
                policy=policy,
                k=k,
                batch_size=batch_size,
                requests=args.requests,
                decode_steps=256,
                output=output,
                permanent_method=args.permanent_method,
                timeline_events=False,
            )
            run = {
                "mode": "measured_bmax",
                "policy": policy,
                "k": k,
                "batch_size": batch_size,
                "output": str(output),
                "command": command,
            }
            all_runs.append(run)
            _write_manifest(args.output_dir / "manifest.json", manifest_base, all_runs)
            if not output.exists():
                _run(command)
                _write_manifest(
                    args.output_dir / "manifest.json", manifest_base, all_runs
                )

    if args.phase in {"all", "common_batch"}:
        output_dir = args.output_dir / f"runtime_common_b{args.common_batch_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for policy, k in configurations(
            config.runtime_k, config.model.num_experts_per_layer
        ):
            measured_bmax = resolve_batch_size(None, bmax_dir, policy, k)
            if args.common_batch_size > measured_bmax:
                raise ValueError(
                    f"common B={args.common_batch_size} exceeds measured Bmax "
                    f"{measured_bmax} for {policy} k={k}"
                )
            output = output_dir / (
                f"{policy}_k{k}_b{args.common_batch_size}_n{args.requests}.json"
            )
            command = _fixed_command(
                config=args.config,
                workload=args.workload,
                calibration_trace=args.calibration_trace,
                forced_routing_trace=args.forced_routing_trace,
                policy=policy,
                k=k,
                batch_size=args.common_batch_size,
                requests=args.requests,
                decode_steps=256,
                output=output,
                permanent_method=args.permanent_method,
                timeline_events=False,
            )
            run = {
                "mode": "common_fixed_batch",
                "policy": policy,
                "k": k,
                "batch_size": args.common_batch_size,
                "measured_bmax": measured_bmax,
                "physical_fixed_batch_feasible": True,
                "output": str(output),
                "command": command,
            }
            all_runs.append(run)
            _write_manifest(args.output_dir / "manifest.json", manifest_base, all_runs)
            if not output.exists():
                _run(command)
                _write_manifest(
                    args.output_dir / "manifest.json", manifest_base, all_runs
                )

    if args.phase in {"all", "profile_common_batch"}:
        output_dir = args.output_dir / (
            f"profile_common_b{args.common_batch_size}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_requests = args.common_batch_size * 2
        for policy, k in configurations(
            config.runtime_k, config.model.num_experts_per_layer
        ):
            measured_bmax = resolve_batch_size(None, bmax_dir, policy, k)
            if args.common_batch_size > measured_bmax:
                raise ValueError(
                    f"common B={args.common_batch_size} exceeds measured Bmax "
                    f"{measured_bmax} for {policy} k={k}"
                )
            output = output_dir / (
                f"{policy}_k{k}_b{args.common_batch_size}_n{profile_requests}.json"
            )
            command = _fixed_command(
                config=args.config,
                workload=args.workload,
                calibration_trace=args.calibration_trace,
                forced_routing_trace=args.forced_routing_trace,
                policy=policy,
                k=k,
                batch_size=args.common_batch_size,
                requests=profile_requests,
                decode_steps=256,
                output=output,
                permanent_method=args.permanent_method,
                timeline_events=True,
            )
            run = {
                "mode": "instrumented_profile_common_fixed_batch",
                "policy": policy,
                "k": k,
                "batch_size": args.common_batch_size,
                "requests": profile_requests,
                "waves": 2,
                "decode_steps": 256,
                "timeline_events": True,
                "excluded_from_throughput_comparison": True,
                "output": str(output),
                "command": command,
            }
            all_runs.append(run)
            _write_manifest(args.output_dir / "manifest.json", manifest_base, all_runs)
            if not output.exists():
                _run(command)
                _write_manifest(
                    args.output_dir / "manifest.json", manifest_base, all_runs
                )

    _write_manifest(args.output_dir / "manifest.json", manifest_base, all_runs)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "runs": len(all_runs),
                "completed": sum(Path(run["output"]).exists() for run in all_runs),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
