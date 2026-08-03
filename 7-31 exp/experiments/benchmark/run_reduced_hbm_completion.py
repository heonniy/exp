from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.common.config import load_config
from experiments.common.io import atomic_write_json, git_sha


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _fixed_command(
    *,
    config: Path,
    workload: Path,
    calibration_trace: Path,
    evaluation_trace: Path,
    policy: str,
    k: int,
    batch_size: int,
    requests: int,
    decode_steps: int,
    output: Path,
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
        str(evaluation_trace),
        "--policy",
        policy,
        "--k",
        str(k),
        "--batch-size",
        str(batch_size),
        "--requests",
        str(requests),
        "--decode-steps",
        str(decode_steps),
        "--kv-setup",
        "real_prefill",
        "--minimum-steady-full-waves",
        "0",
        "--max-pinned-experts",
        "6144",
        "--prefetch-depth",
        "1",
        "--prefetch-submit-order",
        "compute_first",
        "--permanent-method",
        "batch_step_union_presence",
        "--output",
        str(output),
    ]
    if timeline_events:
        command.append("--timeline-events")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete a restartable reduced-HBM prefill+decode sweep with "
            "physical Bmax, uninstrumented runtime, and one-wave profiles."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--evaluation-trace", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--dense-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--phase",
        choices=["all", "bmax", "runtime", "profile"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" and not args.dry_run:
        raise RuntimeError("use ./scripts/gpu0.sh for the reduced HBM completion")
    if config.runtime.effective_hbm_gib is None:
        raise ValueError("reduced completion requires effective_hbm_gib")
    if args.output_dir is None:
        args.output_dir = (
            Path("experiments/results/by_commit")
            / git_sha()[:12]
            / config.name
        )
    provisional_dir = args.output_dir / "bmax_one_step_provisional"
    bmax_dir = args.output_dir / "bmax_prefill_decode"
    runtime_dir = args.output_dir / "runtime_at_bmax"
    profile_dir = args.output_dir / "profiles_at_bmax"
    for path in (provisional_dir, bmax_dir, runtime_dir, profile_dir):
        path.mkdir(parents=True, exist_ok=True)

    commands: list[dict] = []
    if args.phase in {"all", "bmax"}:
        provisional_command = [
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
            str(args.evaluation_trace),
            "--expert-bytes",
            str(args.expert_bytes),
            "--dense-bytes",
            str(args.dense_bytes),
            "--output-dir",
            str(provisional_dir),
        ]
        commands.append({"phase": "bmax_provisional", "command": provisional_command})
        _run(provisional_command, dry_run=args.dry_run)
        boundary_command = [
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
            str(args.evaluation_trace),
            "--provisional-dir",
            str(provisional_dir),
            "--output-dir",
            str(bmax_dir),
            "--decode-steps",
            str(config.dataset.output_tokens),
            "--include-prefill",
        ]
        commands.append({"phase": "bmax_boundary", "command": boundary_command})
        _run(boundary_command, dry_run=args.dry_run)

    if args.dry_run and not (bmax_dir / "manifest.json").exists():
        atomic_write_json(
            args.output_dir / "manifest.json",
            {
                "config": config.name,
                "dry_run": True,
                "commands": commands,
            },
        )
        return

    bmax_manifest = json.loads(
        (bmax_dir / "manifest.json").read_text(encoding="utf-8")
    )
    completed = [row for row in bmax_manifest["runs"] if row["completed"]]
    if args.phase in {"all", "runtime", "profile"} and not completed:
        raise ValueError("no completed Bmax runs are available")
    for row in completed:
        policy, k = str(row["policy"]), int(row["k"])
        bmax_result = json.loads(Path(row["output"]).read_text(encoding="utf-8"))
        batch_size = int(bmax_result["measured_bmax"])
        if args.phase in {"all", "runtime"}:
            output = runtime_dir / f"{policy}_k{k}_b{batch_size}_n{config.dataset.evaluation_requests}.json"
            command = _fixed_command(
                config=args.config,
                workload=args.workload,
                calibration_trace=args.calibration_trace,
                evaluation_trace=args.evaluation_trace,
                policy=policy,
                k=k,
                batch_size=batch_size,
                requests=config.dataset.evaluation_requests,
                decode_steps=config.dataset.output_tokens,
                output=output,
                timeline_events=False,
            )
            commands.append(
                {"phase": "runtime", "policy": policy, "k": k, "command": command}
            )
            if not output.exists():
                _run(command, dry_run=args.dry_run)
        if args.phase in {"all", "profile"}:
            profile_requests = min(batch_size, config.dataset.evaluation_requests)
            output = profile_dir / f"{policy}_k{k}_b{batch_size}_n{profile_requests}.json"
            command = _fixed_command(
                config=args.config,
                workload=args.workload,
                calibration_trace=args.calibration_trace,
                evaluation_trace=args.evaluation_trace,
                policy=policy,
                k=k,
                batch_size=batch_size,
                requests=profile_requests,
                decode_steps=config.dataset.output_tokens,
                output=output,
                timeline_events=True,
            )
            commands.append(
                {"phase": "profile", "policy": policy, "k": k, "command": command}
            )
            if not output.exists():
                _run(command, dry_run=args.dry_run)

    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "config": config.name,
            "effective_hbm_gib": config.runtime.effective_hbm_gib,
            "input_tokens": config.dataset.input_tokens,
            "decode_steps": config.dataset.output_tokens,
            "evaluation_requests": config.dataset.evaluation_requests,
            "policies": list(config.policies),
            "runtime_k_candidates": list(config.runtime_k),
            "provisional_dir": str(provisional_dir),
            "bmax_dir": str(bmax_dir),
            "runtime_dir": str(runtime_dir),
            "profile_dir": str(profile_dir),
            "dry_run": args.dry_run,
            "commands": commands,
        },
    )


if __name__ == "__main__":
    main()
