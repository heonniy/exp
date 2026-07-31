from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from experiments.common.config import load_config
from experiments.common.io import atomic_write_json
from experiments.runtime.policies import (
    FullResidentPolicy,
    PermanentPolicy,
    QuotaLRUPolicy,
    Stream2Policy,
)
from experiments.trace.select_permanent import select_topk
from experiments.trace.simulator import SimulationResult, simulate
from experiments.trace.trace_schema import RoutingTrace


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _summary_row(result: SimulationResult) -> dict:
    value = result.as_dict()
    value.pop("per_layer")
    return value


def run_sweep(
    *,
    evaluation: RoutingTrace,
    calibration: RoutingTrace,
    k_values: tuple[int, ...],
    policy_names: tuple[str, ...],
    expert_bytes: int,
    batch_size: int,
    permanent_method: str,
    retain_state_across_waves: bool,
) -> list[SimulationResult]:
    if evaluation.num_layers != calibration.num_layers:
        raise ValueError("calibration/evaluation trace layer counts do not match")
    num_layers = evaluation.num_layers
    num_experts = int(evaluation.metadata.get("num_experts", 128))
    results: list[SimulationResult] = []
    for k in k_values:
        if k == 0:
            policies = [Stream2Policy(num_layers, num_experts)]
        elif k == num_experts:
            policies = [FullResidentPolicy(num_layers, num_experts)]
        else:
            policies = []
            if "permanent_k" in policy_names:
                selection_trace = (
                    evaluation if permanent_method == "oracle" else calibration
                )
                selection_method = (
                    "presence" if permanent_method == "oracle" else permanent_method
                )
                selected = select_topk(selection_trace, k, selection_method)
                policies.append(
                    PermanentPolicy(
                        num_layers,
                        num_experts,
                        k,
                        selected.tolist(),
                        name=(
                            "permanent_oracle"
                            if permanent_method == "oracle"
                            else "permanent_k"
                        ),
                    )
                )
            if "quota_lru_k" in policy_names:
                policies.append(QuotaLRUPolicy(num_layers, num_experts, k))
        for policy in policies:
            result = simulate(
                evaluation,
                policy,
                expert_bytes,
                batch_size,
                retain_state_across_waves,
            )
            results.append(result)
            print(
                f"{result.policy} k={k}: hit={result.hit_rate:.4f}, "
                f"fetches={result.fetches:,}",
                flush=True,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stream2/permanent/quota-LRU trace simulations."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--permanent-method",
        choices=["presence", "token_frequency", "streaming_reload", "oracle"],
        default="presence",
        help="oracle selects from evaluation and is a non-deployable upper bound",
    )
    parser.add_argument(
        "--cold-each-wave",
        action="store_true",
        help="Reset adaptive quota state before every workload wave.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results/trace_sweep")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation = RoutingTrace.load(args.trace)
    calibration = RoutingTrace.load(args.calibration_trace)
    results = run_sweep(
        evaluation=evaluation,
        calibration=calibration,
        k_values=config.trace_k,
        policy_names=config.policies,
        expert_bytes=args.expert_bytes,
        batch_size=args.batch_size,
        permanent_method=args.permanent_method,
        retain_state_across_waves=not args.cold_each_wave,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_summary_row(result) for result in results]
    stream2 = next(result for result in results if result.policy == "stream2")
    for summary in summaries:
        persistent_bytes = (
            evaluation.num_layers * int(summary["k"]) * args.expert_bytes
        )
        saved_bytes = stream2.h2d_bytes - int(summary["h2d_bytes"])
        summary["persistent_expert_bytes"] = persistent_bytes
        summary["h2d_reduction_vs_stream2"] = (
            saved_bytes / stream2.h2d_bytes if stream2.h2d_bytes else 0.0
        )
        summary["refetch_savings_per_reserved_gib"] = (
            saved_bytes / (1024**3) / (persistent_bytes / (1024**3))
            if persistent_bytes
            else None
        )
    per_layer = []
    stream2_layers = {
        int(row["layer_id"]): row for row in stream2.per_layer
    }
    for result in results:
        for row in result.per_layer:
            baseline_fetches = int(
                stream2_layers[int(row["layer_id"])]["fetches"]
            )
            per_layer.append(
                {
                    "policy": result.policy,
                    "k": result.k,
                    "batch_size": result.batch_size,
                    **row,
                    "hit_rate": (
                        int(row["hits"]) / int(row["accesses"])
                        if int(row["accesses"])
                        else 0.0
                    ),
                    "h2d_bytes": int(row["fetches"]) * args.expert_bytes,
                    "h2d_reduction_vs_stream2": (
                        1 - int(row["fetches"]) / baseline_fetches
                        if baseline_fetches
                        else 0.0
                    ),
                }
            )
    _atomic_csv(
        args.output_dir / "cache_summary.csv",
        list(summaries[0]),
        summaries,
    )
    _atomic_csv(
        args.output_dir / "per_layer.csv",
        list(per_layer[0]),
        per_layer,
    )
    atomic_write_json(
        args.output_dir / "summary.json",
        {
            "config": config.name,
            "evaluation_trace": str(args.trace),
            "calibration_trace": str(args.calibration_trace),
            "expert_bytes": args.expert_bytes,
            "batch_size": args.batch_size,
            "permanent_method": args.permanent_method,
            "results": summaries,
        },
    )


if __name__ == "__main__":
    main()
