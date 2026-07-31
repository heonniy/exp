from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.common.config import load_config
from experiments.common.io import atomic_write_json
from experiments.runtime.policies import (
    FullResidentPolicy,
    PermanentPolicy,
    QuotaLRUPolicy,
    Stream2Policy,
)
from experiments.trace.select_permanent import score_experts, select_topk_from_scores
from experiments.trace.simulator import SimulationResult, simulate
from experiments.trace.trace_schema import RoutingTrace


_WORKER_EVALUATION: RoutingTrace | None = None
_WORKER_EXPERT_BYTES = 0
_WORKER_BATCH_SIZE = 0
_WORKER_RETAIN_STATE = True

QUOTA_CONTROLS = (
    "ascending_always_admit",
    "resident_hit_first",
    "miss_bypass",
    "no_admission",
    "window_frequency",
    "random_order",
)


def _simulate_worker(policy) -> SimulationResult:
    if _WORKER_EVALUATION is None:
        raise RuntimeError("trace-sweep worker was not initialized")
    return simulate(
        _WORKER_EVALUATION,
        policy,
        _WORKER_EXPERT_BYTES,
        _WORKER_BATCH_SIZE,
        _WORKER_RETAIN_STATE,
    )


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


def _quota_control_policies(
    *,
    num_layers: int,
    num_experts: int,
    k: int,
    controls: tuple[str, ...],
    random_seeds: tuple[int, ...],
    window_size: int,
    window_min_frequency: int,
) -> list[QuotaLRUPolicy]:
    policies = []
    for control in controls:
        if control == "ascending_always_admit":
            policies.append(QuotaLRUPolicy(num_layers, num_experts, k))
        elif control == "resident_hit_first":
            policies.append(
                QuotaLRUPolicy(
                    num_layers,
                    num_experts,
                    k,
                    access_order="resident_hit_first",
                    name="quota_lru_resident_hit_first",
                )
            )
        elif control == "miss_bypass":
            policies.append(
                QuotaLRUPolicy(
                    num_layers,
                    num_experts,
                    k,
                    admission_policy="miss_bypass_when_full",
                    name="quota_lru_miss_bypass",
                )
            )
        elif control == "no_admission":
            policies.append(
                QuotaLRUPolicy(
                    num_layers,
                    num_experts,
                    k,
                    admission_policy="no_admission",
                    name="quota_lru_no_admission",
                )
            )
        elif control == "window_frequency":
            policies.append(
                QuotaLRUPolicy(
                    num_layers,
                    num_experts,
                    k,
                    admission_policy="window_frequency",
                    window_size=window_size,
                    window_min_frequency=window_min_frequency,
                    name="quota_lru_window_frequency",
                )
            )
        elif control == "random_order":
            for seed in random_seeds:
                policies.append(
                    QuotaLRUPolicy(
                        num_layers,
                        num_experts,
                        k,
                        access_order="random_expert_order",
                        random_seed=seed,
                        name=f"quota_lru_random_order_seed{seed}",
                    )
                )
        else:
            raise ValueError(f"unknown Quota-LRU control: {control}")
    return policies


def _load_measured_bmax(path: Path | None) -> dict[tuple[str, int], int]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (str(row["policy"]), int(row["k"])): int(row["measured_bmax"])
            for row in rows
        }


def _bmax_policy_name(policy: str) -> str:
    if policy.startswith("quota_lru_"):
        return "quota_lru_k"
    if policy.startswith("permanent_"):
        return "permanent_k"
    return policy


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
    workers: int = 1,
    quota_controls: tuple[str, ...] = ("ascending_always_admit",),
    random_seeds: tuple[int, ...] = (731, 732, 733),
    window_size: int = 8,
    window_min_frequency: int = 2,
) -> list[SimulationResult]:
    if evaluation.num_layers != calibration.num_layers:
        raise ValueError("calibration/evaluation trace layer counts do not match")
    num_layers = evaluation.num_layers
    num_experts = int(evaluation.metadata.get("num_experts", 128))
    permanent_scores = None
    if "permanent_k" in policy_names:
        selection_trace = evaluation if permanent_method == "oracle" else calibration
        selection_method = (
            "presence" if permanent_method == "oracle" else permanent_method
        )
        permanent_scores = score_experts(
            selection_trace,
            selection_method,
            batch_size=(
                batch_size
                if selection_method == "batch_step_union_presence"
                else None
            ),
        )
    policies_to_run = []
    for k in k_values:
        if k == 0:
            policies = [Stream2Policy(num_layers, num_experts)]
        elif k == num_experts:
            policies = [FullResidentPolicy(num_layers, num_experts)]
        else:
            policies = []
            if "permanent_k" in policy_names:
                assert permanent_scores is not None
                selected = select_topk_from_scores(permanent_scores, k)
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
                policies.extend(
                    _quota_control_policies(
                        num_layers=num_layers,
                        num_experts=num_experts,
                        k=k,
                        controls=quota_controls,
                        random_seeds=random_seeds,
                        window_size=window_size,
                        window_min_frequency=window_min_frequency,
                    )
                )
        for policy in policies:
            policies_to_run.append(policy)
    if workers <= 1:
        results = [
            simulate(
                evaluation,
                policy,
                expert_bytes,
                batch_size,
                retain_state_across_waves,
            )
            for policy in policies_to_run
        ]
    else:
        global _WORKER_EVALUATION
        global _WORKER_EXPERT_BYTES
        global _WORKER_BATCH_SIZE
        global _WORKER_RETAIN_STATE
        _WORKER_EVALUATION = evaluation
        _WORKER_EXPERT_BYTES = expert_bytes
        _WORKER_BATCH_SIZE = batch_size
        _WORKER_RETAIN_STATE = retain_state_across_waves
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(policies_to_run)),
            mp_context=context,
        ) as executor:
            results = list(executor.map(_simulate_worker, policies_to_run))
        _WORKER_EVALUATION = None
    for result in results:
        print(
            f"{result.policy} k={result.k}: hit={result.hit_rate:.4f}, "
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
        "--requests",
        type=int,
        help="Use the first N evaluation requests and record the derived trace digest.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        help="Override config trace_k for focused policy-control sweeps.",
    )
    parser.add_argument(
        "--permanent-method",
        choices=[
            "presence",
            "token_frequency",
            "batch_step_union_presence",
            "streaming_reload",
            "oracle",
        ],
        default="batch_step_union_presence",
        help="oracle selects from evaluation and is a non-deployable upper bound",
    )
    parser.add_argument(
        "--quota-controls",
        nargs="+",
        choices=QUOTA_CONTROLS,
        default=["ascending_always_admit"],
    )
    parser.add_argument(
        "--random-seeds", nargs="+", type=int, default=[731, 732, 733]
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-min-frequency", type=int, default=2)
    parser.add_argument(
        "--bmax-csv",
        type=Path,
        help=(
            "Annotate whether this simulated fixed batch is physically feasible; "
            "an infeasible row is explicitly labeled cache-upper-bound-only."
        ),
    )
    parser.add_argument(
        "--cold-each-wave",
        action="store_true",
        help="Reset adaptive quota state before every workload wave.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results/trace_sweep")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(20, os.cpu_count() or 1),
        help="Policy-level fork workers; trace arrays are shared copy-on-write.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluation = RoutingTrace.load(args.trace)
    if args.requests is not None:
        evaluation = evaluation.first_requests(args.requests)
    calibration = RoutingTrace.load(args.calibration_trace)
    results = run_sweep(
        evaluation=evaluation,
        calibration=calibration,
        k_values=(tuple(args.k_values) if args.k_values else config.trace_k),
        policy_names=config.policies,
        expert_bytes=args.expert_bytes,
        batch_size=args.batch_size,
        permanent_method=args.permanent_method,
        retain_state_across_waves=not args.cold_each_wave,
        workers=args.workers,
        quota_controls=tuple(args.quota_controls),
        random_seeds=tuple(args.random_seeds),
        window_size=args.window_size,
        window_min_frequency=args.window_min_frequency,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_summary_row(result) for result in results]
    measured_bmax = _load_measured_bmax(args.bmax_csv)
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
        bmax = measured_bmax.get(
            (_bmax_policy_name(str(summary["policy"])), int(summary["k"]))
        )
        summary["measured_bmax"] = bmax
        summary["physical_fixed_batch_feasible"] = (
            args.batch_size <= bmax if bmax is not None else None
        )
        summary["evidence_scope"] = (
            "cache_upper_bound_only"
            if bmax is not None and args.batch_size > bmax
            else (
                "cache_simulation_physical_batch_feasible"
                if bmax is not None
                else "cache_simulation_bmax_unknown"
            )
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
            "requests": evaluation.num_requests,
            "requested_prefix": args.requests,
            "k_values": args.k_values or list(config.trace_k),
            "permanent_method": args.permanent_method,
            "quota_controls": args.quota_controls,
            "random_seeds": args.random_seeds,
            "window_size": args.window_size,
            "window_min_frequency": args.window_min_frequency,
            "bmax_csv": str(args.bmax_csv) if args.bmax_csv else None,
            "cache_simulation_only": True,
            "results": summaries,
        },
    )


if __name__ == "__main__":
    main()
