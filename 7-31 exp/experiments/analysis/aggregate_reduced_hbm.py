from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from experiments.common.config import load_config
from experiments.common.io import atomic_write_json


GIB = 1024**3


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def _runtime_path(directory: Path, policy: str, k: int, batch: int, requests: int) -> Path:
    return directory / f"{policy}_k{k}_b{batch}_n{requests}.json"


def _profile_path(directory: Path, policy: str, k: int, batch: int) -> Path:
    return directory / f"{policy}_k{k}_b{batch}_n{batch}.json"


def _validate_bmax(value: dict, *, policy: str, k: int, cap_bytes: int) -> None:
    _require((value["policy"], int(value["k"])) == (policy, k), "Bmax identity mismatch")
    _require(value["probe_mode"] == "real_runtime_prefill_and_full_decode_boundary", "Bmax did not include real prefill and full decode")
    _require(value["includes_real_prefill"] is True, "Bmax omitted real prefill")
    _require(value["boundary_closed"] is True, "Bmax boundary is open")
    _require(value["allocator_hbm_cap_enforced"] is True, "Bmax allocator cap was not enforced")
    _require(int(value["effective_hbm_cap_bytes"]) == cap_bytes, "Bmax HBM cap mismatch")
    _require(int(value["decode_steps"]) == 128, "Bmax decode length mismatch")
    _require(int(value["gpu_physical_index"]) == 0, "Bmax did not use physical GPU 0")


def _validate_fixed(
    value: dict,
    *,
    policy: str,
    k: int,
    batch: int,
    requests: int,
    cap_bytes: int,
    instrumented: bool,
) -> None:
    kind = "profile" if instrumented else "runtime"
    _require((value["policy"], int(value["k"])) == (policy, k), f"{kind} identity mismatch")
    _require(int(value["batch_size"]) == batch, f"{kind} batch mismatch")
    _require(int(value["requests"]) == requests, f"{kind} request count mismatch")
    _require(int(value["decode_steps"]) == 128, f"{kind} decode length mismatch")
    _require(value["kv_setup"] == "real_prefill", f"{kind} did not execute real prefill")
    _require(value["prefetch_submit_order"] == "compute_first", f"{kind} did not use compute-first prefetch")
    _require(value["forced_routing_weight_source"] == "recorded_trace_weights", f"{kind} did not replay recorded router weights")
    _require(value["forced_routing_weight_alignment_caveat"] is False, f"{kind} reports a routing-weight alignment caveat")
    _require(value["trace_sha256"] == value["forced_routing_trace_sha256"], f"{kind} trace digests disagree")
    _require(value["allocator_hbm_cap_enforced"] is True, f"{kind} allocator cap was not enforced")
    _require(int(value["effective_hbm_cap_bytes"]) == cap_bytes, f"{kind} HBM cap mismatch")
    _require(int(value["peak_reserved_bytes"]) <= cap_bytes, f"{kind} exceeded HBM cap")
    _require(int(value["gpu_physical_index"]) == 0, f"{kind} did not use physical GPU 0")
    _require(value["timeline_events_enabled"] is instrumented, f"{kind} timeline flag mismatch")
    _require(value["instrumented_profile_only"] is instrumented, f"{kind} profile-only flag mismatch")
    _require(value["eligible_for_throughput_and_makespan_comparison"] is (not instrumented), f"{kind} performance eligibility mismatch")
    expected_interpretation = (
        "intrusive_component_profile_not_performance_evidence"
        if instrumented
        else "uninstrumented_performance_run"
    )
    _require(value["timing_interpretation"] == expected_interpretation, f"{kind} timing interpretation mismatch")
    _require(int(value["expert_h2d_copy_operations"]) == int(value["expert_h2d_fetches"]), f"{kind} decode copy/fetch ratio is not one")
    _require(int(value["prefill_expert_h2d_copy_operations"]) == int(value["prefill_expert_h2d_fetches"]), f"{kind} prefill copy/fetch ratio is not one")
    _require(_close(float(value["fixed_workload_e2e_makespan_seconds"]), float(value["fixed_workload_prefill_makespan_seconds"]) + float(value["fixed_workload_decode_makespan_seconds"])), f"{kind} E2E does not equal prefill + decode")
    _require(int(value["prompt_tokens"]) == requests * 4096, f"{kind} prompt token count mismatch")
    _require(int(value["generated_tokens"]) == requests * 128, f"{kind} generated token count mismatch")
    _require(len(value["wave_results"]) == math.ceil(requests / batch), f"{kind} wave count mismatch")
    if policy == "permanent_k":
        _require(value["permanent_method"] == "batch_step_union_presence", f"{kind} Permanent scoring mismatch")
        _require(int(value["prefill_permanent_hits"]) > 0, f"{kind} Permanent buffer was not used during prefill")


def _profile_metrics(value: dict, prefix: str) -> dict:
    field_prefix = "prefill_" if prefix == "prefill" else ""
    total = float(value[f"{field_prefix}total_h2d_duration_ms"])
    exposed = float(value[f"{field_prefix}exposed_h2d_stall_ms"])
    overlap = float(value[f"{field_prefix}overlapped_h2d_ms"])
    _require(_close(total, exposed + overlap), f"{prefix} H2D decomposition does not close")
    return {
        f"{prefix}_profile_h2d_total_ms": total,
        f"{prefix}_profile_h2d_exposed_ms": exposed,
        f"{prefix}_profile_h2d_overlap_ms": overlap,
        f"{prefix}_profile_h2d_overlap_pct": 100.0 * overlap / total if total else 0.0,
        f"{prefix}_profile_compute_ms": float(value[f"{field_prefix}expert_compute_ms"]),
        f"{prefix}_profile_attention_ms": float(value[f"{field_prefix}attention_ms"]),
        f"{prefix}_profile_router_ms": float(value[f"{field_prefix}router_ms"]),
        f"{prefix}_profile_other_ms": float(value[f"{field_prefix}other_dense_host_idle_ms"]),
    }


def _normalized_profile_metrics(value: dict, prefix: str, tokens: int) -> dict:
    raw = _profile_metrics(value, prefix)
    result = {}
    for component in ("h2d_total", "h2d_exposed", "h2d_overlap", "compute", "attention", "router", "other"):
        result[f"{prefix}_profile_{component}_us_per_token"] = (
            1000.0 * raw[f"{prefix}_profile_{component}_ms"] / tokens
        )
    return result


def _row(bmax: dict, runtime: dict, profile: dict) -> dict:
    waves = runtime["wave_results"]
    warmup = waves[0]
    full_steady_decode = [
        float(wave["decode_wall_seconds"])
        for wave in waves[1:]
        if int(wave["batch_size"]) == int(runtime["batch_size"])
    ]
    return {
        "policy": runtime["policy"],
        "k": int(runtime["k"]),
        "measured_bmax": int(bmax["measured_bmax"]),
        "waves": int(runtime["waves"]),
        "peak_reserved_gib": int(runtime["peak_reserved_bytes"]) / GIB,
        "prefill_seconds": float(runtime["fixed_workload_prefill_makespan_seconds"]),
        "prefill_prompt_tokens_per_second": float(runtime["prefill_prompt_tokens_per_second"]),
        "prefill_requests_per_second": int(runtime["requests"]) / float(runtime["fixed_workload_prefill_makespan_seconds"]),
        "decode_seconds": float(runtime["fixed_workload_decode_makespan_seconds"]),
        "decode_tokens_per_second": float(runtime["fixed_workload_tokens_per_second"]),
        "e2e_seconds": float(runtime["fixed_workload_e2e_makespan_seconds"]),
        "e2e_requests_per_second": float(runtime["e2e_requests_per_second"]),
        "e2e_total_tokens_per_second": float(runtime["e2e_total_tokens_per_second"]),
        "cold_initialization_seconds": float(runtime["cold_start_seconds"]),
        "warmup_wave_prefill_seconds": float(warmup["prefill_wall_seconds"]),
        "warmup_wave_decode_seconds": float(warmup["decode_wall_seconds"]),
        "warmup_wave_e2e_seconds": float(warmup["e2e_wall_seconds"]),
        "steady_full_wave_count": len(full_steady_decode),
        "steady_full_wave_decode_median_seconds": float(runtime["steady_state_full_wave_seconds_median"]) if runtime["steady_state_full_wave_seconds_median"] is not None else None,
        "prefill_h2d_fetches": int(runtime["prefill_expert_h2d_fetches"]),
        "decode_h2d_fetches": int(runtime["expert_h2d_fetches"]),
        "prefill_permanent_hits": int(runtime["prefill_permanent_hits"]),
        "decode_permanent_hits": int(runtime["permanent_hits"]),
        "decode_natural_route_set_mismatch_pct": 100.0 * float(runtime["natural_route_set_mismatch_rate"]),
        "profile_batch_size": int(profile["batch_size"]),
        "prefill_profile_wall_seconds": float(profile["fixed_workload_prefill_makespan_seconds"]),
        "decode_profile_wall_seconds": float(profile["fixed_workload_decode_makespan_seconds"]),
        **_profile_metrics(profile, "prefill"),
        **_normalized_profile_metrics(profile, "prefill", int(profile["prompt_tokens"])),
        **_profile_metrics(profile, "decode"),
        **_normalized_profile_metrics(profile, "decode", int(profile["generated_tokens"])),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    ks = [row["k"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(ks, [row["measured_bmax"] for row in rows], marker="o")
    axes[0, 0].set(title="Physical Bmax under 40 GiB", xlabel="Permanent experts/layer (k)", ylabel="Bmax")
    phase_axis = axes[0, 1]
    prefill_axis = phase_axis.twinx()
    phase_lines = phase_axis.plot(ks, [row["decode_seconds"] for row in rows], marker="o", color="tab:orange", label="decode")
    phase_lines += phase_axis.plot(ks, [row["e2e_seconds"] for row in rows], marker="o", color="tab:green", label="E2E")
    phase_lines += prefill_axis.plot(ks, [row["prefill_seconds"] for row in rows], marker="s", linestyle="--", color="tab:blue", label="prefill")
    phase_axis.set(title="200-request phase and E2E makespan", xlabel="k", ylabel="decode / E2E seconds")
    prefill_axis.set_ylabel("prefill seconds", color="tab:blue")
    phase_axis.legend(phase_lines, [line.get_label() for line in phase_lines], loc="upper left")

    throughput_axis = axes[1, 0]
    prompt_axis = throughput_axis.twinx()
    throughput_lines = throughput_axis.plot(ks, [row["decode_tokens_per_second"] for row in rows], marker="o", color="tab:green", label="decode")
    throughput_lines += prompt_axis.plot(ks, [row["prefill_prompt_tokens_per_second"] for row in rows], marker="s", linestyle="--", color="tab:blue", label="prefill")
    throughput_axis.set(title="Phase throughput", xlabel="k", ylabel="generated tokens/s")
    prompt_axis.set_ylabel("prompt tokens/s", color="tab:blue")
    throughput_axis.legend(throughput_lines, [line.get_label() for line in throughput_lines], loc="lower left")

    fetch_axis = axes[1, 1]
    prefill_fetch_axis = fetch_axis.twinx()
    fetch_lines = fetch_axis.plot(ks, [row["decode_h2d_fetches"] for row in rows], marker="o", color="tab:orange", label="decode")
    fetch_lines += prefill_fetch_axis.plot(ks, [row["prefill_h2d_fetches"] for row in rows], marker="s", linestyle="--", color="tab:blue", label="prefill")
    fetch_axis.set(title="Expert H2D fetches", xlabel="k", ylabel="decode fetches")
    prefill_fetch_axis.set_ylabel("prefill fetches", color="tab:blue")
    fetch_axis.legend(fetch_lines, [line.get_label() for line in fetch_lines], loc="upper right")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout(pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and aggregate the 40 GiB 4K/128 reduced sweep.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    _require(config.runtime.effective_hbm_gib == 40.0, "config is not the 40 GiB sweep")
    _require(config.dataset.input_tokens == 4096, "config is not 4K prefill")
    _require(config.dataset.output_tokens == 128, "config is not 128 decode")
    _require(config.dataset.evaluation_requests == 200, "config is not the 200-request workload")
    _require(tuple(config.policies) == ("stream2", "permanent_k"), "unexpected policies")
    cap_bytes = int(config.runtime.effective_hbm_gib * GIB)

    bmax_dir = args.result_dir / "bmax_prefill_decode"
    runtime_dir = args.result_dir / "runtime_at_bmax"
    profile_dir = args.result_dir / "profiles_at_bmax"
    bmax_manifest = _load(bmax_dir / "manifest.json")
    _require(bmax_manifest["includes_real_prefill"] is True, "Bmax manifest omitted prefill")
    _require(int(bmax_manifest["effective_hbm_cap_bytes"]) == cap_bytes, "manifest cap mismatch")
    skipped = bmax_manifest["skipped_infeasible_candidates"]
    _require([(item["k"], item["policy"]) for item in skipped] == [(96, "permanent_k"), (128, "permanent_k")], "unexpected infeasible candidates")

    rows: list[dict] = []
    runtime_trace_digests: set[str] = set()
    runtime_output_digests: set[str] = set()
    for run in bmax_manifest["runs"]:
        _require(run["completed"] is True, "incomplete Bmax configuration")
        policy, k = str(run["policy"]), int(run["k"])
        bmax = _load(bmax_dir / f"{policy}_k{k}.json")
        _validate_bmax(bmax, policy=policy, k=k, cap_bytes=cap_bytes)
        batch = int(bmax["measured_bmax"])
        runtime = _load(_runtime_path(runtime_dir, policy, k, batch, config.dataset.evaluation_requests))
        profile = _load(_profile_path(profile_dir, policy, k, batch))
        _validate_fixed(runtime, policy=policy, k=k, batch=batch, requests=config.dataset.evaluation_requests, cap_bytes=cap_bytes, instrumented=False)
        _validate_fixed(profile, policy=policy, k=k, batch=batch, requests=batch, cap_bytes=cap_bytes, instrumented=True)
        runtime_trace_digests.add(str(runtime["trace_sha256"]))
        runtime_output_digests.add(str(runtime["forced_output_ids_sha256"]))
        rows.append(_row(bmax, runtime, profile))

    _require(len(rows) == 9, "expected nine feasible configurations")
    _require(len(runtime_trace_digests) == 1, "runtime trace digests do not match")
    _require(len(runtime_output_digests) == 1, "runtime forced-output digests do not match")
    rows.sort(key=lambda item: int(item["k"]))
    baseline = rows[0]
    for row in rows:
        row["e2e_speedup_vs_stream2_pct"] = 100.0 * (baseline["e2e_seconds"] - row["e2e_seconds"]) / baseline["e2e_seconds"]
        row["decode_throughput_gain_vs_stream2_pct"] = 100.0 * (row["decode_tokens_per_second"] / baseline["decode_tokens_per_second"] - 1.0)
    best = min(rows, key=lambda item: float(item["e2e_seconds"]))
    result = {
        "config": config.name,
        "complete": True,
        "effective_hbm_cap_bytes": cap_bytes,
        "effective_hbm_cap_gib": 40.0,
        "physical_gpu_index": 0,
        "input_tokens": 4096,
        "decode_steps": 128,
        "evaluation_requests": 200,
        "policies": ["stream2", "permanent_k"],
        "feasible_k": [row["k"] for row in rows],
        "skipped_infeasible_k": [96, 128],
        "performance_source": "uninstrumented runtime_at_bmax",
        "component_breakdown_source": "instrumented one-wave profiles_at_bmax",
        "one_contiguous_h2d_copy_per_expert_fetch": True,
        "best_by_e2e_makespan": {"policy": best["policy"], "k": best["k"], "batch_size": best["measured_bmax"], "e2e_seconds": best["e2e_seconds"], "speedup_vs_stream2_pct": best["e2e_speedup_vs_stream2_pct"]},
        "rows": rows,
    }
    atomic_write_json(args.output_json, result)
    _write_csv(args.output_csv, rows)
    _plot(args.output_plot, rows)
    print(json.dumps({"complete": True, "rows": len(rows), "best": result["best_by_e2e_makespan"], "output_json": str(args.output_json), "output_csv": str(args.output_csv), "output_plot": str(args.output_plot)}, indent=2))


if __name__ == "__main__":
    main()
