from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from experiments.benchmark.run_runtime_sweep import configurations
from experiments.common.config import load_config
from experiments.common.io import atomic_write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def _validate_performance(
    value: dict,
    *,
    policy: str,
    k: int,
    batch_size: int,
    requests: int,
) -> None:
    identity = (value.get("policy"), int(value.get("k", -1)))
    if identity != (policy, k):
        raise ValueError(f"runtime identity mismatch: {identity} != {(policy, k)}")
    if int(value.get("batch_size", -1)) != batch_size:
        raise ValueError("runtime batch size mismatch")
    if int(value.get("requests", -1)) != requests:
        raise ValueError("runtime request count mismatch")
    if int(value.get("decode_steps", -1)) != 256:
        raise ValueError("completion runtime is not a 256-step decode")
    if value.get("kv_setup") != "static_zero":
        raise ValueError("completion runtime did not use static 4K KV setup")
    if int(value.get("gpu_physical_index", -1)) != 0:
        raise ValueError("completion runtime did not use physical GPU 0")
    if value.get("timeline_events_enabled") is not False:
        raise ValueError("performance runtime must not enable timeline events")
    if value.get("instrumented_profile_only") is not False:
        raise ValueError("performance runtime is labeled as an instrumented profile")
    if value.get("eligible_for_throughput_and_makespan_comparison") is not True:
        raise ValueError("performance runtime is not eligible for comparison")
    if value.get("timing_interpretation") != "uninstrumented_performance_run":
        raise ValueError("performance runtime has the wrong timing interpretation")
    if value.get("prefetch_submit_order") != "compute_first":
        raise ValueError("performance runtime did not use compute-first prefetch")
    if int(value.get("expert_h2d_copy_operations", -1)) != int(
        value.get("expert_h2d_fetches", -2)
    ):
        raise ValueError("performance runtime did not issue one H2D copy per fetch")
    if value.get("forced_routing_weight_source") != "recorded_trace_weights":
        raise ValueError("performance runtime did not replay recorded routing weights")
    if value.get("forced_routing_weight_alignment_caveat") is not False:
        raise ValueError("performance runtime reports a routing-weight caveat")
    if value.get("trace_sha256") != value.get("forced_routing_trace_sha256"):
        raise ValueError("performance runtime trace digests disagree")
    if int(value.get("d2d_admission_copies", -1)) != 0:
        raise ValueError("performance runtime used D2D admission copies")
    if int(value.get("waves", -1)) != math.ceil(requests / batch_size):
        raise ValueError("performance runtime wave count mismatch")
    if int(value.get("steady_state_full_wave_repeats", -1)) < 5:
        raise ValueError("performance runtime has fewer than five steady repeats")
    if value.get("performance_warmup_and_repeat_protocol_valid") is not True:
        raise ValueError("performance warmup/repeat protocol is invalid")
    expected_total = (
        float(value["cold_start_seconds"])
        + float(value["kv_setup_seconds"])
        + float(value["fixed_workload_decode_makespan_seconds"])
    )
    if not _close(
        float(value["cold_start_and_kv_included_makespan_seconds"]),
        expected_total,
    ):
        raise ValueError("performance cold/KV/decode makespan does not close")
    for field in ("git_sha", "command", "timestamp", "trace_sha256"):
        if not value.get(field):
            raise ValueError(f"performance runtime is missing provenance: {field}")


def _validate_profile(
    value: dict,
    *,
    policy: str,
    k: int,
    batch_size: int,
) -> None:
    identity = (value.get("policy"), int(value.get("k", -1)))
    if identity != (policy, k):
        raise ValueError(f"profile identity mismatch: {identity} != {(policy, k)}")
    if int(value.get("batch_size", -1)) != batch_size:
        raise ValueError("profile batch size mismatch")
    if int(value.get("requests", -1)) != batch_size * 2:
        raise ValueError("profile must contain exactly two common-batch waves")
    if int(value.get("decode_steps", -1)) != 256:
        raise ValueError("profile is not a 256-step decode")
    if value.get("kv_setup") != "static_zero":
        raise ValueError("profile did not use static 4K KV setup")
    if int(value.get("gpu_physical_index", -1)) != 0:
        raise ValueError("profile did not use physical GPU 0")
    if value.get("timeline_events_enabled") is not True:
        raise ValueError("breakdown profile must enable timeline events")
    if value.get("instrumented_profile_only") is not True:
        raise ValueError("breakdown profile is not labeled instrumented")
    if value.get("eligible_for_throughput_and_makespan_comparison") is not False:
        raise ValueError("breakdown profile is incorrectly performance-eligible")
    if value.get("timing_interpretation") != (
        "intrusive_component_profile_not_performance_evidence"
    ):
        raise ValueError("breakdown profile has the wrong timing interpretation")
    if value.get("prefetch_submit_order") != "compute_first":
        raise ValueError("breakdown profile did not use compute-first prefetch")
    if len(value.get("wave_results", [])) != 2:
        raise ValueError("breakdown profile does not contain two waves")
    cold, steady = value["wave_results"]
    if cold.get("measurement_phase") != "warmup":
        raise ValueError("profile first wave is not the cold/warmup wave")
    if steady.get("measurement_phase") != "steady_state":
        raise ValueError("profile second wave is not the steady wave")
    if int(value.get("expert_h2d_copy_operations", -1)) != int(
        value.get("expert_h2d_fetches", -2)
    ):
        raise ValueError("profile did not issue one H2D copy per fetch")
    if value.get("trace_sha256") != value.get("forced_routing_trace_sha256"):
        raise ValueError("profile trace digests disagree")
    for prefix, row in (("cold", cold), ("steady", steady)):
        total = float(row["total_h2d_duration_ms"])
        exposed = float(row["exposed_h2d_stall_ms"])
        overlap = float(row["overlapped_h2d_ms"])
        if not _close(total, exposed + overlap):
            raise ValueError(f"profile {prefix} H2D decomposition does not close")
    for field in ("total_h2d_duration_ms", "exposed_h2d_stall_ms", "overlapped_h2d_ms"):
        if not _close(
            float(value[field]),
            sum(float(wave[field]) for wave in value["wave_results"]),
        ):
            raise ValueError(f"profile aggregate does not match waves: {field}")


def _validate_quota_controls(value: dict) -> dict:
    expected_controls = [
        "ascending_always_admit",
        "resident_hit_first",
        "miss_bypass",
        "no_admission",
        "window_frequency",
        "random_order",
    ]
    if value.get("cache_simulation_only") is not True:
        raise ValueError("Quota controls are not labeled cache-simulation-only")
    if int(value.get("requests", -1)) != 1200:
        raise ValueError("Quota controls do not contain 1,200 requests")
    if int(value.get("batch_size", -1)) != 40:
        raise ValueError("Quota controls are not fixed B=40")
    if value.get("quota_controls") != expected_controls:
        raise ValueError("Quota controls are incomplete or reordered")
    if value.get("random_seeds") != [731, 732, 733]:
        raise ValueError("Quota random-order sensitivity seeds are incomplete")
    if value.get("k_values") != [0, 8, 32, 64, 96, 128]:
        raise ValueError("Quota control k values are incomplete")
    rows = value.get("results", [])
    if len(rows) != 38:
        raise ValueError(f"expected 38 Quota control rows, got {len(rows)}")
    trace_digests = set()
    policies_by_k: dict[int, set[str]] = {}
    for row in rows:
        k = int(row["k"])
        policies_by_k.setdefault(k, set()).add(str(row["policy"]))
        trace_digests.add(str(row["trace_sha256"]))
        if int(row["batch_size"]) != 40 or int(row["requests"]) != 1200:
            raise ValueError("Quota control row workload mismatch")
        if int(row["waves"]) != 30 or int(row["generated_tokens"]) != 307200:
            raise ValueError("Quota control row decode extent mismatch")
        if int(row["hits"]) + int(row["misses"]) != int(row["expert_executions"]):
            raise ValueError("Quota control hit/miss accounting does not close")
        if int(row["fetches"]) != int(row["misses"]):
            raise ValueError("Quota control fetch/miss accounting does not close")
        if int(row["h2d_bytes"]) != int(row["fetches"]) * 9437184:
            raise ValueError("Quota control H2D byte accounting does not close")
    if len(trace_digests) != 1:
        raise ValueError("Quota control trace digests do not match")
    expected_positive = {
        "permanent_k",
        "quota_lru_k",
        "quota_lru_resident_hit_first",
        "quota_lru_miss_bypass",
        "quota_lru_no_admission",
        "quota_lru_window_frequency",
        "quota_lru_random_order_seed731",
        "quota_lru_random_order_seed732",
        "quota_lru_random_order_seed733",
    }
    for k in (8, 32, 64, 96):
        if policies_by_k.get(k) != expected_positive:
            raise ValueError(f"Quota control policy matrix is incomplete at k={k}")
    if policies_by_k.get(0) != {"stream2"} or policies_by_k.get(128) != {"full_resident"}:
        raise ValueError("Quota control endpoints are incomplete")
    return {
        "validated": True,
        "rows": len(rows),
        "batch_size": 40,
        "requests": 1200,
        "decode_steps": 256,
        "trace_sha256": next(iter(trace_digests)),
        "controls": expected_controls,
        "random_seeds": [731, 732, 733],
    }


def _performance_fields(prefix: str, value: dict) -> dict:
    generated = int(value["generated_tokens"])
    return {
        f"{prefix}_batch_size": int(value["batch_size"]),
        f"{prefix}_makespan_seconds": float(
            value["fixed_workload_decode_makespan_seconds"]
        ),
        f"{prefix}_tokens_per_second": float(
            value["fixed_workload_tokens_per_second"]
        ),
        f"{prefix}_steady_full_batch_tokens_per_second": float(
            value["steady_full_batch_tokens_per_second"]
        ),
        f"{prefix}_cold_start_seconds": float(value["cold_start_seconds"]),
        f"{prefix}_kv_setup_seconds": float(value["kv_setup_seconds"]),
        f"{prefix}_h2d_fetches": int(value["expert_h2d_fetches"]),
        f"{prefix}_h2d_bytes": int(value["expert_h2d_bytes"]),
        f"{prefix}_h2d_bytes_per_generated_token": (
            int(value["expert_h2d_bytes"]) / generated
        ),
        f"{prefix}_h2d_copy_operations_per_fetch": (
            int(value["expert_h2d_copy_operations"])
            / int(value["expert_h2d_fetches"])
            if int(value["expert_h2d_fetches"])
            else 1.0
        ),
        f"{prefix}_natural_route_set_mismatch_rate": float(
            value["natural_route_set_mismatch_rate"]
        ),
    }


def _profile_fields(value: dict) -> dict:
    cold, steady = value["wave_results"]
    result = {
        "profile_instrumented": True,
        "profile_excluded_from_throughput_comparison": True,
        "profile_total_h2d_duration_ms": float(value["total_h2d_duration_ms"]),
        "profile_exposed_h2d_stall_ms": float(value["exposed_h2d_stall_ms"]),
        "profile_overlapped_h2d_ms": float(value["overlapped_h2d_ms"]),
        "profile_expert_compute_ms": float(value["expert_compute_ms"]),
        "profile_attention_ms": float(value["attention_ms"]),
        "profile_router_ms": float(value["router_ms"]),
    }
    for label, wave in (("cold_wave", cold), ("steady_wave", steady)):
        for source, target in (
            ("decode_wall_seconds", "wall_seconds"),
            ("total_h2d_duration_ms", "total_h2d_duration_ms"),
            ("exposed_h2d_stall_ms", "exposed_h2d_stall_ms"),
            ("overlapped_h2d_ms", "overlapped_h2d_ms"),
            ("compute_stream_h2d_wait_ms", "compute_stream_h2d_wait_ms"),
            ("expert_compute_ms", "expert_compute_ms"),
            ("attention_ms", "attention_ms"),
            ("router_ms", "router_ms"),
            ("other_dense_host_idle_ms", "other_dense_host_idle_ms"),
            ("expert_h2d_fetches", "h2d_fetches"),
        ):
            result[f"profile_{label}_{target}"] = wave[source]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the strict 4K/256 completion matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/h100_lmsys_4k256.yaml"),
    )
    parser.add_argument(
        "--completion-dir",
        type=Path,
        default=Path("experiments/results/completion_4k256_1200"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--quota-controls-summary",
        type=Path,
        default=Path(
            "experiments/results/legacy_logical_cache/"
            "trace_controls_b40_n1200/summary.json"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = _load(args.completion_dir / "manifest.json")
    common_batch = int(manifest["common_batch_size"])
    requests = int(manifest["validation_requests"])
    if common_batch != 40 or requests != 1200:
        raise ValueError("completion aggregate requires common B=40 and 1,200 requests")
    if int(manifest.get("gpu_physical_index", -1)) != 0:
        raise ValueError("completion manifest did not use physical GPU 0")
    if int(manifest.get("decode_steps", -1)) != 256:
        raise ValueError("completion manifest is not a 256-step decode")
    quota_controls = _validate_quota_controls(
        _load(args.quota_controls_summary)
    )
    rows = []
    missing = []
    common_digests: list[tuple[str, int, tuple[str, ...]]] = []
    trace_digests: set[str] = set()
    output_digests: set[str] = set()
    for policy, k in configurations(
        config.runtime_k, config.model.num_experts_per_layer
    ):
        bmax_path = args.completion_dir / "bmax" / f"{policy}_k{k}.json"
        bmax = _load(bmax_path)
        if bmax.get("probe_mode") != (
            "real_runtime_static_peak_kv_full_256_step_boundary"
        ):
            raise ValueError("Bmax result was not validated for all 256 steps")
        if bmax.get("boundary_closed") is not True:
            raise ValueError("Bmax result does not close the B/B+1 boundary")
        measured_bmax = int(bmax["measured_bmax"])
        at_bmax_path = (
            args.completion_dir
            / "runtime_at_bmax"
            / f"{policy}_k{k}_b{measured_bmax}_n{requests}.json"
        )
        common_path = (
            args.completion_dir
            / f"runtime_common_b{common_batch}"
            / f"{policy}_k{k}_b{common_batch}_n{requests}.json"
        )
        profile_path = (
            args.completion_dir
            / f"profile_common_b{common_batch}"
            / f"{policy}_k{k}_b{common_batch}_n{common_batch * 2}.json"
        )
        required = (at_bmax_path, common_path, profile_path)
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            missing.extend(absent)
            continue
        at_bmax = _load(at_bmax_path)
        common = _load(common_path)
        profile = _load(profile_path)
        _validate_performance(
            at_bmax,
            policy=policy,
            k=k,
            batch_size=measured_bmax,
            requests=requests,
        )
        _validate_performance(
            common,
            policy=policy,
            k=k,
            batch_size=common_batch,
            requests=requests,
        )
        _validate_profile(profile, policy=policy, k=k, batch_size=common_batch)
        trace_digests.update(
            str(value["trace_sha256"])
            for value in (at_bmax, common, profile)
        )
        output_digests.update(
            str(value["forced_output_ids_sha256"])
            for value in (at_bmax, common)
        )
        if tuple(profile["final_logits_sha256_by_wave"]) != tuple(
            common["final_logits_sha256_by_wave"][:2]
        ):
            raise ValueError(
                f"profile/common logits mismatch for {policy} k={k}"
            )
        common_digests.append(
            (policy, k, tuple(common["final_logits_sha256_by_wave"]))
        )
        row = {
            "policy": policy,
            "k": k,
            "measured_bmax": measured_bmax,
            "theoretical_bmax": int(bmax["theoretical_bmax"]),
            "persistent_expert_bytes": int(bmax["persistent_expert_bytes"]),
            "transient_expert_bytes": int(bmax["transient_expert_bytes"]),
            **_performance_fields("at_bmax", at_bmax),
            **_performance_fields("common", common),
            **_profile_fields(profile),
        }
        rows.append(row)

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"completion matrix is missing {len(missing)} outputs")
    digest_sets = {value for _, _, value in common_digests}
    all_common_digests_match = len(digest_sets) <= 1
    if rows and not all_common_digests_match:
        raise ValueError("common-batch final logits digests do not match")
    if len(trace_digests) > 1:
        raise ValueError("completion runtime trace digests do not match")
    if len(output_digests) > 1:
        raise ValueError("completion forced-output digests do not match")
    if trace_digests and trace_digests != {
        str(manifest["forced_routing_trace_sha256"])
    }:
        raise ValueError("completion runtime/manifest trace digests disagree")
    result = {
        "config": config.name,
        "validation_requests": requests,
        "decode_steps": 256,
        "common_batch_size": common_batch,
        "expected_configurations": len(
            list(configurations(config.runtime_k, config.model.num_experts_per_layer))
        ),
        "completed_configurations": len(rows),
        "complete": not missing,
        "missing_outputs": missing,
        "all_common_final_logits_digests_match": all_common_digests_match,
        "all_runtime_trace_digests_match_manifest": len(trace_digests) <= 1,
        "all_forced_output_digests_match": len(output_digests) <= 1,
        "instrumented_profiles_excluded_from_performance": True,
        "quota_controls": quota_controls,
        "rows": rows,
    }
    atomic_write_json(args.output_json, result)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        args.output_csv.write_text("", encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": not missing,
                "completed_configurations": len(rows),
                "missing_outputs": len(missing),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
