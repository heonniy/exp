from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.benchmark.run_runtime_sweep import configurations
from experiments.common.config import load_config
from experiments.common.io import atomic_write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if value.get("timeline_events_enabled") is not False:
        raise ValueError("performance runtime must not enable timeline events")
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
    if int(value.get("steady_state_full_wave_repeats", -1)) < 5:
        raise ValueError("performance runtime has fewer than five steady repeats")
    if value.get("performance_warmup_and_repeat_protocol_valid") is not True:
        raise ValueError("performance warmup/repeat protocol is invalid")
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
    if value.get("timeline_events_enabled") is not True:
        raise ValueError("breakdown profile must enable timeline events")
    if value.get("prefetch_submit_order") != "compute_first":
        raise ValueError("breakdown profile did not use compute-first prefetch")
    if len(value.get("wave_results", [])) != 2:
        raise ValueError("breakdown profile does not contain two waves")


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
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = _load(args.completion_dir / "manifest.json")
    common_batch = int(manifest["common_batch_size"])
    requests = int(manifest["validation_requests"])
    rows = []
    missing = []
    common_digests: list[tuple[str, int, tuple[str, ...]]] = []
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
        "all_common_final_logits_digests_match": len(digest_sets) <= 1,
        "instrumented_profiles_excluded_from_performance": True,
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
