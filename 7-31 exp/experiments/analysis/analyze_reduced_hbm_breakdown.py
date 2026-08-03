from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from experiments.common.io import atomic_write_json


PHASES = {
    "prefill": {
        "wall": "fixed_workload_prefill_makespan_seconds",
        "tokens": "prompt_tokens",
        "prefix": "prefill_",
    },
    "decode": {
        "wall": "fixed_workload_decode_makespan_seconds",
        "tokens": "generated_tokens",
        "prefix": "",
    },
}

ADDITIVE_COMPONENTS = (
    "attention",
    "router_projection",
    "expert_execution",
    "exposed_h2d",
    "residual_dense_dispatch_host_sync_idle",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, *, abs_tol: float = 1e-3) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=abs_tol)


def _identity(value: dict) -> tuple[str, int]:
    return str(value["policy"]), int(value["k"])


def _profile_phase(value: dict, phase: str) -> dict:
    spec = PHASES[phase]
    prefix = spec["prefix"]
    tokens = int(value[spec["tokens"]])
    wall_ms = 1000.0 * float(value[spec["wall"]])
    raw_h2d_ms = float(value[f"{prefix}total_h2d_duration_ms"])
    exposed_h2d_ms = float(value[f"{prefix}exposed_h2d_stall_ms"])
    overlapped_h2d_ms = float(value[f"{prefix}overlapped_h2d_ms"])
    compute_stream_h2d_wait_ms = float(
        value[f"{prefix}compute_stream_h2d_wait_ms"]
    )
    first_miss_stall_ms = float(value[f"{prefix}first_miss_stall_ms"])
    attention_ms = float(value[f"{prefix}attention_ms"])
    router_projection_ms = float(value[f"{prefix}router_ms"])
    expert_execution_ms = float(value[f"{prefix}expert_compute_ms"])
    residual_ms = float(value[f"{prefix}other_dense_host_idle_ms"])

    _require(
        _close(raw_h2d_ms, exposed_h2d_ms + overlapped_h2d_ms),
        f"{phase} H2D intervals do not close",
    )
    additive_total_ms = (
        attention_ms
        + router_projection_ms
        + expert_execution_ms
        + exposed_h2d_ms
        + residual_ms
    )
    _require(
        _close(wall_ms, additive_total_ms),
        f"{phase} additive wall partition does not close: "
        f"wall={wall_ms}, components={additive_total_ms}",
    )

    metrics_ms = {
        "profile_wall": wall_ms,
        "raw_h2d": raw_h2d_ms,
        "exposed_h2d": exposed_h2d_ms,
        "overlapped_h2d": overlapped_h2d_ms,
        "compute_stream_h2d_wait": compute_stream_h2d_wait_ms,
        "first_miss_stall": first_miss_stall_ms,
        "attention": attention_ms,
        "router_projection": router_projection_ms,
        "expert_execution": expert_execution_ms,
        "residual_dense_dispatch_host_sync_idle": residual_ms,
    }
    result = {
        "tokens": tokens,
        "h2d_overlap_pct": (
            100.0 * overlapped_h2d_ms / raw_h2d_ms if raw_h2d_ms else 0.0
        ),
    }
    for name, milliseconds in metrics_ms.items():
        result[f"{name}_ms"] = milliseconds
        result[f"{name}_us_per_token"] = 1000.0 * milliseconds / tokens
    for name in ADDITIVE_COMPONENTS:
        result[f"{name}_wall_pct"] = 100.0 * result[f"{name}_ms"] / wall_ms
    return result


def _wave_rows(runtime: dict) -> list[dict]:
    rows = []
    batch = int(runtime["batch_size"])
    for wave in runtime["wave_results"]:
        wave_batch = int(wave["batch_size"])
        rows.append(
            {
                "policy": runtime["policy"],
                "k": int(runtime["k"]),
                "bmax": batch,
                "total_waves": len(runtime["wave_results"]),
                "wave_index": int(wave["wave_index"]),
                "measurement_phase": wave["measurement_phase"],
                "start_request": int(wave["start"]),
                "stop_request": int(wave["stop"]),
                "wave_batch_size": wave_batch,
                "is_full_wave": wave_batch == batch,
                "is_partial_wave": wave_batch != batch,
                "prompt_tokens": int(wave["prompt_tokens"]),
                "generated_tokens": int(wave["generated_tokens"]),
                "prefill_wall_seconds": float(wave["prefill_wall_seconds"]),
                "decode_wall_seconds": float(wave["decode_wall_seconds"]),
                "e2e_wall_seconds": float(wave["e2e_wall_seconds"]),
                "prefill_prompt_tokens_per_second": float(
                    wave["prefill_prompt_tokens_per_second"]
                ),
                "decode_tokens_per_second": float(
                    wave["decode_tokens_per_second"]
                ),
                "e2e_total_tokens_per_second": float(
                    wave["e2e_total_tokens_per_second"]
                ),
                "prefill_expert_h2d_fetches": int(
                    wave["prefill_expert_h2d_fetches"]
                ),
                "decode_expert_h2d_fetches": int(wave["expert_h2d_fetches"]),
            }
        )
    return rows


def _wide_row(runtime: dict, profile: dict) -> dict:
    _require(_identity(runtime) == _identity(profile), "runtime/profile identity mismatch")
    _require(runtime["timeline_events_enabled"] is False, "runtime is instrumented")
    _require(profile["timeline_events_enabled"] is True, "profile is not instrumented")
    _require(
        int(runtime["batch_size"]) == int(profile["batch_size"]),
        "runtime/profile batch mismatch",
    )
    _require(int(runtime["requests"]) == 200, "runtime is not the 200-request run")
    _require(
        int(profile["requests"]) == int(profile["batch_size"]),
        "profile is not exactly one wave",
    )
    _require(int(runtime["decode_steps"]) == 128, "runtime is not 128-step decode")
    _require(int(profile["decode_steps"]) == 128, "profile is not 128-step decode")

    wave_rows = _wave_rows(runtime)
    _require(
        _close(
            sum(row["prefill_wall_seconds"] for row in wave_rows),
            float(runtime["fixed_workload_prefill_makespan_seconds"]),
        ),
        "runtime prefill does not equal the sum of wave prefill times",
    )
    _require(
        _close(
            sum(row["decode_wall_seconds"] for row in wave_rows),
            float(runtime["fixed_workload_decode_makespan_seconds"]),
        ),
        "runtime decode does not equal the sum of wave decode times",
    )
    _require(
        _close(
            sum(row["e2e_wall_seconds"] for row in wave_rows),
            float(runtime["fixed_workload_e2e_makespan_seconds"]),
        ),
        "runtime E2E does not equal the sum of wave E2E times",
    )
    full_steady = [
        row["e2e_wall_seconds"]
        for row in wave_rows
        if row["measurement_phase"] == "steady_state" and row["is_full_wave"]
    ]
    prefill = _profile_phase(profile, "prefill")
    decode = _profile_phase(profile, "decode")
    row = {
        "policy": runtime["policy"],
        "k": int(runtime["k"]),
        "bmax": int(runtime["batch_size"]),
        "waves": len(wave_rows),
        "runtime_prefill_seconds": float(
            runtime["fixed_workload_prefill_makespan_seconds"]
        ),
        "runtime_decode_seconds": float(
            runtime["fixed_workload_decode_makespan_seconds"]
        ),
        "runtime_e2e_seconds": float(runtime["fixed_workload_e2e_makespan_seconds"]),
        "warmup_wave_e2e_seconds": wave_rows[0]["e2e_wall_seconds"],
        "steady_full_wave_e2e_median_seconds": (
            float(statistics.median(full_steady))
            if full_steady
            else None
        ),
        "last_wave_batch_size": wave_rows[-1]["wave_batch_size"],
        "last_wave_e2e_seconds": wave_rows[-1]["e2e_wall_seconds"],
    }
    for phase, metrics in (("prefill", prefill), ("decode", decode)):
        for name, value in metrics.items():
            row[f"{phase}_{name}"] = value
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_breakdown(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "attention": "#4c78a8",
        "router_projection": "#72b7b2",
        "expert_execution": "#f58518",
        "exposed_h2d": "#e45756",
        "residual_dense_dispatch_host_sync_idle": "#b279a2",
    }
    labels = {
        "attention": "Attention",
        "router_projection": "Router projection",
        "expert_execution": "Expert exec. (gather+MLP+scatter)",
        "exposed_h2d": "Exposed H2D",
        "residual_dense_dispatch_host_sync_idle": "Residual dense/dispatch/sync/idle",
    }
    ks = [row["k"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    for component in ADDITIVE_COMPONENTS:
        axes[0].plot(
            ks,
            [row[f"decode_{component}_us_per_token"] / 1000.0 for row in rows],
            marker="o",
            linewidth=2,
            color=colors[component],
            label=labels[component],
        )
    axes[0].set_yscale("log")
    axes[0].set(
        title="Decode component cost per generated token",
        xlabel="Permanent experts per layer (k)",
        ylabel="ms / generated token (log scale)",
        xticks=ks,
    )
    axes[0].grid(alpha=0.25, which="both")

    h2d_lines = (
        ("decode_raw_h2d_us_per_token", "Raw H2D", "#9c755f"),
        ("decode_exposed_h2d_us_per_token", "Exposed H2D", "#e45756"),
        ("decode_overlapped_h2d_us_per_token", "H2D-compute overlap", "#54a24b"),
        (
            "decode_compute_stream_h2d_wait_us_per_token",
            "Compute-stream H2D wait",
            "#ff9da6",
        ),
    )
    for field, label, color in h2d_lines:
        axes[1].plot(
            ks,
            [row[field] / 1000.0 for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1].set(
        title="H2D timing and dependency wait",
        xlabel="Permanent experts per layer (k)",
        ylabel="ms / generated token",
        xticks=ks,
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    bottom = [0.0] * len(rows)
    for component in ADDITIVE_COMPONENTS:
        values = [row[f"decode_{component}_wall_pct"] for row in rows]
        axes[2].bar(
            ks,
            values,
            width=5.5,
            bottom=bottom,
            color=colors[component],
            label=labels[component],
        )
        bottom = [left + value for left, value in zip(bottom, values)]
    axes[2].set(
        title="Intrusive decode profile wall-time composition",
        xlabel="Permanent experts per layer (k)",
        ylabel="profile wall time (%)",
        xticks=ks,
        ylim=(0, 100),
    )
    axes[2].grid(alpha=0.2, axis="y")
    handles, legend_labels = axes[2].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "40 GiB, 4K prefill + 128 decode: H2D falls, small-batch costs rise",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_waves(path: Path, rows: list[dict], wave_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis([index / (len(rows) - 1) for index in range(len(rows))])
    color_by_k = {row["k"]: color for row, color in zip(rows, colors)}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for row in rows:
        values = [wave for wave in wave_rows if wave["k"] == row["k"]]
        axes[0].plot(
            [wave["wave_index"] + 1 for wave in values],
            [wave["e2e_wall_seconds"] for wave in values],
            marker="o" if len(values) <= 16 else None,
            markersize=3,
            linewidth=1.4,
            color=color_by_k[row["k"]],
            label=f"k={row['k']} (B={row['bmax']})",
        )
    axes[0].set(
        title="Exact uninstrumented latency of every wave",
        xlabel="wave index",
        ylabel="prefill + decode seconds",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    ks = [row["k"] for row in rows]
    axes[1].bar(
        ks,
        [row["runtime_e2e_seconds"] for row in rows],
        width=5.5,
        color=[color_by_k[k] for k in ks],
        label="200-request E2E",
    )
    wave_axis = axes[1].twinx()
    wave_axis.plot(
        ks,
        [row["waves"] for row in rows],
        color="#d62728",
        marker="D",
        linewidth=2,
        label="wave count",
    )
    axes[1].set(
        title="Capacity loss turns short waves into a longer workload",
        xlabel="Permanent experts per layer (k)",
        ylabel="200-request E2E seconds",
        xticks=ks,
    )
    wave_axis.set_ylabel("number of waves", color="#d62728")
    axes[1].grid(alpha=0.2, axis="y")
    lines = [axes[1].patches[0], wave_axis.lines[0]]
    axes[1].legend(lines, ["200-request E2E", "wave count"], loc="upper left")
    figure.suptitle(
        "Wave-level evidence: the k=16 capacity boundary is the first reversal",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a K-by-K component and wave-latency study for the reduced HBM sweep."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-breakdown-csv", type=Path, required=True)
    parser.add_argument("--output-wave-csv", type=Path, required=True)
    parser.add_argument("--output-breakdown-plot", type=Path, required=True)
    parser.add_argument("--output-wave-plot", type=Path, required=True)
    args = parser.parse_args()

    runtime_values = {
        _identity(value): value
        for value in (
            _load(path) for path in (args.result_dir / "runtime_at_bmax").glob("*.json")
        )
    }
    profile_values = {
        _identity(value): value
        for value in (
            _load(path) for path in (args.result_dir / "profiles_at_bmax").glob("*.json")
        )
    }
    _require(runtime_values.keys() == profile_values.keys(), "runtime/profile K sets differ")
    _require(len(runtime_values) == 9, "expected nine feasible configurations")

    rows = [
        _wide_row(runtime_values[key], profile_values[key])
        for key in sorted(runtime_values, key=lambda item: item[1])
    ]
    wave_rows = [
        wave
        for key in sorted(runtime_values, key=lambda item: item[1])
        for wave in _wave_rows(runtime_values[key])
    ]
    _require(len(wave_rows) == sum(row["waves"] for row in rows), "wave row count mismatch")
    _require(len(wave_rows) == 151, "expected 151 exact runtime waves")

    baseline = rows[0]
    best = min(rows, key=lambda row: row["runtime_e2e_seconds"])
    largest = rows[-1]
    first_extra_wave = next(row for row in rows if row["waves"] > baseline["waves"])
    diagnosis = {
        "best_k": best["k"],
        "best_e2e_seconds": best["runtime_e2e_seconds"],
        "best_e2e_reduction_vs_k0_pct": 100.0
        * (baseline["runtime_e2e_seconds"] - best["runtime_e2e_seconds"])
        / baseline["runtime_e2e_seconds"],
        "first_extra_wave_k": first_extra_wave["k"],
        "first_extra_wave_count": first_extra_wave["waves"],
        "k80_e2e_ratio_vs_k0": largest["runtime_e2e_seconds"]
        / baseline["runtime_e2e_seconds"],
        "k80_raw_h2d_per_token_reduction_vs_k0_pct": 100.0
        * (
            1.0
            - largest["decode_raw_h2d_us_per_token"]
            / baseline["decode_raw_h2d_us_per_token"]
        ),
        "k80_exposed_h2d_per_token_reduction_vs_k0_pct": 100.0
        * (
            1.0
            - largest["decode_exposed_h2d_us_per_token"]
            / baseline["decode_exposed_h2d_us_per_token"]
        ),
        "k80_compute_stream_h2d_wait_per_token_reduction_vs_k0_pct": 100.0
        * (
            1.0
            - largest["decode_compute_stream_h2d_wait_us_per_token"]
            / baseline["decode_compute_stream_h2d_wait_us_per_token"]
        ),
        "k80_expert_execution_per_token_ratio_vs_k0": largest[
            "decode_expert_execution_us_per_token"
        ]
        / baseline["decode_expert_execution_us_per_token"],
        "k80_attention_per_token_ratio_vs_k0": largest[
            "decode_attention_us_per_token"
        ]
        / baseline["decode_attention_us_per_token"],
        "k80_router_projection_per_token_ratio_vs_k0": largest[
            "decode_router_projection_us_per_token"
        ]
        / baseline["decode_router_projection_us_per_token"],
        "k80_residual_per_token_ratio_vs_k0": largest[
            "decode_residual_dense_dispatch_host_sync_idle_us_per_token"
        ]
        / baseline["decode_residual_dense_dispatch_host_sync_idle_us_per_token"],
    }
    result = {
        "scope": {
            "effective_hbm_gib": 40,
            "prompt_tokens_per_request": 4096,
            "decode_tokens_per_request": 128,
            "requests": 200,
            "physical_gpu_index": 0,
            "policies": ["stream2", "permanent_k"],
        },
        "metric_definitions": {
            "performance_wall_time": "uninstrumented runtime_at_bmax; exact for the 200-request makespan and every wave",
            "component_timing": "intrusive profiles_at_bmax; exactly one Bmax wave per k and normalized by phase-token count",
            "attention": "CUDA event time around each self_attn module",
            "router_projection": "CUDA event time around each MLP gate projection only; dispatch is not included",
            "expert_execution": "CUDA event time from token index_select through Expert MLP and weighted index_add; not pure GEMM",
            "raw_h2d": "sum of single-contiguous-buffer Expert H2D copy intervals",
            "overlapped_h2d": "intersection of Expert H2D intervals and expert-execution intervals",
            "exposed_h2d": "raw_h2d minus overlapped_h2d",
            "compute_stream_h2d_wait": "CUDA event time across compute-stream wait_event(copy_done); a direct H2D dependency wait, not total host synchronization",
            "residual_dense_dispatch_host_sync_idle": "profile wall minus attention, router projection, expert execution, and exposed H2D; includes dense layers, unisolated dispatch, host work, explicit stream synchronize, and idle",
            "additivity_warning": "raw/overlapped H2D and compute-stream H2D wait overlap other categories; only attention + router projection + expert execution + exposed H2D + residual is an additive wall partition",
        },
        "diagnosis": diagnosis,
        "rows": rows,
        "wave_rows": wave_rows,
    }
    atomic_write_json(args.output_json, result)
    _write_csv(args.output_breakdown_csv, rows)
    _write_csv(args.output_wave_csv, wave_rows)
    _plot_breakdown(args.output_breakdown_plot, rows)
    _plot_waves(args.output_wave_plot, rows, wave_rows)
    print(
        json.dumps(
            {
                "configurations": len(rows),
                "wave_rows": len(wave_rows),
                "diagnosis": diagnosis,
                "output_json": str(args.output_json),
                "output_breakdown_csv": str(args.output_breakdown_csv),
                "output_wave_csv": str(args.output_wave_csv),
                "output_breakdown_plot": str(args.output_breakdown_plot),
                "output_wave_plot": str(args.output_wave_plot),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
