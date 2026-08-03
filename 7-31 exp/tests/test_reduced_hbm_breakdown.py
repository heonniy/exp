import pytest

from experiments.analysis.analyze_reduced_hbm_breakdown import (
    _decode_miss_metrics,
    _profile_phase,
    _wave_rows,
)


def _profile() -> dict:
    return {
        "fixed_workload_prefill_makespan_seconds": 0.010,
        "prompt_tokens": 100,
        "prefill_total_h2d_duration_ms": 4.0,
        "prefill_exposed_h2d_stall_ms": 1.0,
        "prefill_overlapped_h2d_ms": 3.0,
        "prefill_compute_stream_h2d_wait_ms": 0.8,
        "prefill_first_miss_stall_ms": 0.2,
        "prefill_attention_ms": 2.0,
        "prefill_router_ms": 1.0,
        "prefill_expert_compute_ms": 3.0,
        "prefill_other_dense_host_idle_ms": 3.0,
    }


def test_profile_phase_keeps_h2d_nonadditive_and_closes_wall_partition() -> None:
    result = _profile_phase(_profile(), "prefill")
    assert result["h2d_overlap_pct"] == pytest.approx(75.0)
    assert result["raw_h2d_us_per_token"] == pytest.approx(40.0)
    assert result["compute_stream_h2d_wait_us_per_token"] == pytest.approx(8.0)
    assert result["non_h2d_wall_us_per_token"] == pytest.approx(90.0)
    assert sum(
        result[f"{name}_wall_pct"]
        for name in (
            "attention",
            "router_module",
            "expert_execution",
            "exposed_h2d",
            "residual_dense_dispatch_host_sync_idle",
        )
    ) == pytest.approx(100.0)


def test_profile_phase_rejects_double_counted_wall_partition() -> None:
    value = _profile()
    value["prefill_other_dense_host_idle_ms"] = 7.0
    with pytest.raises(ValueError, match="additive wall partition"):
        _profile_phase(value, "prefill")


def test_wave_rows_marks_last_partial_wave() -> None:
    runtime = {
        "policy": "permanent_k",
        "k": 8,
        "batch_size": 3,
        "wave_results": [],
    }
    for index, batch in enumerate((3, 2)):
        runtime["wave_results"].append(
            {
                "wave_index": index,
                "measurement_phase": "warmup" if index == 0 else "steady_state",
                "start": index * 3,
                "stop": index * 3 + batch,
                "batch_size": batch,
                "prompt_tokens": batch * 4096,
                "generated_tokens": batch * 128,
                "prefill_wall_seconds": 1.0,
                "decode_wall_seconds": 2.0,
                "e2e_wall_seconds": 3.0,
                "prefill_prompt_tokens_per_second": 1.0,
                "decode_tokens_per_second": 1.0,
                "e2e_total_tokens_per_second": 1.0,
                "prefill_expert_h2d_fetches": 1,
                "expert_h2d_fetches": 2,
            }
        )
    rows = _wave_rows(runtime)
    assert rows[0]["is_full_wave"] is True
    assert rows[0]["is_partial_wave"] is False
    assert rows[1]["is_full_wave"] is False
    assert rows[1]["is_partial_wave"] is True


def test_decode_miss_metrics_close_fetches_hits_and_executions() -> None:
    result = _decode_miss_metrics(
        {
            "expert_h2d_fetches": 20,
            "expert_h2d_copy_operations": 20,
            "permanent_hits": 80,
            "expert_executions": 100,
            "generated_tokens": 10,
        }
    )
    assert result["decode_fetch_miss_rate_pct"] == pytest.approx(20.0)
    assert result["decode_fetches_per_generated_token"] == pytest.approx(2.0)
    assert result["decode_active_expert_executions_per_generated_token"] == pytest.approx(
        10.0
    )


def test_decode_miss_metrics_rejects_non_one_copy_fetches() -> None:
    with pytest.raises(ValueError, match="exactly one H2D copy"):
        _decode_miss_metrics(
            {
                "expert_h2d_fetches": 20,
                "expert_h2d_copy_operations": 40,
                "permanent_hits": 80,
                "expert_executions": 100,
                "generated_tokens": 10,
            }
        )
