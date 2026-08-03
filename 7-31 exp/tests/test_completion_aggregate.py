import pytest

from experiments.analysis.aggregate_completion import (
    _performance_fields,
    _validate_performance,
    _validate_profile,
)


def test_completion_aggregate_rejects_instrumented_performance() -> None:
    value = {
        "policy": "stream2",
        "k": 0,
        "batch_size": 157,
        "requests": 1200,
        "decode_steps": 256,
        "kv_setup": "static_zero",
        "gpu_physical_index": 0,
        "timeline_events_enabled": False,
        "instrumented_profile_only": False,
        "eligible_for_throughput_and_makespan_comparison": True,
        "timing_interpretation": "uninstrumented_performance_run",
        "prefetch_submit_order": "compute_first",
        "expert_h2d_fetches": 10,
        "expert_h2d_copy_operations": 10,
        "d2d_admission_copies": 0,
        "forced_routing_weight_source": "recorded_trace_weights",
        "forced_routing_weight_alignment_caveat": False,
        "forced_routing_trace_sha256": "trace",
        "waves": 8,
        "steady_state_full_wave_repeats": 5,
        "performance_warmup_and_repeat_protocol_valid": True,
        "cold_start_seconds": 3.0,
        "kv_setup_seconds": 2.0,
        "fixed_workload_decode_makespan_seconds": 5.0,
        "cold_start_and_kv_included_makespan_seconds": 10.0,
        "git_sha": "abc",
        "command": "python test.py",
        "timestamp": "2026-07-31T00:00:00+00:00",
        "trace_sha256": "trace",
    }
    _validate_performance(
        value, policy="stream2", k=0, batch_size=157, requests=1200
    )
    value["timeline_events_enabled"] = True
    with pytest.raises(ValueError):
        _validate_performance(
            value, policy="stream2", k=0, batch_size=157, requests=1200
        )


def test_completion_aggregate_requires_two_profile_waves() -> None:
    value = {
        "policy": "permanent_k",
        "k": 32,
        "batch_size": 40,
        "requests": 80,
        "decode_steps": 256,
        "kv_setup": "static_zero",
        "gpu_physical_index": 0,
        "timeline_events_enabled": True,
        "instrumented_profile_only": True,
        "eligible_for_throughput_and_makespan_comparison": False,
        "timing_interpretation": (
            "intrusive_component_profile_not_performance_evidence"
        ),
        "prefetch_submit_order": "compute_first",
        "expert_h2d_fetches": 20,
        "expert_h2d_copy_operations": 20,
        "trace_sha256": "trace",
        "forced_routing_trace_sha256": "trace",
        "total_h2d_duration_ms": 30.0,
        "exposed_h2d_stall_ms": 18.0,
        "overlapped_h2d_ms": 12.0,
        "wave_results": [
            {
                "measurement_phase": "warmup",
                "total_h2d_duration_ms": 10.0,
                "exposed_h2d_stall_ms": 6.0,
                "overlapped_h2d_ms": 4.0,
            },
            {
                "measurement_phase": "steady_state",
                "total_h2d_duration_ms": 20.0,
                "exposed_h2d_stall_ms": 12.0,
                "overlapped_h2d_ms": 8.0,
            },
        ],
    }
    _validate_profile(value, policy="permanent_k", k=32, batch_size=40)
    value["wave_results"].pop()
    with pytest.raises(ValueError):
        _validate_profile(value, policy="permanent_k", k=32, batch_size=40)


def test_completion_aggregate_computes_h2d_bytes_per_token() -> None:
    value = {
        "batch_size": 40,
        "generated_tokens": 100,
        "fixed_workload_decode_makespan_seconds": 5.0,
        "fixed_workload_tokens_per_second": 20.0,
        "steady_full_batch_tokens_per_second": 21.0,
        "cold_start_seconds": 3.0,
        "kv_setup_seconds": 2.0,
        "expert_h2d_fetches": 4,
        "expert_h2d_bytes": 400,
        "expert_h2d_copy_operations": 4,
        "natural_route_set_mismatch_rate": 0.1,
    }
    fields = _performance_fields("common", value)
    assert fields["common_h2d_bytes_per_generated_token"] == 4.0
    assert fields["common_h2d_copy_operations_per_fetch"] == 1.0
