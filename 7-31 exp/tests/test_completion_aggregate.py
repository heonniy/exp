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
        "timeline_events_enabled": False,
        "prefetch_submit_order": "compute_first",
        "expert_h2d_fetches": 10,
        "expert_h2d_copy_operations": 10,
        "forced_routing_weight_source": "recorded_trace_weights",
        "forced_routing_weight_alignment_caveat": False,
        "steady_state_full_wave_repeats": 5,
        "performance_warmup_and_repeat_protocol_valid": True,
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
        "timeline_events_enabled": True,
        "prefetch_submit_order": "compute_first",
        "wave_results": [{}, {}],
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
