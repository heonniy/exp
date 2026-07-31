import json
from pathlib import Path

import pytest

from experiments.benchmark.run_4k256_completion import (
    _fixed_command,
    resolve_common_batch_size,
)
from experiments.benchmark.run_runtime_sweep import (
    configurations,
    resolve_batch_size,
)


def test_runtime_sweep_has_unique_extreme_endpoints() -> None:
    values = list(configurations((0, 2, 8, 128), 128))
    assert values == [
        ("stream2", 0),
        ("permanent_k", 2),
        ("quota_lru_k", 2),
        ("permanent_k", 8),
        ("quota_lru_k", 8),
        ("full_resident", 128),
    ]


def test_runtime_sweep_resolves_measured_bmax(tmp_path) -> None:
    path = tmp_path / "quota_lru_k_k8.json"
    path.write_text(
        json.dumps({"policy": "quota_lru_k", "k": 8, "measured_bmax": 37}),
        encoding="utf-8",
    )
    assert resolve_batch_size(None, tmp_path, "quota_lru_k", 8) == 37
    assert resolve_batch_size(12, None, "quota_lru_k", 8) == 12


def test_completion_separates_wall_runtime_from_timeline_profile() -> None:
    arguments = {
        "config": Path("config.yaml"),
        "workload": Path("workload.jsonl"),
        "calibration_trace": Path("calibration.npz"),
        "forced_routing_trace": Path("evaluation.npz"),
        "policy": "permanent_k",
        "k": 32,
        "batch_size": 40,
        "requests": 1200,
        "decode_steps": 256,
        "output": Path("result.json"),
        "permanent_method": "batch_step_union_presence",
    }
    performance = _fixed_command(**arguments, timeline_events=False)
    profile = _fixed_command(**arguments, timeline_events=True)
    assert "--timeline-events" not in performance
    assert profile[-1] == "--timeline-events"
    order = performance.index("--prefetch-submit-order")
    assert performance[order + 1] == "compute_first"


def test_completion_falls_back_from_common_b40_to_preapproved_b32() -> None:
    assert resolve_common_batch_size(40, 38) == (32, True)
    assert resolve_common_batch_size(40, 41) == (40, False)
    with pytest.raises(ValueError):
        resolve_common_batch_size(40, 31)
