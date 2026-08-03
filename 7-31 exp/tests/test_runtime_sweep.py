import json
from pathlib import Path

import pytest

from experiments.benchmark.run_4k256_completion import (
    _fixed_command,
    _write_manifest,
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


def test_runtime_sweep_honors_permanent_only_policy_list() -> None:
    values = list(
        configurations(
            (0, 2, 8, 80),
            128,
            ("stream2", "permanent_k"),
        )
    )
    assert values == [
        ("stream2", 0),
        ("permanent_k", 2),
        ("permanent_k", 8),
        ("permanent_k", 80),
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


def test_completion_manifest_retains_runs_across_phases(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    common_output = tmp_path / "common.json"
    profile_output = tmp_path / "profile.json"
    common_output.touch()
    profile_output.touch()
    _write_manifest(
        manifest,
        {"phase": "common"},
        [
            {
                "mode": "common_fixed_batch",
                "policy": "stream2",
                "k": 0,
                "batch_size": 40,
                "output": str(common_output),
            }
        ],
    )
    _write_manifest(
        manifest,
        {"phase": "profile"},
        [
            {
                "mode": "instrumented_profile_common_fixed_batch",
                "policy": "stream2",
                "k": 0,
                "batch_size": 40,
                "output": str(profile_output),
            }
        ],
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(value["runs"]) == 2
    assert value["run_modes"] == [
        "common_fixed_batch",
        "instrumented_profile_common_fixed_batch",
    ]
    assert all(run["completed"] for run in value["runs"])
