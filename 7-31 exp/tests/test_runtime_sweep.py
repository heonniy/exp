import json

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
