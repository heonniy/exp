import json
import sqlite3

from experiments.analysis.analyze_nsys_expert_copies import analyze


def test_nsys_analysis_matches_expert_sized_copies(tmp_path) -> None:
    database = tmp_path / "profile.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
            "(bytes INTEGER, start INTEGER, end INTEGER)"
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?)",
            [
                (9437184, 0, 10),
                (9437184, 10, 20),
                (8, 20, 21),
            ],
        )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "expert_h2d_fetches": 2,
                "expert_h2d_copy_operations": 2,
                "forced_routing_trace_sha256": "trace",
            }
        ),
        encoding="utf-8",
    )
    result = analyze(database, runtime, 9437184)
    assert result["observed_expert_sized_memcpy_operations"] == 2
    assert result["one_copy_per_fetch_verified"]
