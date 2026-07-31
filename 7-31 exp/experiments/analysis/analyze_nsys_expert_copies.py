from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from experiments.common.io import atomic_write_json


def analyze(sqlite_path: Path, runtime_path: Path, expert_bytes: int) -> dict:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    with sqlite3.connect(sqlite_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        if table not in tables:
            raise ValueError(f"Nsight export does not contain {table}")
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "bytes" not in columns:
            raise ValueError("Nsight memcpy table does not contain a byte count")
        rows = list(
            connection.execute(
                f"SELECT bytes, COUNT(*), SUM(end - start) FROM {table} "
                "GROUP BY bytes ORDER BY bytes"
            )
        )
    distribution = [
        {
            "bytes": int(size),
            "operations": int(count),
            "total_duration_ns": int(duration or 0),
        }
        for size, count, duration in rows
    ]
    expert_rows = [row for row in distribution if row["bytes"] == expert_bytes]
    observed = expert_rows[0]["operations"] if expert_rows else 0
    expected = int(runtime["expert_h2d_fetches"])
    return {
        "validation": "nsight_one_packed_h2d_copy_per_expert_fetch",
        "nsight_sqlite": str(sqlite_path),
        "runtime_result": str(runtime_path),
        "expert_bytes": expert_bytes,
        "expected_expert_fetches": expected,
        "runtime_expert_h2d_copy_operations": int(
            runtime["expert_h2d_copy_operations"]
        ),
        "observed_expert_sized_memcpy_operations": observed,
        "one_copy_per_fetch_verified": (
            observed == expected
            and int(runtime["expert_h2d_copy_operations"]) == expected
        ),
        "forced_routing_trace_sha256": runtime.get(
            "forced_routing_trace_sha256"
        ),
        "memcpy_size_distribution": distribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify packed Expert copy count from an Nsight SQLite export."
    )
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--expert-bytes", type=int, default=9437184)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite, args.runtime, args.expert_bytes)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["one_copy_per_fetch_verified"]:
        raise RuntimeError("Nsight Expert H2D copy count does not match runtime fetches")


if __name__ == "__main__":
    main()
