from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "policy",
    "k",
    "batch_size",
    "decode_steps",
    "generated_tokens",
    "decode_wall_seconds",
    "decode_tokens_per_second",
    "throughput_gain_vs_stream2",
    "expert_h2d_fetches",
    "expert_h2d_bytes",
    "h2d_reduction_vs_stream2",
    "permanent_hits",
    "quota_hits",
    "quota_evictions",
    "d2d_admission_copies",
    "peak_allocated_bytes",
    "policy_initialization_seconds",
    "natural_route_mismatch_rate",
    "host_memory_mode",
    "source",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate real offloaded-decode JSON results."
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source"] = str(path)
        rows.append(value)
    baselines = {
        (row["batch_size"], row["decode_steps"]): row
        for row in rows
        if row["policy"] == "stream2" and row["k"] == 0
    }
    output_rows = []
    for row in rows:
        baseline = baselines.get((row["batch_size"], row["decode_steps"]))
        normalized = dict(row)
        normalized["throughput_gain_vs_stream2"] = (
            row["decode_tokens_per_second"]
            / baseline["decode_tokens_per_second"]
            if baseline
            else ""
        )
        normalized["h2d_reduction_vs_stream2"] = (
            1 - row["expert_h2d_bytes"] / baseline["expert_h2d_bytes"]
            if baseline and baseline["expert_h2d_bytes"]
            else ""
        )
        output_rows.append(
            {field: normalized.get(field, "") for field in FIELDS}
        )
    output_rows.sort(key=lambda row: (int(row["batch_size"]), int(row["k"]), row["policy"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)


if __name__ == "__main__":
    main()

