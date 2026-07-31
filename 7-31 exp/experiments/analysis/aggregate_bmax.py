from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "policy",
    "k",
    "theoretical_bmax",
    "measured_bmax",
    "total_hbm_bytes",
    "dense_resident_bytes",
    "fixed_workspace_bytes",
    "safety_margin_bytes",
    "persistent_expert_bytes",
    "transient_expert_bytes",
    "kv_budget_bytes",
    "kv_bytes_per_request",
    "source",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate measured Bmax JSON files.")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source"] = str(path)
        rows.append({field: value.get(field, "") for field in FIELDS})
    rows.sort(key=lambda row: (int(row["k"]), row["policy"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
