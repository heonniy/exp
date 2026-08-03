from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.common.io import atomic_write_json, atomic_write_jsonl
from experiments.trace.trace_schema import RoutingTrace


def workload_prefix(
    source: Path,
    *,
    requests: int,
    output_tokens: int,
) -> list[dict]:
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            forced = row["forced_output_ids"]
            if len(forced) < output_tokens:
                raise ValueError(
                    f"{source}: output prefix {output_tokens} exceeds source row"
                )
            value = dict(row)
            value["forced_output_ids"] = forced[:output_tokens]
            value["output_length"] = output_tokens
            rows.append(value)
            if len(rows) == requests:
                break
    if len(rows) != requests:
        raise ValueError(f"{source}: contains only {len(rows)} rows; need {requests}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive auditable request/token prefixes from a fixed workload."
    )
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--output-workload", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    if args.requests <= 0 or args.output_tokens <= 0:
        raise ValueError("requests and output-tokens must be positive")
    rows = workload_prefix(
        args.source_workload,
        requests=args.requests,
        output_tokens=args.output_tokens,
    )
    source_trace = RoutingTrace.load(args.source_trace)
    trace = source_trace.prefix(args.requests, args.output_tokens)
    workload_ids = [str(row["conversation_id"]) for row in rows]
    trace_ids = [str(value) for value in trace.conversation_ids.tolist()]
    if workload_ids != trace_ids:
        raise ValueError("source trace and workload row order differ")
    atomic_write_jsonl(args.output_workload, rows)
    trace.save(args.output_trace)
    atomic_write_json(
        args.metadata,
        {
            "source_workload": str(args.source_workload),
            "source_trace": str(args.source_trace),
            "source_trace_sha256": source_trace.digest(),
            "output_workload": str(args.output_workload),
            "output_trace": str(args.output_trace),
            "output_trace_sha256": trace.digest(),
            "requests": args.requests,
            "input_tokens": len(rows[0]["input_ids"]),
            "output_tokens": args.output_tokens,
            "conversation_ids_match": True,
        },
    )


if __name__ == "__main__":
    main()
