from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def provenance(*, trace_sha256: str | None = None) -> dict[str, Any]:
    return {
        "git_sha": git_sha(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_sha256": trace_sha256,
    }


def with_provenance(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    trace_sha256 = result.get("trace_sha256")
    if trace_sha256 is None:
        trace_sha256 = result.get("forced_routing_trace_sha256")
    current = provenance(trace_sha256=trace_sha256)
    result["git_sha"] = current["git_sha"]
    result["command"] = current["command"]
    result["timestamp"] = current["timestamp"]
    result.setdefault("trace_sha256", current["trace_sha256"])
    return result


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = ensure_parent(path)
    value = with_provenance(value)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = ensure_parent(path)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
                handle.write("\n")
                count += 1
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count
