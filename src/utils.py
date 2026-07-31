"""Shared helpers: text normalization, hashing, IO, longest-common-prefix."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
_MANY_BLANKLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Deterministic, content-preserving normalization.

    Allowed (per spec): CRLF -> LF, strip trailing whitespace per line,
    collapse >2 consecutive blank lines to exactly 2. Forbidden: reordering,
    summarizing, truncation. Content itself is never changed.
    """
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _MANY_BLANKLINES.sub("\n\n", text)
    return text.strip("\n")


def normalize_query_text(text: str) -> str:
    """Normalize a question string for equality comparison (plot detection)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    for tok in ids:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    return h.hexdigest()


# --------------------------------------------------------------------------
# Longest common prefix (over token-id sequences)
# --------------------------------------------------------------------------
def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> int:
    """Return the length of the longest common prefix shared by all sequences."""
    if not sequences:
        return 0
    shortest = min(len(s) for s in sequences)
    for i in range(shortest):
        token = sequences[0][i]
        if any(seq[i] != token for seq in sequences):
            return i
    return shortest


# --------------------------------------------------------------------------
# JSONL / JSON IO
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
