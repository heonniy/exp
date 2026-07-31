"""QMSum test-split loader + preprocessing into the common request schema.

Source repo: Yale-LILY/QMSum. The current upstream layout stores the merged
split under data/ALL/{test,train,val}/*.json (one meeting per file), with an
equivalent data/ALL/jsonl/{split}.jsonl. We use specific_query_list only and
the full serialized meeting transcript as the shared prefix.

One meeting with >=4 specific queries = one prefix group of 4 requests
(first 4 specific queries, original order).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from src.utils import normalize_text

GROUP_SIZE = 4


def find_split_meetings(raw_dir: str | Path, split: str = "test") -> list[Path]:
    """Locate per-meeting json files for the given split.

    Priority:
      1. data/ALL/<split>/*.json         (merged, one meeting per file)
      2. data/ALL/jsonl/<split>.jsonl    (merged jsonl -> materialized rows)
    Raises with a directory listing if neither is found.
    """
    raw_dir = Path(raw_dir)
    all_dir = raw_dir / "data" / "ALL" / split
    if all_dir.is_dir():
        files = sorted(all_dir.glob("*.json"))
        if files:
            return files
    raise FileNotFoundError(
        f"Could not locate QMSum {split} meetings under {raw_dir}/data/ALL/{split}. "
        f"data/ tree: {[str(p) for p in (raw_dir / 'data').glob('*')]}"
    )


def find_split_jsonl(raw_dir: str | Path, split: str = "test") -> Path | None:
    cand = Path(raw_dir) / "data" / "ALL" / "jsonl" / f"{split}.jsonl"
    return cand if cand.exists() else None


def resolve_meeting_id(path: Path) -> str:
    return path.stem


def serialize_transcript(meeting_transcripts: list[dict]) -> str:
    """Deterministic transcript serialization, identical for every query.

    Turn numbering + speaker formatting must be constant across all requests
    that share this meeting so the token-level prefix stays exact.
    """
    lines = []
    for idx, turn in enumerate(meeting_transcripts):
        speaker = (turn.get("speaker") or "").strip()
        content = (turn.get("content") or "").strip()
        lines.append(f"[Turn {idx:04d}] {speaker}: {content}")
    return normalize_text("\n".join(lines))


def _load_meeting(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_requests(
    raw_dir: str | Path, split: str = "test", group_size: int = GROUP_SIZE
) -> Iterator[dict]:
    """Yield common-schema request dicts (+ _status diagnostics)."""
    for path in find_split_meetings(raw_dir, split):
        meeting_id = resolve_meeting_id(path)
        meeting = _load_meeting(path)
        transcript = serialize_transcript(meeting.get("meeting_transcripts", []) or [])
        specific = meeting.get("specific_query_list", []) or []
        topic_list = meeting.get("topic_list", []) or []

        if len(specific) < group_size:
            yield {
                "_status": "excluded",
                "_reason": f"fewer_than_{group_size}_specific_queries_got_{len(specific)}",
                "prefix_id": f"qmsum:{meeting_id}",
            }
            continue

        selected = specific[:group_size]
        for q_idx, q in enumerate(selected):
            answer = (q.get("answer") or "").strip()
            yield {
                "_status": "ok",
                "schema_version": 1,
                "dataset": "qmsum",
                "split": split,
                "prefix_id": f"qmsum:{meeting_id}",
                "request_id": f"qmsum:{meeting_id}:q{q_idx}",
                "request_index_in_group": q_idx,
                "shared_content": transcript,
                "query": (q.get("query") or "").strip(),
                "references": [answer] if answer else [],
                "metadata": {
                    "meeting_id": meeting_id,
                    "query_index": q_idx,
                    "relevant_text_span": q.get("relevant_text_span"),
                    "topic_list": topic_list,
                },
            }
