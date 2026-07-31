"""SQuALITY test-split loader + preprocessing into the common request schema.

Source repo: nyu-mll/SQuALITY, file data/v1-3/test.jsonl.
One story = one prefix group. The plot question ("What is the plot of the
story?") is excluded, leaving exactly 4 focused questions per story.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.utils import normalize_query_text, normalize_text, read_jsonl

PLOT_QUESTION = "what is the plot of the story?"


def find_test_jsonl(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    # Prefer the newest available versioned split. The spec recommends v1-3, but
    # the current upstream HEAD ships v1-1 as the newest; fall back accordingly.
    candidates = [
        raw_dir / "data" / "v1-3" / "test.jsonl",
        raw_dir / "data" / "v1-2" / "test.jsonl",
        raw_dir / "data" / "v1-1" / "test.jsonl",
        raw_dir / "data" / "v1" / "test.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # last-resort: glob
    hits = sorted(raw_dir.glob("data/*/test.jsonl"))
    if hits:
        return hits[-1]
    raise FileNotFoundError(
        f"Could not locate SQuALITY test.jsonl under {raw_dir}. "
        f"Found: {[str(p) for p in raw_dir.rglob('*.jsonl')][:20]}"
    )


def _question_text(q: dict) -> str:
    # v1-1 uses 'question_text'; older/newer variants may use 'question'.
    return (q.get("question_text") or q.get("question") or "").strip()


def _question_num(q: dict):
    return q.get("question_number", q.get("question_num"))


def _response_text(r: dict) -> str:
    if not isinstance(r, dict):
        return ""
    return (r.get("response_text") or r.get("response") or "").strip()


def resolve_story_id(row: dict) -> str:
    meta = row.get("metadata", {}) or {}
    # Prefer a stable, human-meaningful id; passage_id is the Gutenberg id.
    for key in ("passage_id", "uid", "story_id", "id", "gutenberg_id", "set_unique_id"):
        if key in meta and meta[key] not in (None, ""):
            return f"story_{meta[key]}"
    # fallback to a stable field on the row
    for key in ("uid", "id"):
        if key in row and row[key]:
            return f"story_{row[key]}"
    raise KeyError(f"cannot resolve story id from metadata keys {list(meta.keys())}")


def iter_requests(raw_dir: str | Path, split: str = "test") -> Iterator[dict]:
    """Yield common-schema request dicts and per-group diagnostics.

    Yields dicts with an extra "_status" key describing acceptance so the
    caller can build a manifest.
    """
    path = find_test_jsonl(raw_dir)
    for row in read_jsonl(path):
        story_id = resolve_story_id(row)
        document = normalize_text(row.get("document", ""))
        questions = row.get("questions", []) or []

        focused = [
            q
            for q in questions
            if normalize_query_text(_question_text(q)) != PLOT_QUESTION
        ]

        if len(focused) != 4:
            yield {
                "_status": "excluded",
                "_reason": f"expected_4_focused_got_{len(focused)}",
                "prefix_id": f"squality:{story_id}",
            }
            continue

        for q_idx, q in enumerate(focused):
            responses = q.get("responses", []) or []
            refs = [_response_text(r) for r in responses if _response_text(r)]
            yield {
                "_status": "ok",
                "schema_version": 1,
                "dataset": "squality",
                "split": split,
                "prefix_id": f"squality:{story_id}",
                "request_id": f"squality:{story_id}:q{q_idx}",
                "request_index_in_group": q_idx,
                "shared_content": document,
                "query": _question_text(q),
                "references": refs,
                "metadata": {
                    "story_id": story_id,
                    "question_num": _question_num(q),
                },
            }
