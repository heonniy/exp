#!/usr/bin/env python3
"""Validate exact token-level shared prefixes for each prefix group.

For every group we assert (spec sections 4, 9, 25):
  * group_size matches expectation (default 4)
  * every rendered prompt shares the exact string prefix up to and including
    the "<query>\\n" marker (i.e. the full shared content is common)
  * the token-level longest-common-prefix covers at least the shared region
  * flags context-overflow requests (prompt + max_new_tokens > ctx limit)

Writes:
  outputs/metrics/prefix_validation.csv
  data/manifests/<dataset>_enriched.jsonl   (per-request token stats + hashes)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import MODEL_DIR  # noqa: E402
from src.prompts import QUERY_MARKER, encode_prompt_with_offsets  # noqa: E402
from src.utils import (  # noqa: E402
    longest_common_prefix,
    read_jsonl,
    sha256_ids,
    write_json,
)


def tokens_covering(offsets: list[tuple[int, int]], char_end: int) -> int:
    """Number of leading tokens whose character span ends at or before char_end."""
    count = 0
    for start, end in offsets:
        # special tokens sometimes carry (0,0) offsets; treat by position order
        if end <= char_end:
            count += 1
        else:
            break
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_DIR))
    ap.add_argument("--input", required=True)
    ap.add_argument("--expected-group-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--ctx-limit", type=int, default=40960)
    ap.add_argument("--csv", default="outputs/metrics/prefix_validation.csv")
    ap.add_argument("--enriched", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    requests = list(read_jsonl(args.input))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in requests:
        groups[r["prefix_id"]].append(r)

    # Tokenize every request once (with offsets), caching per request.
    enc_cache: dict[str, dict] = {}
    for r in requests:
        text, ids, offsets = encode_prompt_with_offsets(
            tok, r["shared_content"], r["query"], enable_thinking=False
        )
        marker_pos = text.find(QUERY_MARKER)
        if marker_pos < 0:
            raise RuntimeError(f"query marker not found for {r['request_id']}")
        shared_str_end = marker_pos + len(QUERY_MARKER)
        enc_cache[r["request_id"]] = {
            "text": text,
            "ids": ids,
            "offsets": offsets,
            "shared_str": text[:shared_str_end],
            "shared_str_end": shared_str_end,
            "expected_shared_tokens": tokens_covering(offsets, shared_str_end),
        }

    rows = []
    enriched = []
    all_ok = True
    for prefix_id, members in sorted(groups.items()):
        dataset = members[0]["dataset"]
        members = sorted(members, key=lambda m: m["request_index_in_group"])
        ids_list = [enc_cache[m["request_id"]]["ids"] for m in members]
        lcp = longest_common_prefix(ids_list)

        # string-level shared prefix must be common to all members
        shared_str = enc_cache[members[0]["request_id"]]["shared_str"]
        string_prefix_ok = all(
            enc_cache[m["request_id"]]["text"].startswith(shared_str)
            for m in members
        )
        expected_shared = enc_cache[members[0]["request_id"]]["expected_shared_tokens"]
        # all members should agree on the expected shared-token count
        expected_consistent = all(
            enc_cache[m["request_id"]]["expected_shared_tokens"] == expected_shared
            for m in members
        )

        prompt_lens = [len(x) for x in ids_list]
        lcp_match = (
            string_prefix_ok
            and expected_consistent
            and lcp >= expected_shared
            and len(members) == args.expected_group_size
        )
        all_ok = all_ok and lcp_match

        rows.append(
            {
                "dataset": dataset,
                "prefix_id": prefix_id,
                "group_size": len(members),
                "min_prompt_tokens": min(prompt_lens),
                "max_prompt_tokens": max(prompt_lens),
                "lcp_tokens": lcp,
                "expected_shared_prefix_tokens": expected_shared,
                "string_prefix_ok": string_prefix_ok,
                "lcp_match": lcp_match,
            }
        )

        for m in members:
            e = enc_cache[m["request_id"]]
            ids = e["ids"]
            n_prompt = len(ids)
            shared_tok = e["expected_shared_tokens"]
            overflow = (n_prompt + args.max_new_tokens) > args.ctx_limit
            enriched.append(
                {
                    "request_id": m["request_id"],
                    "prefix_id": prefix_id,
                    "dataset": dataset,
                    "input_token_count": n_prompt,
                    "shared_prefix_token_count": shared_tok,
                    "query_suffix_token_count": n_prompt - shared_tok,
                    "shared_prefix_ratio": round(shared_tok / n_prompt, 6),
                    "group_lcp_tokens": lcp,
                    "prompt_sha256": sha256_ids(ids),
                    "shared_prefix_sha256": sha256_ids(ids[:shared_tok]),
                    "excluded_context_overflow": overflow,
                }
            )

    # write CSV
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    # merge with any existing rows from the other dataset
    existing = []
    csv_path = Path(args.csv)
    if csv_path.exists():
        with csv_path.open() as fh:
            existing = [
                r for r in csv.DictReader(fh) if r["dataset"] != rows[0]["dataset"]
            ]
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
        for r in rows:
            writer.writerow(r)

    # write enriched manifest jsonl
    dataset = rows[0]["dataset"]
    enriched_path = args.enriched or f"data/manifests/{dataset}_enriched.jsonl"
    Path(enriched_path).parent.mkdir(parents=True, exist_ok=True)
    with open(enriched_path, "w") as fh:
        import json

        for e in enriched:
            fh.write(json.dumps(e) + "\n")

    n_overflow = sum(1 for e in enriched if e["excluded_context_overflow"])
    summary = {
        "dataset": dataset,
        "num_groups": len(rows),
        "all_lcp_match": all_ok,
        "num_groups_lcp_match": sum(1 for r in rows if r["lcp_match"]),
        "num_context_overflow_requests": n_overflow,
        "prompt_token_min": min(e["input_token_count"] for e in enriched),
        "prompt_token_max": max(e["input_token_count"] for e in enriched),
    }
    write_json(f"data/manifests/{dataset}_validation_summary.json", summary)
    print(f"[validate:{dataset}] {summary}")
    if not all_ok:
        print(f"[validate:{dataset}] WARNING: not all groups passed lcp_match")
        sys.exit(2)
    print(f"[validate:{dataset}] VALIDATE_OK")


if __name__ == "__main__":
    main()
